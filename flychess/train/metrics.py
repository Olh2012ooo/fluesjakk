"""MetricsLogger — append-only JSONL metrics for a training run (docs/SPEC.md §7).

Layout of ``runs/<run>/``:

* ``metrics.jsonl`` — one JSON object per line, always carrying ``t`` (unix time), ``step`` and ``kind``.
* ``run.json``      — static description of the run (config, graph meta, ``started_at``).

The dashboard (``flychess.dashboard.server``) tails ``metrics.jsonl``; everything written here must therefore be
a *complete* line per write (we flush after every record) and must be plain JSON (NaN/Inf are turned into
``null`` because ``JSON.parse`` in the browser rejects them).
"""
from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Self

import numpy as np

METRICS_FILE = "metrics.jsonl"
RUN_FILE = "run.json"

#: Minimum spacing between consecutive records of a rate-limited kind (seconds).
RATE_LIMITS: dict[str, float] = {"game": 60.0, "activity": 30.0}


# ------------------------------------------------------------------------------------------------
# JSON helpers
# ------------------------------------------------------------------------------------------------
def _jsonable(obj: Any) -> Any:
    """Recursively convert numpy / torch scalars & arrays to JSON-serialisable values; non-finite -> None."""
    if isinstance(obj, (str, bool)) or obj is None:
        return obj
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "detach") and hasattr(obj, "tolist"):  # torch.Tensor without importing torch
        return _jsonable(obj.detach().cpu().tolist())
    if hasattr(obj, "item"):  # 0-d tensor-likes
        return _jsonable(obj.item())
    if hasattr(obj, "__dataclass_fields__"):
        return _jsonable({k: getattr(obj, k) for k in obj.__dataclass_fields__})
    return str(obj)


def dumps(record: dict) -> str:
    return json.dumps(_jsonable(record), separators=(",", ":"), ensure_ascii=False)


# ------------------------------------------------------------------------------------------------
# Logger
# ------------------------------------------------------------------------------------------------
class MetricsLogger:
    """Append metrics records to ``<run_dir>/metrics.jsonl`` and write ``<run_dir>/run.json``.

    Every ``log*`` call writes exactly one line and flushes it so that a tailing dashboard sees it immediately.
    ``log_game`` and ``log_activity`` are rate limited (≤ 1/min and ≤ 1/30 s) and return ``False`` when dropped.
    """

    def __init__(self, run_dir: str | os.PathLike, rate_limits: dict[str, float] | None = None) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / METRICS_FILE
        self.run_json_path = self.run_dir / RUN_FILE
        self.rate_limits = dict(RATE_LIMITS if rate_limits is None else rate_limits)
        self._last_write: dict[str, float] = {}
        self._file = open(self.metrics_path, "a", encoding="utf-8")  # noqa: SIM115 - long-lived handle

    # ---- core --------------------------------------------------------------------------------
    def log(self, kind: str, step: int, *, force: bool = False, **fields: Any) -> dict | None:
        """Append one record. Returns the record written, or ``None`` if it was rate limited (``force`` writes it
        regardless of the limit and restarts the interval)."""
        now = time.time()
        limit = self.rate_limits.get(kind)
        if limit is not None and not force:
            last = self._last_write.get(kind)
            if last is not None and now - last < limit:
                return None
        if limit is not None:
            self._last_write[kind] = now
        record = {"t": now, "step": int(step), "kind": kind, **fields}
        self._file.write(dumps(record) + "\n")
        self._file.flush()
        return record

    # ---- convenience (field names per SPEC §7) ------------------------------------------------
    def log_train(
        self,
        step: int,
        *,
        loss: float,
        policy_loss: float,
        value_loss: float,
        top1: float,
        top3: float,
        lr: float,
        pos_per_sec: float,
        gpu_mem_gb: float,
        epoch: float,
        stage: str,
        **extra: Any,
    ) -> dict | None:
        return self.log(
            "train", step, loss=loss, policy_loss=policy_loss, value_loss=value_loss, top1=top1, top3=top3,
            lr=lr, pos_per_sec=pos_per_sec, gpu_mem_gb=gpu_mem_gb, epoch=epoch, stage=stage, **extra,
        )

    def log_eval(
        self, step: int, *, val_loss: float, val_top1: float, val_top3: float, val_value_mse: float, **extra: Any
    ) -> dict | None:
        return self.log(
            "eval", step, val_loss=val_loss, val_top1=val_top1, val_top3=val_top3, val_value_mse=val_value_mse,
            **extra,
        )

    def log_elo(
        self, step: int, *, opponent: str, games: int, wins: int, draws: int, losses: int, elo_estimate: float,
        **extra: Any,
    ) -> dict | None:
        return self.log(
            "elo", step, opponent=opponent, games=games, wins=wins, draws=draws, losses=losses,
            elo_estimate=elo_estimate, **extra,
        )

    def log_game(self, step: int, *, pgn: str, result: str, moves: int, source: str, force: bool = False,
                 **extra: Any) -> bool:
        """Sample game (≤ 1 per minute unless ``force``). Returns ``True`` if written."""
        return self.log("game", step, pgn=pgn, result=result, moves=moves, source=source, force=force,
                        **extra) is not None

    def log_activity(self, step: int, *, neuron_idx: Iterable[int], values: Iterable[float], **extra: Any) -> bool:
        """Sampled neuron activity (≤ 1 per 30 s). Returns ``True`` if written."""
        return self.log("activity", step, neuron_idx=neuron_idx, values=values, **extra) is not None

    def log_status(
        self, step: int, *, message: str, stage: str, total_steps: int, eta_s: float | None, **extra: Any
    ) -> dict | None:
        return self.log(
            "status", step, message=message, stage=stage, total_steps=total_steps, eta_s=eta_s, **extra
        )

    # ---- run.json ------------------------------------------------------------------------------
    def write_run_json(self, config: dict, graph_meta: dict, extra: dict | None = None) -> dict:
        """Write ``run.json``. Idempotent: an existing ``started_at`` is preserved, other fields are refreshed."""
        previous = read_run_json(self.run_dir) or {}
        info = {
            "name": self.run_dir.name,
            "started_at": previous.get("started_at", time.time()),
            "updated_at": time.time(),
            "config": config,
            "graph_meta": graph_meta,
        }
        if "graph_path" in previous:
            info["graph_path"] = previous["graph_path"]
        if extra:
            info.update(extra)
        if "graph_path" not in info:
            gp = config.get("graph_path") or (config.get("brain") or {}).get("graph_path")
            if gp:
                info["graph_path"] = gp
        tmp = self.run_json_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(_jsonable(info), indent=2), encoding="utf-8")
        os.replace(tmp, self.run_json_path)
        return info

    # ---- lifecycle -----------------------------------------------------------------------------
    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - best effort
        try:
            self.close()
        except Exception:  # noqa: BLE001, S110 - interpreter may be tearing down
            pass


# ------------------------------------------------------------------------------------------------
# Readers
# ------------------------------------------------------------------------------------------------
def read_run_json(run_dir: str | os.PathLike) -> dict | None:
    path = Path(run_dir) / RUN_FILE
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _parse_line(line: bytes | str) -> dict | None:
    if not line or (isinstance(line, bytes) and not line.strip()) or (isinstance(line, str) and not line.strip()):
        return None
    try:
        rec = json.loads(line)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None  # torn/partial line (writer mid-write) or garbage
    return rec if isinstance(rec, dict) else None


def iter_lines_reversed(path: str | os.PathLike, block_size: int = 1 << 16) -> Iterator[bytes]:
    """Yield the lines of ``path`` from the last to the first, reading blocks from the end (no full read)."""
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        pos = f.tell()
        remainder = b""
        while pos > 0:
            size = min(block_size, pos)
            pos -= size
            f.seek(pos)
            chunk = f.read(size) + remainder
            lines = chunk.split(b"\n")
            remainder = lines[0]
            for line in reversed(lines[1:]):
                if line:
                    yield line
        if remainder:
            yield remainder


def read_metrics(
    run_dir: str | os.PathLike, last_n: int | None = None, kinds: Iterable[str] | None = None
) -> list[dict]:
    """Read records from ``<run_dir>/metrics.jsonl`` in chronological order.

    ``last_n`` keeps only the last *n* (matching) records and is read from the end of the file, so it stays
    cheap on huge logs. ``kinds`` filters by the ``kind`` field.
    """
    path = Path(run_dir) / METRICS_FILE
    if not path.exists():
        return []
    kind_set = set(kinds) if kinds is not None else None
    if last_n is None:
        out: list[dict] = []
        with open(path, "rb") as f:
            for line in f:
                rec = _parse_line(line)
                if rec is not None and (kind_set is None or rec.get("kind") in kind_set):
                    out.append(rec)
        return out
    if last_n <= 0:
        return []
    out = []
    for line in iter_lines_reversed(path):
        rec = _parse_line(line)
        if rec is None or (kind_set is not None and rec.get("kind") not in kind_set):
            continue
        out.append(rec)
        if len(out) >= last_n:
            break
    out.reverse()
    return out


def _tail_summary(run_path: Path, max_scan: int = 500) -> dict:
    """Last step / time / stage of a run, scanning at most ``max_scan`` lines from the end of metrics.jsonl."""
    summary: dict[str, Any] = {"last_step": None, "last_t": None, "stage": None}
    path = run_path / METRICS_FILE
    if not path.exists():
        return summary
    scanned = 0
    for line in iter_lines_reversed(path):
        scanned += 1
        rec = _parse_line(line)
        if rec is None:
            if scanned >= max_scan:
                break
            continue
        if summary["last_step"] is None:
            summary["last_step"] = rec.get("step")
            summary["last_t"] = rec.get("t")
        if summary["stage"] is None and rec.get("stage"):
            summary["stage"] = rec["stage"]
        if summary["stage"] is not None or scanned >= max_scan:
            break
    return summary


def list_runs(runs_dir: str | os.PathLike) -> list[dict]:
    """List runs under ``runs_dir`` (directories with a ``run.json`` or ``metrics.jsonl``), newest activity first.

    Each entry: ``{name, started_at, last_step, stage, last_t}``.
    """
    root = Path(runs_dir)
    if not root.is_dir():
        return []
    runs: list[dict] = []
    for run_path in root.iterdir():
        if not run_path.is_dir():
            continue
        info = read_run_json(run_path)
        if info is None and not (run_path / METRICS_FILE).exists():
            continue
        entry = {"name": run_path.name, "started_at": (info or {}).get("started_at")}
        entry.update(_tail_summary(run_path))
        if entry["stage"] is None and info:
            entry["stage"] = (info.get("config") or {}).get("stage")
        runs.append(entry)
    runs.sort(key=lambda r: (r["last_t"] or r["started_at"] or 0.0), reverse=True)
    return runs


__all__ = [
    "METRICS_FILE",
    "RATE_LIMITS",
    "RUN_FILE",
    "MetricsLogger",
    "dumps",
    "iter_lines_reversed",
    "list_runs",
    "read_metrics",
    "read_run_json",
]
