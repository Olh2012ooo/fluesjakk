"""Dashboard server (docs/SPEC.md §10).

* ``GET /``                            static ``index.html``
* ``GET /static/...``                  the vanilla-JS UI (no build step, no CDN)
* ``GET /api/runs``                    ``list_runs(runs_dir)``
* ``GET /api/default``                 ``{"run": <name or null>}`` — run the CLI asked to open
* ``GET /api/run/{name}``              ``{"run": run.json, "metrics": [history], "offset": <byte cursor>}``
* ``GET /api/run/{name}/positions``    normalised 3-D positions of the sampled neurons (from the graph npz)
* ``WS  /ws/{name}[?after=N]``         ``{"kind": "init", "run": ..., "metrics": [...], "offset": N}`` then every
                                       new record, followed by ``{"kind": "cursor", "offset": N}`` after each batch

The *history* sent on init is per kind (see :class:`RunIndex`): the complete ``train`` / ``eval`` / ``elo`` series
(train capped at ``INIT_MAX_TRAIN`` points by bucket averaging), the last ``game`` and ``activity`` record and the
last ``INIT_STATUS_LINES`` ``status`` / ``error`` records — so a reload shows the whole run, not a fixed tail. The
index is built once per run and then only parses the bytes appended since (``offset``).

A client that reconnects with ``?after=N`` (the cursor it last received for the *same* run) gets
``{"kind": "resume", "offset": N}`` instead of an init and the tail continues from ``N`` — the history it already
holds is kept. A missing or stale cursor (``N > size``, or the file was truncated) falls back to a full init.

The websocket tails ``metrics.jsonl`` by polling its size every ``POLL_INTERVAL`` seconds; if the run does not
exist yet it simply waits for it to appear. If the file shrinks (rewritten run) a fresh init is sent.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import webbrowser
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from flychess import paths
from flychess.train.metrics import METRICS_FILE, list_runs, read_metrics, read_run_json

log = logging.getLogger("flychess.dashboard")

STATIC_DIR = Path(__file__).resolve().parent / "static"
INIT_STATUS_LINES = 200    # status / error records kept for the init (only used for the log tail + status bar)
INIT_MAX_TRAIN = 20000     # train records sent on init; longer series are bucket-averaged down to this many
POLL_INTERVAL = 0.5        # seconds between metrics.jsonl size checks
MAX_POSITION_SAMPLES = 4096


# ------------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------------
def _safe_name(name: str) -> str:
    """Reject path traversal in run names."""
    if not name or name in {".", ".."} or "/" in name or "\\" in name or name.startswith("."):
        raise HTTPException(status_code=400, detail="invalid run name")
    return name


def _resolve_graph_path(run_dir: Path, info: dict | None) -> Path | None:
    info = info or {}
    gp = info.get("graph_path") or (info.get("config") or {}).get("graph_path")
    if not gp:
        return None
    p = Path(gp).expanduser()
    for candidate in (p, run_dir / p, paths.HOME / p, paths.REPO_ROOT / p):
        if candidate.is_file():
            return candidate
    return None


def _neuron_positions(graph_path: Path, neuron_idx: np.ndarray | None) -> dict:
    """Load only ``position`` (+ ``super_class``) from a BrainGraph npz and normalise to [0, 1] per axis."""
    with np.load(graph_path, allow_pickle=False) as z:
        pos = np.asarray(z["position"], dtype=np.float32)
        sc = z["super_class"].astype(str) if "super_class" in z.files else None
    n = int(pos.shape[0])
    finite = np.isfinite(pos).all(axis=1)
    lo = np.nanmin(pos[finite], axis=0) if finite.any() else np.zeros(3, np.float32)
    hi = np.nanmax(pos[finite], axis=0) if finite.any() else np.ones(3, np.float32)
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    norm = (pos - lo) / span
    if neuron_idx is None:
        k = min(n, MAX_POSITION_SAMPLES)
        neuron_idx = np.linspace(0, n - 1, k).astype(np.int64) if k > 0 else np.zeros(0, np.int64)
    neuron_idx = np.asarray(neuron_idx, dtype=np.int64)
    neuron_idx = neuron_idx[(neuron_idx >= 0) & (neuron_idx < n)]
    sel = norm[neuron_idx]
    sel_list: list[list[float] | None] = [
        [float(x), float(y), float(zz)] if np.isfinite(row).all() else None for (x, y, zz), row in zip(sel, sel)
    ]
    return {
        "n": n,
        "neuron_idx": neuron_idx.tolist(),
        "positions": sel_list,
        "super_class": sc[neuron_idx].tolist() if sc is not None else None,
        "graph_path": str(graph_path),
    }


def _read_new_lines(path: Path, offset: int) -> tuple[list[bytes], int]:
    """Read complete lines appended after ``offset``. Returns (lines, new_offset) — a torn last line is left."""
    with open(path, "rb") as f:
        f.seek(offset)
        data = f.read()
    if not data:
        return [], offset
    end = data.rfind(b"\n")
    if end < 0:
        return [], offset
    complete = data[: end + 1]
    return [ln for ln in complete.split(b"\n") if ln.strip()], offset + len(complete)


def _bucket_mean(records: list[dict], max_points: int) -> list[dict]:
    """Equal-count bucket means of ``records`` (numeric fields averaged, ``step`` / non-numeric fields from the
    last record of the bucket) so that at most ``max_points`` remain; the last raw record is always kept."""
    n = len(records)
    if n <= max_points:
        return records
    per = -(-n // max_points)
    out: list[dict] = []
    for i in range(0, n, per):
        chunk = records[i : i + per]
        agg = dict(chunk[-1])
        for key, last in list(agg.items()):
            if key in ("step", "t", "kind") or isinstance(last, bool):
                continue
            vals = [r.get(key) for r in chunk]
            if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
                agg[key] = sum(vals) / len(vals)
        out.append(agg)
    out[-1] = records[-1]  # the status bar reads gpu_mem_gb / pos_per_sec / epoch from the true last record
    return out


class RunIndex:
    """Per-kind, incrementally maintained view of one run's ``metrics.jsonl`` (what the UI needs on load).

    ``update()`` parses only the bytes appended since the previous call (a full pass on the first call, or after
    the file shrank) and ``snapshot()`` returns the records in file order together with the byte ``offset`` they
    were built up to, so a websocket tail can continue exactly from there. Thread-safe (called via ``to_thread``).
    """

    def __init__(self, path: Path, status_lines: int = INIT_STATUS_LINES, max_train: int = INIT_MAX_TRAIN) -> None:
        self.path = path
        self.status_lines = status_lines
        self.max_train = max_train
        self._lock = threading.Lock()
        self._reset()

    def _reset(self) -> None:
        self.offset = 0
        self._seq = 0
        self.train: list[tuple[int, dict]] = []
        self.eval: list[tuple[int, dict]] = []
        self.elo: list[tuple[int, dict]] = []
        self.status: deque[tuple[int, dict]] = deque(maxlen=self.status_lines)
        self.last_game: tuple[int, dict] | None = None
        self.last_activity: tuple[int, dict] | None = None
        self.counts: dict[str, int] = {}

    def _add(self, rec: dict) -> None:
        kind = rec.get("kind")
        if not isinstance(kind, str):
            return
        self._seq += 1
        item = (self._seq, rec)
        self.counts[kind] = self.counts.get(kind, 0) + 1
        if kind == "train":
            self.train.append(item)
        elif kind == "eval":
            self.eval.append(item)
        elif kind == "elo":
            self.elo.append(item)
        elif kind == "game":
            self.last_game = item
        elif kind == "activity":
            self.last_activity = item
        else:  # status, error and anything unknown: log-tail material only
            self.status.append(item)

    def update(self) -> None:
        """Parse the lines appended since the last update (or everything, if the file shrank / is new)."""
        with self._lock:
            try:
                size = self.path.stat().st_size
            except FileNotFoundError:
                self._reset()
                return
            if size < self.offset:
                self._reset()
            if size == self.offset:
                return
            lines, self.offset = _read_new_lines(self.path, self.offset)
            for line in lines:
                try:
                    rec = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if isinstance(rec, dict):
                    self._add(rec)

    def snapshot(self) -> tuple[list[dict], int, dict[str, int]]:
        """``(records in file order, offset, counts per kind)`` for the init payload."""
        with self._lock:
            items: list[tuple[int, dict]] = list(self.eval) + list(self.elo) + list(self.status)
            n = len(self.train)
            if n > self.max_train:
                per = -(-n // self.max_train)
                # one averaged record per bucket, ordered by the sequence number of the bucket's last record
                seqs = [self.train[min(n - 1, i + per - 1)][0] for i in range(0, n, per)]
                items += list(zip(seqs, _bucket_mean([r for _, r in self.train], self.max_train), strict=True))
            else:
                items += self.train
            for extra in (self.last_game, self.last_activity):
                if extra is not None:
                    items.append(extra)
            items.sort(key=lambda it: it[0])
            return [r for _, r in items], self.offset, dict(self.counts)

    def load(self) -> tuple[list[dict], int, dict[str, int]]:
        self.update()
        return self.snapshot()


# ------------------------------------------------------------------------------------------------
# app factory
# ------------------------------------------------------------------------------------------------
def create_app(
    runs_dir: str | os.PathLike | None = None,
    default_run: str | None = None,
    *,
    init_max_train: int = INIT_MAX_TRAIN,
    init_status_lines: int = INIT_STATUS_LINES,
) -> FastAPI:
    runs_root = Path(runs_dir) if runs_dir is not None else paths.RUNS_DIR
    app = FastAPI(title="fly-chess dashboard", docs_url=None, redoc_url=None)
    app.state.runs_dir = runs_root
    app.state.default_run = default_run
    positions_cache: dict[tuple[str, tuple[int, ...]], dict] = {}
    indexes: dict[str, RunIndex] = {}
    indexes_lock = threading.Lock()

    def index_for(run_dir: Path) -> RunIndex:
        with indexes_lock:
            idx = indexes.get(run_dir.name)
            if idx is None:
                idx = indexes[run_dir.name] = RunIndex(
                    run_dir / METRICS_FILE, status_lines=init_status_lines, max_train=init_max_train
                )
            return idx

    def load_init(run_dir: Path) -> dict:
        """run.json + per-kind history + the byte ``offset`` the history was built up to (one consistent view)."""
        metrics, offset, counts = index_for(run_dir).load()
        return {"run": read_run_json(run_dir) or {}, "metrics": metrics, "offset": offset, "counts": counts}

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html")

    @app.get("/api/runs")
    async def api_runs() -> list[dict]:
        return await asyncio.to_thread(list_runs, runs_root)

    @app.get("/api/default")
    async def api_default() -> dict:
        return {"run": app.state.default_run, "runs_dir": str(runs_root)}

    @app.get("/api/run/{name}")
    async def api_run(name: str) -> dict:
        run_dir = runs_root / _safe_name(name)
        if not run_dir.is_dir():
            raise HTTPException(status_code=404, detail=f"run {name!r} not found")
        return await asyncio.to_thread(load_init, run_dir)

    @app.get("/api/run/{name}/positions")
    async def api_positions(name: str) -> JSONResponse:
        run_dir = runs_root / _safe_name(name)
        if not run_dir.is_dir():
            raise HTTPException(status_code=404, detail=f"run {name!r} not found")
        info = read_run_json(run_dir)
        graph_path = _resolve_graph_path(run_dir, info)
        if graph_path is None:
            return JSONResponse({"available": False, "reason": "no graph_path in run.json (or file missing)"})
        # positions of the neurons in the latest activity sample, else an even sample of the graph
        latest = await asyncio.to_thread(read_metrics, run_dir, 1, ["activity"])
        idx = latest[0].get("neuron_idx") if latest else None
        idx_arr = np.asarray(idx, dtype=np.int64) if idx else None
        key = (str(graph_path), tuple(idx_arr.tolist()) if idx_arr is not None else ())
        if key not in positions_cache:
            try:
                positions_cache.clear()  # keep at most one entry (activity samples are usually fixed per run)
                positions_cache[key] = await asyncio.to_thread(_neuron_positions, graph_path, idx_arr)
            except Exception as exc:  # noqa: BLE001 - corrupt / foreign npz must not take the dashboard down
                log.warning("cannot read positions from %s: %s", graph_path, exc)
                return JSONResponse({"available": False, "reason": str(exc)})
        return JSONResponse({"available": True, **positions_cache[key]})

    @app.websocket("/ws/{name}")
    async def ws_run(websocket: WebSocket, name: str, after: int | None = None) -> None:
        await websocket.accept()
        try:
            run_dir = runs_root / _safe_name(name)
        except HTTPException:
            await websocket.close(code=1008)
            return
        metrics_path = run_dir / METRICS_FILE

        async def watch_client() -> None:
            # Pump incoming frames so that a client disconnect raises here and cancels the tail loop.
            while True:
                await websocket.receive_text()

        async def send_init() -> int:
            payload = await asyncio.to_thread(load_init, run_dir)
            await websocket.send_text(json.dumps({"kind": "init", **payload}))
            return payload["offset"]

        async def tail() -> None:
            # wait for the run to appear (the dashboard may be started before training)
            while not metrics_path.exists():
                await asyncio.sleep(POLL_INTERVAL)
            resumable = after is not None and 0 <= after <= metrics_path.stat().st_size
            if resumable:  # the client still holds the history up to `after`: skip the init
                offset = after
                await websocket.send_text(json.dumps({"kind": "resume", "offset": offset}))
            else:
                offset = await send_init()
            while True:
                await asyncio.sleep(POLL_INTERVAL)
                try:
                    size = metrics_path.stat().st_size
                except FileNotFoundError:
                    continue
                if size < offset:  # truncated / rewritten: rebuild the client from scratch
                    offset = await send_init()
                    continue
                if size == offset:
                    continue
                lines, offset = await asyncio.to_thread(_read_new_lines, metrics_path, offset)
                sent = 0
                for line in lines:
                    try:
                        json.loads(line)  # only forward valid JSON objects
                    except json.JSONDecodeError:
                        continue
                    await websocket.send_text(line.decode("utf-8", errors="replace"))
                    sent += 1
                if sent:
                    await websocket.send_text(json.dumps({"kind": "cursor", "offset": offset}))

        watcher = asyncio.create_task(watch_client())
        tailer = asyncio.create_task(tail())
        try:
            done, pending = await asyncio.wait({watcher, tailer}, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            for task in done:
                exc = task.exception()
                if exc is not None and not isinstance(exc, (WebSocketDisconnect, RuntimeError)):
                    log.warning("websocket /ws/%s ended: %r", name, exc)
        except asyncio.CancelledError:
            watcher.cancel()
            tailer.cancel()
            raise

    return app


# ------------------------------------------------------------------------------------------------
# entry points
# ------------------------------------------------------------------------------------------------
def serve(
    run: str | None = None,
    port: int = 8765,
    open_browser: bool = False,
    host: str = "127.0.0.1",
    runs_dir: str | os.PathLike | None = None,
) -> None:
    """Run the dashboard with uvicorn (blocking). ``run`` selects the run the UI opens by default."""
    import uvicorn

    app = create_app(runs_dir=runs_dir, default_run=run)
    url = f"http://{host}:{port}/"
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")


def run_dashboard(run: str | None = None, port: int = 8765, open_browser: bool = False, **kwargs: Any) -> None:
    """CLI entry (``fly dashboard [--run NAME] [--port PORT]``)."""
    print(f"fly-chess dashboard: http://127.0.0.1:{port}/" + (f"  (run: {run})" if run else ""))
    serve(run=run, port=port, open_browser=open_browser, **kwargs)


__all__ = [
    "INIT_MAX_TRAIN",
    "INIT_STATUS_LINES",
    "POLL_INTERVAL",
    "STATIC_DIR",
    "RunIndex",
    "create_app",
    "run_dashboard",
    "serve",
]
