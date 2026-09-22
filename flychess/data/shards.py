"""Shard format (SPEC §5) and the torch dataset that streams it.

A shard ``<name>-NNNNN.npz`` holds ``S`` positions:
    planes: uint8 (S, 20, 8, 8)   planes 0-17 and 19 are 0/1, plane 18 is (min(halfmove,100)*255+50)//100
    move:   int16 (S,)            move index (SPEC §3.3), mover's perspective
    value:  int8  (S,)            +1 / 0 / -1 from the mover's perspective (game outcome), or a quantised
                                  engine evaluation ``round(v * value_scale)`` (``flychess.data.evals``)
    elo:    int16 (S,)            the mover's Elo (0 when unknown, e.g. engine-evaluated positions)
    ply:    int16 (S,)            0-based ply of the position (number of half-moves played before it)
    value_scale: scalar, optional 1 when missing (old shards); ``value / value_scale`` is the float target in
                                  [-1, 1] (eval shards use 127)
``collate`` rescales plane 18 back to [0, 1], flattens to ``float32 (B, 1280)`` and divides ``value`` by
``value_scale`` (``float32``).

A build with a game-level hold-out (``build_shards(..., val_every=k)``) writes a second series
``<name>.val-NNNNN.npz`` holding every position of one game in ``k``; ``list_shards`` never mixes the two
series (train shards are ``<name>-NNNNN.npz`` only) and ``load_split`` returns the ``.val`` series as the
validation set when it exists.
"""
from __future__ import annotations

import os
import re
import warnings
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from ..chessenv.encoding import FLAT_INPUT, NUM_PLANES

SHARD_SIZE = 262144
HALFMOVE_SCALE = 255.0
SHARD_KEYS = ("planes", "move", "value", "elo", "ply")
VALUE_SCALE_KEY = "value_scale"   # optional scalar npz key; missing = 1 (old shards store the outcome +1/0/-1)
VAL_SUFFIX = ".val"   # the game-level validation series of <name> is named <name>.val
_SHARD_RE = re.compile(r"^(?P<series>.+)-(?P<index>\d{5,})\.npz$")


# ---------------------------------------------------------------------------------------------
# Shard files
# ---------------------------------------------------------------------------------------------
def shard_path(out_dir: str | Path, name: str, index: int) -> Path:
    return Path(out_dir) / f"{name}-{index:05d}.npz"


def write_shard(path: str | Path, planes: np.ndarray, move: np.ndarray, value: np.ndarray,
                elo: np.ndarray, ply: np.ndarray, value_scale: float = 1.0) -> Path:
    """Write one shard atomically (``np.savez_compressed`` into a temp file, then rename).

    ``value`` is stored as int8; ``value_scale != 1`` records that the float target is ``value / value_scale``
    (a scalar ``value_scale`` key is written only then, so outcome shards keep the old five-array layout).
    """
    path = Path(path)
    n = planes.shape[0]
    assert planes.shape == (n, NUM_PLANES, 8, 8) and planes.dtype == np.uint8, planes.shape
    assert move.shape == (n,) and value.shape == (n,) and elo.shape == (n,) and ply.shape == (n,)
    if not value_scale > 0:
        raise ValueError(f"value_scale must be > 0, got {value_scale!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    arrays = {
        "planes": planes,
        "move": move.astype(np.int16),
        "value": value.astype(np.int8),
        "elo": elo.astype(np.int16),
        "ply": ply.astype(np.int16),
    }
    if value_scale != 1.0:
        arrays[VALUE_SCALE_KEY] = np.float32(value_scale)
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)
    return path


def read_shard(path: str | Path) -> dict[str, np.ndarray]:
    """The five arrays of a shard plus ``'value_scale'`` (a python float, 1.0 when the file has none)."""
    with np.load(Path(path), allow_pickle=False) as z:
        out = {k: z[k] for k in SHARD_KEYS}
        out[VALUE_SCALE_KEY] = read_value_scale(z)
        return out


def read_value_scale(npz) -> float:
    """``value_scale`` of an open ``np.load`` handle (1.0 when the key is absent: old outcome shards)."""
    return float(npz[VALUE_SCALE_KEY]) if VALUE_SCALE_KEY in npz.files else 1.0


def val_name(name: str) -> str:
    """Series name of the game-level validation shards of ``name`` (``<name>.val``)."""
    return f"{name}{VAL_SUFFIX}"


def shard_series(path: str | Path) -> str | None:
    """``<series>`` of a ``<series>-NNNNN.npz`` shard file name, None if the name is not a shard."""
    m = _SHARD_RE.match(Path(path).name)
    return m.group("series") if m else None


def list_shards(shard_dir: str | Path, name: str | None = None) -> list[Path]:
    """Sorted *training* shard files ``<series>-NNNNN.npz`` in ``shard_dir`` (only series ``name`` if given).

    Validation series (``<name>.val-NNNNN.npz``, see :func:`list_val_shards`) are never returned here, so a
    ``name=None`` / ``*.npz`` caller cannot sweep the hold-out games back into training.
    """
    return _list_series(shard_dir, name, val=False)


def list_val_shards(shard_dir: str | Path, name: str | None = None) -> list[Path]:
    """Sorted game-level validation shards ``<name>.val-NNNNN.npz`` (every ``.val`` series if ``name`` is None)."""
    return _list_series(shard_dir, name, val=True)


def _list_series(shard_dir: str | Path, name: str | None, val: bool) -> list[Path]:
    out = []
    for p in Path(shard_dir).glob("*.npz"):
        series = shard_series(p)
        if series is None:
            continue
        if name is not None:
            if series != (val_name(name) if val else name):
                continue
        elif series.endswith(VAL_SUFFIX) != val:
            continue
        out.append(p)
    return sorted(out)


def count_positions(shard_dir_or_files: str | Path | Sequence[str | Path]) -> int:
    """Total number of positions across shard files (reads only the small ``move`` array of each)."""
    files = _as_files(shard_dir_or_files)
    total = 0
    for f in files:
        with np.load(f, allow_pickle=False) as z:
            total += int(z["move"].shape[0])
    return total


def parse_shard_names(spec: str | Sequence[str | tuple[str, int]] | None) -> list[tuple[str, int]] | None:
    """``'lichess2014,lichess2015,evals:3'`` -> ``[('lichess2014', 1), ('lichess2015', 1), ('evals', 3)]``.

    A comma-separated list of shard series names, each with an optional ``:k`` repeat factor (``k >= 1``, an
    integer: the series' training shards are listed ``k`` times, which oversamples it ``k``-fold; validation
    shards are never repeated).  Whitespace around names is ignored; ``None`` / ``''`` -> ``None`` (every
    series).  A list of names / ``(name, repeat)`` pairs is normalised the same way.
    """
    if spec is None:
        return None
    if isinstance(spec, str):
        items: list = [x for x in spec.split(",") if x.strip()]
    else:
        items = list(spec)
    if not items:
        return None
    out: list[tuple[str, int]] = []
    for item in items:
        if isinstance(item, str):
            name, _, rep = item.strip().partition(":")
            try:
                repeat = int(rep) if rep else 1
            except ValueError:
                raise ValueError(f"bad repeat factor in shard name {item!r} (expected '<name>:<int>')") from None
        else:
            name, repeat = str(item[0]).strip(), int(item[1])
        if not name:
            raise ValueError(f"empty shard series name in {spec!r}")
        if repeat < 1:
            raise ValueError(f"repeat factor of shard series {name!r} must be >= 1, got {repeat}")
        out.append((name, repeat))
    return out


def load_split(shard_dir: str | Path, val_fraction: float = 0.05, seed: int = 0,
               name: str | Sequence[str | tuple[str, int]] | None = None) -> tuple[list[Path], list[Path]]:
    """Deterministically split shard files into ``(train_files, val_files)``.

    ``name`` selects the shard series: ``None`` (every series), one series name, or several with repeat
    factors — a ``'lichess2014,lichess2015,evals:3'`` string (see :func:`parse_shard_names`) or the parsed
    ``[(name, repeat), ...]`` list.  Each series is split on its own (below) and the training lists are
    concatenated with a series repeated ``repeat`` times (``ShardDataset`` shuffles shard order, so a
    ``k``-fold repeat samples that series ``k`` times as often and ``count_positions`` counts it ``k`` times);
    validation files are never repeated.  Every listed series must exist.

    Per series: if the build wrote a game-level hold-out series (``<name>.val-NNNNN.npz``,
    ``build_shards(val_every=k)`` / eval shards' FEN-hash split) that series *is* the validation set
    (``val_fraction`` is ignored) and no validation game has a position in the training shards.  Otherwise
    the files are sorted, permuted with ``seed`` and the last ``round(val_fraction * n)`` (at least one when
    ``val_fraction > 0`` and there are >= 2 shards) become the validation set — a *shard*-level split: shards
    are shuffled at the position level, so positions of the same game then sit in both sets (a warning says so).
    """
    series = parse_shard_names(name)
    if series is None or (len(series) == 1 and series[0][1] == 1):
        return _split_series(shard_dir, val_fraction, seed, series[0][0] if series else None)
    train_files: list[Path] = []
    val_files: list[Path] = []
    for series_name, repeat in series:
        tr, va = _split_series(shard_dir, val_fraction, seed, series_name)
        train_files.extend(tr * repeat)
        val_files.extend(va)
    return train_files, val_files


def _split_series(shard_dir: str | Path, val_fraction: float, seed: int,
                  name: str | None) -> tuple[list[Path], list[Path]]:
    files = list_shards(shard_dir, name)
    if not files:
        pattern = f"{name}-*.npz" if name else "*.npz"
        msg = f"no shards matching {pattern} in {shard_dir}"
        if name:
            n_all = len(list_shards(shard_dir))
            msg += (f" ({n_all} other *.npz shard file(s) are there; the shard_name filter {name!r} excludes them:"
                    " pass --shard-name / --set shard_name=null or fix the config)" if n_all else
                    " (shard_name filter: pass --shard-name / --set shard_name=null or fix the config)")
        raise FileNotFoundError(msg)
    val_files = list_val_shards(shard_dir, name)
    if val_files:
        return files, val_files
    if val_fraction > 0 and len(files) >= 2:
        warnings.warn(
            f"no game-level validation series ({(name or '<name>') + VAL_SUFFIX}-NNNNN.npz) in {shard_dir}: "
            "holding out whole shards instead, which is NOT game-disjoint (positions of one game are spread "
            "over many shards) — rebuild with `fly build-shards --val-every K` for a clean val set",
            stacklevel=3)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(files))
    n_val = round(val_fraction * len(files))
    if val_fraction > 0 and len(files) >= 2:
        n_val = max(1, n_val)
    n_val = min(n_val, len(files) - 1) if len(files) > 1 else 0
    val_idx = set(perm[len(files) - n_val:].tolist()) if n_val else set()
    train = [f for i, f in enumerate(files) if i not in val_idx]
    val = [f for i, f in enumerate(files) if i in val_idx]
    return train, val


def _as_files(x: str | Path | Sequence[str | Path]) -> list[Path]:
    if isinstance(x, (str, Path)):
        p = Path(x)
        return list_shards(p) if p.is_dir() else [p]
    return [Path(f) for f in x]


# ---------------------------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------------------------
class ShardDataset(IterableDataset):
    """Streams positions from shard files.

    Every epoch (``set_epoch``) the shard order is reshuffled with ``seed + epoch``; shards are
    then split across DataLoader workers (and optionally distributed ``rank``/``world_size``) at
    the shard level, and positions inside a shard are shuffled.  Yields dicts of numpy values:
    ``planes uint8 (20, 8, 8)``, ``move``, ``value``, ``elo``, ``ply`` (python ints) and ``value_scale``
    (float, the shard's; the float value target is ``value / value_scale``).
    Use ``collate`` (below) as the DataLoader ``collate_fn``.
    """

    def __init__(self, shard_files: Sequence[str | Path] | str | Path, seed: int = 0,
                 shuffle: bool = True, rank: int = 0, world_size: int = 1) -> None:
        super().__init__()
        self.files = _as_files(shard_files)
        if not self.files:
            raise FileNotFoundError(f"no shards: {shard_files}")
        self.seed = seed
        self.shuffle = shuffle
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:  # total positions (all workers)
        return count_positions(self.files)

    def _my_files(self) -> list[Path]:
        order = np.arange(len(self.files))
        if self.shuffle:
            order = np.random.default_rng(self.seed + self.epoch).permutation(len(self.files))
        files = [self.files[i] for i in order]
        # shard-level split: first across distributed ranks, then across DataLoader workers
        files = files[self.rank::self.world_size]
        info = get_worker_info()
        if info is not None:
            files = files[info.id::info.num_workers]
        return files

    def __iter__(self) -> Iterator[dict]:
        info = get_worker_info()
        worker_id = info.id if info is not None else 0
        rng = np.random.default_rng(self.seed * 1000003 + self.epoch * 1009 + self.rank * 101 + worker_id)
        for f in self._my_files():
            shard = read_shard(f)
            n = shard["move"].shape[0]
            scale = float(shard[VALUE_SCALE_KEY])
            idx = rng.permutation(n) if self.shuffle else np.arange(n)
            for i in idx:
                yield {
                    "planes": shard["planes"][i],
                    "move": int(shard["move"][i]),
                    "value": int(shard["value"][i]),
                    "elo": int(shard["elo"][i]),
                    "ply": int(shard["ply"][i]),
                    VALUE_SCALE_KEY: scale,
                }


def planes_to_float(planes_u8: np.ndarray) -> np.ndarray:
    """uint8 stored planes -> float32 with plane 18 rescaled to [0, 1]; keeps the leading dims."""
    out = planes_u8.astype(np.float32)
    out[..., 18, :, :] /= HALFMOVE_SCALE
    return out


def collate(batch: Sequence[dict]) -> dict[str, torch.Tensor]:
    """DataLoader collate: ``planes float32 (B, 1280)``, ``move int64``, ``value float32``, ``elo``, ``ply`` int64.

    ``value`` is the stored int8 divided by the item's ``value_scale`` (1 when absent): outcomes stay
    ``+1 / 0 / -1``, quantised engine evaluations become ``int8 / 127`` in [-1, 1].
    """
    planes = planes_to_float(np.stack([b["planes"] for b in batch])).reshape(len(batch), FLAT_INPUT)
    return {
        "planes": torch.from_numpy(planes),
        "move": torch.tensor([b["move"] for b in batch], dtype=torch.int64),
        "value": torch.tensor([b["value"] / b.get(VALUE_SCALE_KEY, 1.0) for b in batch], dtype=torch.float32),
        "elo": torch.tensor([b["elo"] for b in batch], dtype=torch.int64),
        "ply": torch.tensor([b["ply"] for b in batch], dtype=torch.int64),
    }


__all__ = [
    "SHARD_KEYS",
    "SHARD_SIZE",
    "VALUE_SCALE_KEY",
    "VAL_SUFFIX",
    "ShardDataset",
    "collate",
    "count_positions",
    "list_shards",
    "list_val_shards",
    "load_split",
    "parse_shard_names",
    "planes_to_float",
    "read_shard",
    "read_value_scale",
    "shard_path",
    "shard_series",
    "val_name",
    "write_shard",
]
