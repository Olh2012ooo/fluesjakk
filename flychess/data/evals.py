"""Lichess engine-evaluation database -> position shards with a Stockfish value target (docs/DATA.md).

Source: ``https://database.lichess.org/lichess_db_eval.jsonl.zst`` (CC0; ~22 GB compressed, ~270 M positions,
~12 k positions per compressed MB).  One JSON object per line::

    {"fen": "<board> <turn> <castling> <ep>",              # 4 fields: no halfmove clock / fullmove number
     "evals": [{"pvs": [{"cp": 311, "line": "g2e4 f8e8 ..."}, ...], "knodes": 206765, "depth": 36}, ...]}

Every position becomes one training example (positions are independent — there is no game to hold
out, so the validation split ``<name>.val-NNNNN.npz`` is a stable hash of the FEN):

* the eval with the greatest ``depth`` is used; positions shallower than ``min_depth`` are skipped;
* its first pv is the best line: the first UCI token is the **policy target** (``move_to_index``; positions
  whose move is unparseable / illegal, or with no legal move at all, are skipped);
* the **value target** is the Lichess win-probability curve of the score
  ``v = 2 / (1 + exp(-0.00368208 * cp)) - 1`` (cp from White's point of view; ``mate: m`` -> ``±1``), negated
  when Black is to move so that, as everywhere else, ``+1`` is good for the **mover**.  Stored quantised:
  ``value = round(v * 127)`` as int8 with ``value_scale = 127`` (``shards.collate`` divides it back);
* ``elo = 0`` (unknown), ``ply = 2 * (fullmove - 1) + (0 | 1)`` from the FEN when it carries a fullmove number
  (the Lichess dump does not: its FENs have four fields, so ``ply`` is 0 for White to move, 1 for Black, and
  the halfmove-clock plane is 0).

Pipeline (``build_eval_shards``): the main process streams the ``.zst`` (from the URL with ``requests`` or
from a local ``.jsonl[.zst]`` file), cuts the byte stream at line boundaries into ~``chunk_bytes`` blocks
and hands them to worker processes, which parse the JSON and encode the positions (``encoding.board_features``,
same storage encoding as the PGN pipeline).  Results go through the PGN pipeline's ``ShufflePool`` and are
written as ``<out_dir>/<name>-NNNNN.npz`` (``SHARD_SIZE`` positions each) by a background thread.  The stream
stops after ``max_positions`` kept positions (train + val), so a 30 M-position build downloads ~2.5 GB.
"""
from __future__ import annotations

import io
import json
import math
import time
import zlib
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from multiprocessing import get_context
from pathlib import Path
from typing import BinaryIO, NamedTuple

import chess
import numpy as np
import zstandard

from ..chessenv.encoding import NUM_FEATURES, board_features, move_to_index, planes_from_features
from ..paths import SHARDS_DIR
from .lichess import GameArrays, ShufflePool, _concat
from .shards import SHARD_SIZE, shard_path, val_name, write_shard

EVAL_DB_URL = "https://database.lichess.org/lichess_db_eval.jsonl.zst"
VALUE_SCALE = 127                 # int8 value = round(v * VALUE_SCALE)
WIN_PROB_K = 0.00368208           # Lichess: win% = 50 + 50 * (2 / (1 + exp(-K * cp)) - 1)
MATE_VALUE = 1.0                  # |v| of a forced mate (before the mover-perspective sign)
VAL_EVERY = 50                    # one position in 50 (hash of the FEN) -> '<name>.val' series (2 %)
VALUE_HIST_BINS = 20              # value histogram over [-1, 1] in steps of 0.1
MAX_ABS_CP = 20_000               # scores beyond this are clipped (tanh is saturated anyway)


# ---------------------------------------------------------------------------------------------
# Scores -> targets
# ---------------------------------------------------------------------------------------------
def cp_to_value(cp: float) -> float:
    """Centipawns (from the scoring side's point of view) -> win-probability value in (-1, 1).

    ``2 / (1 + exp(-0.00368208 * cp)) - 1``: the curve Lichess uses for its win-percentage display
    (0 cp -> 0, +100 cp -> +0.18, +300 cp -> +0.50, +1000 cp -> +0.95).
    """
    cp = max(-MAX_ABS_CP, min(MAX_ABS_CP, float(cp)))
    return 2.0 / (1.0 + math.exp(-WIN_PROB_K * cp)) - 1.0


def mate_to_value(mate: int) -> float:
    """``mate: m`` (m > 0: the scoring side mates in m; m < 0: it gets mated) -> ``±MATE_VALUE``.

    ``mate: 0`` (the scoring side is already checkmated) is ``-MATE_VALUE``; such positions have no legal move
    and are dropped anyway.
    """
    return MATE_VALUE if mate > 0 else -MATE_VALUE


def score_to_value(pv: dict, white_to_move: bool) -> tuple[float, bool]:
    """``(value from the mover's perspective, is_mate)`` of one pv entry (``{'cp': int}`` or ``{'mate': int}``).

    Scores in the Lichess database are from White's point of view; the mover's value is negated when Black
    is to move.
    """
    if "mate" in pv:
        v, is_mate = mate_to_value(int(pv["mate"])), True
    else:
        v, is_mate = cp_to_value(pv["cp"]), False
    return (v if white_to_move else -v), is_mate


def quantize_value(v: float, scale: int = VALUE_SCALE) -> int:
    """``round(v * scale)`` clipped to the int8 range."""
    return int(max(-127, min(127, round(v * scale))))


def is_val_fen(fen: str, val_every: int = VAL_EVERY) -> bool:
    """Deterministic 1-in-``val_every`` validation split keyed on ``zlib.crc32`` of the FEN (0 = none)."""
    if val_every <= 0:
        return False
    return zlib.crc32(fen.encode("utf-8", "replace")) % val_every == 0


def fen_ply(board: chess.Board, fen: str) -> int:
    """0-based ply from the FEN's fullmove number (4-field FENs, as in the dump, give 0 / 1 by side to move)."""
    if len(fen.split()) >= 6:
        return 2 * (board.fullmove_number - 1) + (0 if board.turn == chess.WHITE else 1)
    return 0 if board.turn == chess.WHITE else 1


# ---------------------------------------------------------------------------------------------
# One line -> one example
# ---------------------------------------------------------------------------------------------
class EvalPosition(NamedTuple):
    """One example decoded from a database line (``positions_from_jsonl`` / tests)."""
    planes: np.ndarray   # uint8 (20, 8, 8), shard storage encoding
    move: int            # move index of the best move (mover's perspective)
    value: int           # quantised value: round(v * VALUE_SCALE)
    value_float: float   # v in [-1, 1] from the mover's perspective
    elo: int             # always 0
    ply: int
    fen: str
    uci: str
    depth: int
    is_mate: bool


class _Decoded(NamedTuple):
    feats: np.ndarray
    move: int
    value: float
    is_mate: bool
    depth: int
    ply: int
    fen: str            # the raw FEN of the line (validation-split key)
    uci: str


def best_eval(evals: list[dict]) -> dict | None:
    """The deepest eval that has at least one pv with a non-empty line (None if there is none)."""
    best = None
    for ev in evals:
        if not ev.get("pvs") or not ev["pvs"][0].get("line"):
            continue
        if best is None or int(ev.get("depth", 0)) > int(best.get("depth", 0)):
            best = ev
    return best


def decode_line(line: str | bytes, min_depth: int, skip: Counter | None = None) -> _Decoded | None:
    """Parse one JSON line into features / targets; None when the position is skipped (reason in ``skip``)."""
    try:
        obj = json.loads(line)
        fen = obj["fen"]
        evals = obj["evals"]
    except (ValueError, KeyError, TypeError):
        if skip is not None:
            skip["bad_json"] += 1
        return None
    ev = best_eval(evals)
    if ev is None:
        if skip is not None:
            skip["no_pv"] += 1
        return None
    depth = int(ev.get("depth", 0))
    if depth < min_depth:
        if skip is not None:
            skip["shallow"] += 1
        return None
    pv = ev["pvs"][0]
    if "cp" not in pv and "mate" not in pv:
        if skip is not None:
            skip["no_score"] += 1
        return None
    try:
        board = chess.Board(fen)
    except ValueError:
        if skip is not None:
            skip["bad_fen"] += 1
        return None
    uci = pv["line"].split(" ", 1)[0]
    try:
        move = chess.Move.from_uci(uci)
    except ValueError:
        if skip is not None:
            skip["bad_move"] += 1
        return None
    if not board.is_legal(move):
        if skip is not None:
            skip["illegal_move" if board.legal_moves.count() else "no_legal_moves"] += 1
        return None
    try:
        idx = move_to_index(move, board)
    except ValueError:
        if skip is not None:
            skip["bad_move"] += 1
        return None
    v, is_mate = score_to_value(pv, board.turn == chess.WHITE)
    return _Decoded(board_features(board, False), idx, v, is_mate, depth, fen_ply(board, fen), str(fen), uci)


class ChunkArrays(NamedTuple):
    """Decoded positions of one byte block (worker -> main process), not yet split into train / val."""
    feats: np.ndarray    # uint64 (P, NUM_FEATURES)
    move: np.ndarray     # int16 (P,)
    value: np.ndarray    # int8 (P,)  round(v * VALUE_SCALE)
    ply: np.ndarray      # int16 (P,)
    depth: np.ndarray    # int16 (P,)  depth of the used eval
    mate: np.ndarray     # bool (P,)   score was a forced mate
    is_val: np.ndarray   # bool (P,)   -> '<name>.val' series (is_val_fen)
    lines: int           # lines read in the block
    skip: dict[str, int]  # skip reason -> count

    def __len__(self) -> int:
        return int(self.move.shape[0])

    def head(self, n: int) -> ChunkArrays:
        """The first ``n`` positions (exact ``max_positions`` cap)."""
        return ChunkArrays(self.feats[:n], self.move[:n], self.value[:n], self.ply[:n], self.depth[:n],
                           self.mate[:n], self.is_val[:n], self.lines, self.skip)

    def split(self) -> tuple[GameArrays, GameArrays]:
        """``(train, val)`` as the PGN pipeline's ``GameArrays`` (elo = 0)."""
        out = []
        for mask in (~self.is_val, self.is_val):
            n = int(mask.sum())
            out.append(GameArrays(self.feats[mask], self.move[mask], self.value[mask],
                                  np.zeros(n, np.int16), self.ply[mask]) if n else _concat([]))
        return out[0], out[1]


def process_chunk(data: bytes, min_depth: int, val_every: int = VAL_EVERY) -> ChunkArrays:
    """Worker entry point: decode every line of a byte block into compact per-position arrays."""
    skip: Counter = Counter()
    feats, moves, values, plies, depths, mates, is_val = [], [], [], [], [], [], []
    n_lines = 0
    for line in data.split(b"\n"):
        if not line.strip():
            continue
        n_lines += 1
        d = decode_line(line, min_depth, skip)
        if d is None:
            continue
        feats.append(d.feats)
        moves.append(d.move)
        values.append(quantize_value(d.value))
        plies.append(d.ply)
        depths.append(d.depth)
        mates.append(d.is_mate)
        is_val.append(is_val_fen(d.fen, val_every))
    n = len(moves)
    return ChunkArrays(
        np.stack(feats) if n else np.zeros((0, NUM_FEATURES), np.uint64),
        np.array(moves, np.int16), np.array(values, np.int8), np.array(plies, np.int16),
        np.array(depths, np.int16), np.array(mates, bool), np.array(is_val, bool), n_lines, dict(skip),
    )


def positions_from_jsonl(path_or_lines, min_depth: int = 20) -> Iterator[EvalPosition]:
    """Generator over the kept examples of a ``.jsonl`` / ``.jsonl.zst`` file (or an iterable of lines).

    Single process; uses the worker's ``decode_line`` so it is the reference for what the shards hold.
    """
    lines = iter_lines(path_or_lines) if isinstance(path_or_lines, (str, Path)) else path_or_lines
    for line in lines:
        if isinstance(line, str):
            line = line.encode("utf-8")
        if not line.strip():
            continue
        d = decode_line(line, min_depth)
        if d is None:
            continue
        planes = planes_from_features(d.feats[None], quantized=True)[0]
        yield EvalPosition(planes, d.move, quantize_value(d.value), d.value, 0, d.ply, d.fen, d.uci,
                           d.depth, d.is_mate)


# ---------------------------------------------------------------------------------------------
# Streaming the database
# ---------------------------------------------------------------------------------------------
def open_eval_stream(source: str | Path, timeout: float = 120.0) -> BinaryIO:
    """Binary stream of decompressed JSONL from a URL (``http(s)://``), a local ``.zst`` or a plain ``.jsonl``.

    The URL is streamed with ``requests`` (nothing is written to disk) and decompressed on the fly; closing the
    returned stream aborts the download, so ``max_positions`` bounds the transferred bytes.
    """
    src = str(source)
    if src.startswith(("http://", "https://")):
        import requests

        resp = requests.get(src, stream=True, timeout=timeout)
        resp.raise_for_status()
        raw: BinaryIO = resp.raw  # urllib3 response: file-like, read(n)
        raw.decode_content = True  # transparently undo any HTTP content-encoding (the body itself is .zst)
        zst = src.endswith(".zst")
        closer = resp.close
    else:
        path = Path(source)
        raw = open(path, "rb")  # noqa: SIM115 - returned to the caller as a stream
        zst = path.suffix == ".zst"
        closer = raw.close
    if zst:
        stream = zstandard.ZstdDecompressor().stream_reader(raw, read_across_frames=True, closefd=False)
        return _ClosingStream(stream, closer)
    return _ClosingStream(raw, closer)


class _ClosingStream(io.RawIOBase):
    """Wraps a readable stream so that ``close()`` also runs ``closer`` (the HTTP response / file)."""

    def __init__(self, stream, closer) -> None:
        super().__init__()
        self._stream = stream
        self._closer = closer

    def readable(self) -> bool:
        return True

    def read(self, n: int = -1) -> bytes:  # type: ignore[override]
        return self._stream.read(n)

    def close(self) -> None:
        if not self.closed:
            try:
                self._stream.close()
            finally:
                try:
                    self._closer()
                finally:
                    super().close()


def iter_line_blocks(stream: BinaryIO, chunk_bytes: int = 4 << 20) -> Iterator[bytes]:
    """Yield blocks of whole lines (about ``chunk_bytes`` each) from a decompressed byte stream."""
    carry = b""
    while True:
        chunk = stream.read(chunk_bytes)
        if not chunk:
            if carry.strip():
                yield carry
            return
        buf = carry + chunk
        cut = buf.rfind(b"\n")
        if cut < 0:
            carry = buf
            continue
        yield buf[: cut + 1]
        carry = buf[cut + 1:]


def iter_lines(source: str | Path, chunk_bytes: int = 4 << 20) -> Iterator[bytes]:
    """Yield every line (bytes, without the newline) of a ``.jsonl`` / ``.jsonl.zst`` file or URL."""
    with open_eval_stream(source) as stream:
        for block in iter_line_blocks(stream, chunk_bytes):
            for line in block.split(b"\n"):
                if line:
                    yield line


# ---------------------------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------------------------
@dataclass
class EvalBuildStats:
    lines: int = 0                 # database lines decoded (incl. the tail of the last blocks beyond max_positions)
    positions: int = 0             # written training positions
    positions_val: int = 0         # written validation positions ('<name>.val' series)
    mates: int = 0                 # written positions whose score is a forced mate
    skipped: dict[str, int] = field(default_factory=dict)   # reason -> count over the decoded lines
    depth_hist: dict[int, int] = field(default_factory=dict)  # depth of the used eval -> written positions
    value_hist: list[int] = field(default_factory=lambda: [0] * VALUE_HIST_BINS)  # written v: [-1,-0.9) ... [0.9,1]
    shards: int = 0
    val_shards: int = 0
    shard_files: list[str] = field(default_factory=list)
    val_shard_files: list[str] = field(default_factory=list)
    seconds: float = 0.0
    bytes_read: int = 0            # decompressed bytes consumed from the stream
    value_scale: int = VALUE_SCALE
    min_depth: int = 20
    source: str = ""

    @property
    def kept(self) -> int:
        return self.positions + self.positions_val

    def add_positions(self, chunk: ChunkArrays) -> None:
        """Accumulate the mate / depth / value statistics of the positions about to be written."""
        self.mates += int(chunk.mate.sum())
        for d, c in zip(*np.unique(chunk.depth, return_counts=True)):
            self.depth_hist[int(d)] = self.depth_hist.get(int(d), 0) + int(c)
        v = chunk.value.astype(np.float64) / self.value_scale
        bins = np.minimum(((v + 1.0) / 2.0 * VALUE_HIST_BINS).astype(np.int64), VALUE_HIST_BINS - 1)
        for i, c in enumerate(np.bincount(bins, minlength=VALUE_HIST_BINS).tolist()):
            self.value_hist[i] += c

    def summary(self) -> str:
        s = max(self.seconds, 1e-9)
        skipped = ", ".join(f"{k} {v:,}" for k, v in sorted(self.skipped.items(), key=lambda kv: -kv[1]))
        return (f"lines {self.lines:,} -> kept {self.kept:,} ({self.positions:,} train in {self.shards} shards, "
                f"{self.positions_val:,} val in {self.val_shards} shards), mates {self.mates:,} "
                f"({100.0 * self.mates / max(self.kept, 1):.2f} %), skipped: {skipped or 'none'}; "
                f"{self.seconds:.1f}s ({self.lines / s:,.0f} lines/s, {self.kept / s:,.0f} kept positions/s, "
                f"{self.bytes_read / s / 1e6:,.1f} MB/s decompressed)")

    def depth_summary(self) -> str:
        """Kept positions by depth bucket (``<20`` never appears with the default ``min_depth``)."""
        buckets: Counter = Counter()
        for d, c in self.depth_hist.items():
            buckets[min(d // 10 * 10, 60)] += c
        total = max(self.kept, 1)
        return ", ".join(f"{'>=' if b == 60 else ''}{b}{'' if b == 60 else '-' + str(b + 9)}: "
                         f"{100.0 * buckets[b] / total:.1f} %" for b in sorted(buckets))

    def value_summary(self) -> str:
        total = max(sum(self.value_hist), 1)
        lo = [-1.0 + i * 2.0 / VALUE_HIST_BINS for i in range(VALUE_HIST_BINS)]
        return ", ".join(f"[{a:+.1f},{a + 2.0 / VALUE_HIST_BINS:+.1f}): {100.0 * c / total:.1f} %"
                         for a, c in zip(lo, self.value_hist))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["kept"] = self.kept
        return d


def _write_eval_shard(path: Path, arrays: GameArrays) -> Path:
    planes = planes_from_features(arrays.feats, quantized=True)
    return write_shard(path, planes, arrays.move, arrays.value, arrays.elo, arrays.ply, value_scale=VALUE_SCALE)


def build_eval_shards(
    source: str | Path = EVAL_DB_URL,
    out_dir: str | Path = SHARDS_DIR,
    name: str = "evals",
    max_positions: int | None = 30_000_000,
    min_depth: int = 20,
    workers: int = 8,
    val_every: int = VAL_EVERY,
    shard_size: int = SHARD_SIZE,
    shuffle_buffer: int = 2_000_000,
    seed: int = 0,
    chunk_bytes: int = 4 << 20,
    progress: bool = True,
    max_lines: int | None = None,
) -> EvalBuildStats:
    """Stream the Lichess eval database, encode, shuffle and write ``<out_dir>/<name>-NNNNN.npz`` shards.

    ``source`` is the database URL (default; streamed, never stored) or a local ``.jsonl`` / ``.jsonl.zst``.
    Stops after exactly ``max_positions`` kept positions (train + val; the last block is truncated) or after
    ``max_lines`` lines read.  One position in ``val_every`` (a hash of the
    FEN, :func:`is_val_fen`) is written to the ``<name>.val-NNNNN.npz`` series; ``0`` disables the split.
    Values are quantised with ``value_scale = 127`` (see the module docstring); Elo is 0.
    """
    if val_every < 0:
        raise ValueError("val_every must be >= 0 (0 = no validation series)")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    pool = ShufflePool(shuffle_buffer, shard_size, rng)
    val_pool = ShufflePool(max(shard_size, shuffle_buffer // max(val_every, 1)), shard_size, rng)
    stats = EvalBuildStats(min_depth=min_depth, source=str(source))
    t0 = time.perf_counter()
    writer = ThreadPoolExecutor(max_workers=2)
    pending_writes: list[Future] = []

    def flush(arrays: GameArrays, val: bool = False) -> None:
        if val:
            path = shard_path(out_dir, val_name(name), stats.val_shards)
            stats.val_shards += 1
            stats.val_shard_files.append(str(path))
        else:
            path = shard_path(out_dir, name, stats.shards)
            stats.shards += 1
            stats.shard_files.append(str(path))
        pending_writes.append(writer.submit(_write_eval_shard, path, arrays))
        while len(pending_writes) > 2:  # bound memory: at most 2 shards in flight
            pending_writes.pop(0).result()

    bar = None
    if progress:
        from tqdm import tqdm
        bar = tqdm(total=max_positions, unit="pos", unit_scale=True, dynamic_ncols=True,
                   desc=f"build-eval-shards {name}")
    stream = open_eval_stream(source)
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as ex:
            in_flight: list[Future] = []
            stop = False

            def consume(fut: Future) -> None:
                nonlocal stop
                chunk: ChunkArrays = fut.result()
                stats.lines += chunk.lines
                for k, v in chunk.skip.items():
                    stats.skipped[k] = stats.skipped.get(k, 0) + v
                if max_positions is not None and stats.kept + len(chunk) >= max_positions:
                    chunk = chunk.head(max_positions - stats.kept)  # exact cap
                    stop = True
                if len(chunk) == 0:
                    return
                stats.add_positions(chunk)
                arrays, val_arrays = chunk.split()
                stats.positions += len(arrays.move)
                stats.positions_val += len(val_arrays.move)
                if bar is not None:
                    bar.update(len(chunk))
                    bar.set_postfix(lines=stats.lines, shards=stats.shards, mates=stats.mates, refresh=False)
                for full in pool.add(arrays):
                    flush(full)
                for full in val_pool.add(val_arrays):
                    flush(full, val=True)

            lines_sent = 0
            for block in iter_line_blocks(stream, chunk_bytes):
                if max_lines is not None:
                    n = block.count(b"\n")
                    if lines_sent + n >= max_lines:
                        block = b"\n".join(block.split(b"\n")[: max_lines - lines_sent]) + b"\n"
                        stop = True
                    lines_sent += n
                stats.bytes_read += len(block)
                in_flight.append(ex.submit(process_chunk, block, min_depth, val_every))
                while len(in_flight) >= 2 * workers:
                    consume(in_flight.pop(0))
                if stop:
                    break
            for fut in in_flight:
                if stop and fut.cancel():  # not started yet: skip it
                    continue
                consume(fut)  # finished / running blocks still count towards the (exact) cap
        for rest in pool.drain():
            flush(rest)
        for rest in val_pool.drain():
            flush(rest, val=True)
        for fut in pending_writes:
            fut.result()
    finally:
        stream.close()
        writer.shutdown(wait=True)
        if bar is not None:
            bar.close()
    stats.seconds = time.perf_counter() - t0
    return stats


__all__ = [
    "EVAL_DB_URL",
    "MATE_VALUE",
    "VALUE_SCALE",
    "VAL_EVERY",
    "ChunkArrays",
    "EvalBuildStats",
    "EvalPosition",
    "best_eval",
    "build_eval_shards",
    "cp_to_value",
    "decode_line",
    "fen_ply",
    "is_val_fen",
    "iter_line_blocks",
    "iter_lines",
    "mate_to_value",
    "open_eval_stream",
    "positions_from_jsonl",
    "process_chunk",
    "quantize_value",
    "score_to_value",
]
