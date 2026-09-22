"""Lichess PGN dumps -> filtered, shuffled position shards (SPEC §5).

Pipeline (``build_shards``):
    1. the main process streams the ``.pgn.zst`` (zstandard) and cuts the text into blocks of whole
       games at ``\\n\\n[Event`` boundaries (cheap: no parsing in the main process);
    2. worker processes split a block into games, reject games on their headers (Elo, result,
       variant, time control) and parse the survivors with ``chess.pgn.read_game``, producing compact
       per-position features (``encoding.board_features``), move index, value, Elo and ply;
    3. the main process pushes everything through a large shuffle pool and writes shards of
       ``SHARD_SIZE`` positions (``shards.write_shard``) from a background thread.

With ``val_every=k > 0`` one game in ``k`` (chosen by a stable hash of its Lichess ``Site`` URL, so the
choice is reproducible across worker processes and rebuilds) is routed *whole* into a separate pool and
written as ``<name>.val-NNNNN.npz``: the validation set is then disjoint from training at the game level
(a position-level shuffle followed by a shard-level split would put neighbouring positions of the same game,
with the same value label, on both sides).

Values are from the mover's perspective (+1 win / 0 draw / -1 loss), Elo is the mover's.
"""
from __future__ import annotations

import io
import re
import time
import zlib
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from multiprocessing import get_context
from pathlib import Path
from typing import BinaryIO, NamedTuple, TextIO

import chess
import chess.pgn
import numpy as np
import zstandard

from ..chessenv.encoding import (
    NUM_FEATURES,
    board_features,
    move_to_index,
    planes_from_features,
    transposition_key,
)
from ..paths import PGN_DIR, SHARDS_DIR
from .shards import SHARD_SIZE, shard_path, val_name, write_shard

LICHESS_URL = "https://database.lichess.org/standard/lichess_db_standard_rated_{month}.pgn.zst"
_HEADER_RE = re.compile(r'^\[(\w+) "([^"]*)"\]', re.MULTILINE)
_GAME_SPLIT_RE = re.compile(r"\n\n(?=\[Event )")
_RESULT_VALUE = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}


# ---------------------------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class GameFilter:
    """Which games / positions to keep."""
    min_elo: int = 1800                  # both players
    max_elo: int | None = None
    results: tuple[str, ...] = ("1-0", "0-1", "1/2-1/2")
    variants: tuple[str, ...] = ("Standard",)   # accepted [Variant] values; no header = Standard
    min_base_seconds: int = 180          # TimeControl base; bullet (< 180 s) is dropped
    allow_unknown_time: bool = True      # TimeControl "-" (unlimited / correspondence) or missing
    skip_openings: int = 4               # first plies of every game are not emitted
    max_plies: int | None = None         # optionally truncate very long games

    def accepts(self, headers: dict[str, str]) -> bool:
        try:
            white = int(headers.get("WhiteElo", ""))
            black = int(headers.get("BlackElo", ""))
        except ValueError:
            return False
        if white < self.min_elo or black < self.min_elo:
            return False
        if self.max_elo is not None and (white > self.max_elo or black > self.max_elo):
            return False
        if headers.get("Result") not in self.results:
            return False
        if headers.get("Variant", "Standard") not in self.variants:
            return False
        if headers.get("SetUp") == "1" or "FEN" in headers:
            return False
        tc = headers.get("TimeControl", "-")
        if tc == "-" or tc == "":
            return self.allow_unknown_time
        try:
            base = int(tc.split("+")[0])
        except ValueError:
            return self.allow_unknown_time
        return base >= self.min_base_seconds


def parse_headers(game_text: str) -> dict[str, str]:
    """Tag pairs of one game's PGN text (only the header block is scanned)."""
    end = game_text.find("\n\n")
    return dict(_HEADER_RE.findall(game_text if end < 0 else game_text[:end]))


# ---------------------------------------------------------------------------------------------
# Per-game extraction
# ---------------------------------------------------------------------------------------------
class GameArrays(NamedTuple):
    feats: np.ndarray   # uint64 (P, NUM_FEATURES)
    move: np.ndarray    # int16 (P,)
    value: np.ndarray   # int8 (P,)
    elo: np.ndarray     # int16 (P,)
    ply: np.ndarray     # int16 (P,)


class Position(NamedTuple):
    """One training example as produced by ``positions_from_pgn`` (tests / inspection)."""
    planes: np.ndarray  # uint8 (20, 8, 8), shard storage encoding
    move: int
    value: int
    elo: int
    ply: int
    fen: str
    uci: str


def game_arrays(game: chess.pgn.Game, headers: dict[str, str], filt: GameFilter) -> GameArrays | None:
    """Features of every kept position of a parsed game (empty arrays if none), None if the game is unusable."""
    if game.errors:
        return None
    result = headers.get("Result", game.headers.get("Result", "*"))
    if result not in _RESULT_VALUE:
        return None
    white_value = _RESULT_VALUE[result]
    elos = (int(headers["WhiteElo"]), int(headers["BlackElo"]))
    board = game.board()
    seen: Counter = Counter()
    feats, moves, values, elo_l, plies = [], [], [], [], []
    for ply, move in enumerate(game.mainline_moves()):
        if filt.max_plies is not None and ply >= filt.max_plies:
            break
        key = transposition_key(board)
        repeated = seen[key] > 0
        seen[key] += 1
        if ply >= filt.skip_openings:
            mover_white = board.turn == chess.WHITE
            feats.append(board_features(board, repeated))
            moves.append(move_to_index(move, board))
            values.append(white_value if mover_white else -white_value)
            elo_l.append(elos[0] if mover_white else elos[1])
            plies.append(ply)
        board.push(move)
    if not feats:
        return _concat([])
    return GameArrays(
        np.stack(feats),
        np.array(moves, dtype=np.int16),
        np.array(values, dtype=np.int8),
        np.array(elo_l, dtype=np.int16),
        np.array(plies, dtype=np.int16),
    )


def _concat(parts: Sequence[GameArrays]) -> GameArrays:
    if not parts:
        return GameArrays(np.zeros((0, NUM_FEATURES), np.uint64), np.zeros(0, np.int16),
                          np.zeros(0, np.int8), np.zeros(0, np.int16), np.zeros(0, np.int16))
    return GameArrays(*(np.concatenate([p[i] for p in parts]) for i in range(5)))


def split_games(block: str) -> list[str]:
    block = block.strip()
    return [g for g in _GAME_SPLIT_RE.split(block) if g] if block else []


def is_val_game(text: str, headers: dict[str, str], val_every: int) -> bool:
    """Deterministic game-level hold-out: one game in ``val_every`` (0 = none) goes to the validation series.

    Keyed on a stable hash (``zlib.crc32``) of the ``Site`` header (the Lichess game URL); games without one
    are keyed on their full PGN text.  Python's ``hash`` is salted per process and would route the same game
    differently in every builder worker.
    """
    if val_every <= 0:
        return False
    key = headers.get("Site") or text
    return zlib.crc32(key.encode("utf-8", "replace")) % val_every == 0


def process_block(block: str, filt: GameFilter,
                  val_every: int = 0) -> tuple[GameArrays, GameArrays, dict[str, int]]:
    """Worker entry point: filter + parse + encode every game in a text block.

    Returns ``(train_arrays, val_arrays, stats)``; whole games are routed to ``val_arrays`` by
    :func:`is_val_game` (``val_every=0``: every game is training data and ``val_arrays`` is empty).
    """
    stats = Counter(games_seen=0, games_kept=0, games_error=0, games_short=0, positions=0,
                    games_val=0, positions_val=0)
    parts: list[GameArrays] = []
    val_parts: list[GameArrays] = []
    for text in split_games(block):
        stats["games_seen"] += 1
        headers = parse_headers(text)
        if not filt.accepts(headers):
            continue
        game = chess.pgn.read_game(io.StringIO(text))
        arrays = game_arrays(game, headers, filt) if game is not None else None
        if arrays is None:
            stats["games_error"] += 1
            continue
        if len(arrays.move) == 0:
            stats["games_short"] += 1
            continue
        if is_val_game(text, headers, val_every):
            stats["games_val"] += 1
            stats["positions_val"] += len(arrays.move)
            val_parts.append(arrays)
        else:
            stats["games_kept"] += 1
            stats["positions"] += len(arrays.move)
            parts.append(arrays)
    return _concat(parts), _concat(val_parts), dict(stats)


# ---------------------------------------------------------------------------------------------
# Streaming the dumps
# ---------------------------------------------------------------------------------------------
def open_pgn_text(path: str | Path) -> TextIO:
    """Text stream of a ``.pgn`` or ``.pgn.zst`` file."""
    path = Path(path)
    raw: BinaryIO = open(path, "rb")  # noqa: SIM115 - returned to the caller as a stream
    if path.suffix == ".zst":
        raw = zstandard.ZstdDecompressor().stream_reader(raw, read_across_frames=True)  # type: ignore[assignment]
    return io.TextIOWrapper(raw, encoding="utf-8", errors="replace")


def iter_game_blocks(pgn_paths: Iterable[str | Path], chunk_chars: int = 1 << 20) -> Iterator[str]:
    """Yield text blocks of whole games (about ``chunk_chars`` each) from one or more PGN files."""
    for path in pgn_paths:
        with open_pgn_text(path) as f:
            carry = ""
            while True:
                chunk = f.read(chunk_chars)
                if not chunk:
                    if carry.strip():
                        yield carry
                    break
                buf = carry + chunk
                cut = buf.rfind("\n\n[Event ")
                if cut < 0:
                    carry = buf
                    continue
                yield buf[: cut + 1]
                carry = buf[cut + 2:]


def _truncate_block(block: str, n_games: int) -> str:
    """Keep only the first ``n_games`` games of a block."""
    pos = 0
    for _ in range(n_games):
        nxt = block.find("\n[Event ", pos + 1)
        if nxt < 0:
            return block
        pos = nxt
    return block[: pos + 1]


def iter_game_texts(path_or_file: str | Path | TextIO, chunk_chars: int = 1 << 20) -> Iterator[str]:
    """Yield the PGN text of every game in a file / text stream."""
    if isinstance(path_or_file, (str, Path)):
        handle = open_pgn_text(path_or_file)
    else:
        handle = path_or_file
    try:
        carry = ""
        while True:
            chunk = handle.read(chunk_chars)
            if not chunk:
                yield from split_games(carry)
                break
            buf = carry + chunk
            cut = buf.rfind("\n\n[Event ")
            if cut < 0:
                carry = buf
                continue
            yield from split_games(buf[: cut + 1])
            carry = buf[cut + 2:]
    finally:
        if handle is not path_or_file:
            handle.close()


def positions_from_pgn(path_or_file: str | Path | TextIO, filt: GameFilter | None = None) -> Iterator[Position]:
    """Generator over kept positions of a PGN file / text stream (single process; for tests and inspection).

    Uses exactly the per-game code path of the parallel builder (``process_block``).
    """
    filt = filt or GameFilter()
    for text in iter_game_texts(path_or_file):
        headers = parse_headers(text)
        if not filt.accepts(headers):
            continue
        game = chess.pgn.read_game(io.StringIO(text))
        arrays = game_arrays(game, headers, filt) if game is not None else None
        if arrays is None or len(arrays.move) == 0:
            continue
        planes = planes_from_features(arrays.feats, quantized=True)
        board = game.board()
        kept = 0
        for ply, move in enumerate(game.mainline_moves()):
            if filt.max_plies is not None and ply >= filt.max_plies:
                break
            if ply >= filt.skip_openings:
                yield Position(planes[kept], int(arrays.move[kept]), int(arrays.value[kept]),
                               int(arrays.elo[kept]), int(arrays.ply[kept]), board.fen(), move.uci())
                kept += 1
            board.push(move)


# ---------------------------------------------------------------------------------------------
# Shuffle pool + shard writing
# ---------------------------------------------------------------------------------------------
class ShufflePool:
    """Fixed-capacity pool: positions accumulate, and whenever it is full ``shard_size`` random
    positions are drawn out (the rest stays and keeps mixing with later arrivals)."""

    def __init__(self, capacity: int, shard_size: int, rng: np.random.Generator) -> None:
        self.capacity = max(capacity, shard_size)
        self.shard_size = shard_size
        self.rng = rng
        self.n = 0
        self.buf = GameArrays(
            np.empty((self.capacity, NUM_FEATURES), np.uint64), np.empty(self.capacity, np.int16),
            np.empty(self.capacity, np.int8), np.empty(self.capacity, np.int16), np.empty(self.capacity, np.int16),
        )

    def add(self, arrays: GameArrays) -> Iterator[GameArrays]:
        offset = 0
        total = len(arrays.move)
        while offset < total:
            take = min(self.capacity - self.n, total - offset)
            for dst, src in zip(self.buf, arrays):
                dst[self.n:self.n + take] = src[offset:offset + take]
            self.n += take
            offset += take
            if self.n == self.capacity:
                yield self._draw(self.shard_size)

    def _draw(self, k: int) -> GameArrays:
        perm = self.rng.permutation(self.n)
        take, keep = perm[:k], perm[k:]
        out = GameArrays(*(a[take].copy() for a in self.buf))
        for a in self.buf:
            a[: len(keep)] = a[keep]
        self.n = len(keep)
        return out

    def drain(self) -> Iterator[GameArrays]:
        while self.n > 0:
            yield self._draw(min(self.shard_size, self.n))


def _write_shard_from_arrays(path: Path, arrays: GameArrays) -> Path:
    planes = planes_from_features(arrays.feats, quantized=True)
    return write_shard(path, planes, arrays.move, arrays.value, arrays.elo, arrays.ply)


@dataclass
class BuildStats:
    games_seen: int = 0
    games_kept: int = 0       # training games (the game-level hold-out is counted in games_val)
    games_error: int = 0
    games_short: int = 0      # passed the filters but had no position after skip_openings
    positions: int = 0        # training positions
    shards: int = 0           # training shards <name>-NNNNN.npz
    seconds: float = 0.0
    shard_files: list[str] = field(default_factory=list)
    games_val: int = 0        # games held out whole into <name>.val-NNNNN.npz (build_shards(val_every=k))
    positions_val: int = 0
    val_shards: int = 0
    val_shard_files: list[str] = field(default_factory=list)

    def summary(self) -> str:
        s = max(self.seconds, 1e-9)
        val = (f", val: {self.games_val:,} games / {self.positions_val:,} positions in {self.val_shards} shards"
               if self.games_val or self.val_shards else "")
        return (f"games seen {self.games_seen:,} / kept {self.games_kept:,} "
                f"(parse errors {self.games_error:,}, too short {self.games_short:,}), "
                f"positions {self.positions:,} in {self.shards} shards{val}, {self.seconds:.1f}s "
                f"({self.games_seen / s:,.0f} games/s, {(self.positions + self.positions_val) / s:,.0f} positions/s)")

    def to_dict(self) -> dict:
        return asdict(self)


def build_shards(
    pgn_paths: Sequence[str | Path] | str | Path,
    out_dir: str | Path = SHARDS_DIR,
    name: str = "lichess",
    min_elo: int = 1800,
    max_positions: int | None = None,
    skip_openings: int = 4,
    workers: int = 16,
    max_games: int | None = None,
    game_filter: GameFilter | None = None,
    shard_size: int = SHARD_SIZE,
    shuffle_buffer: int = 2_000_000,
    seed: int = 0,
    chunk_chars: int = 1 << 20,
    progress: bool = True,
    val_every: int = 0,
) -> BuildStats:
    """Stream PGN dumps, filter, encode, shuffle and write ``<out_dir>/<name>-NNNNN.npz`` shards.

    ``game_filter`` overrides ``min_elo``/``skip_openings``. ``max_games`` bounds the number of games
    read (for smoke runs); ``max_positions`` stops after roughly that many kept training positions.
    The last shard of each series may be smaller than ``shard_size``.

    ``val_every=k > 0`` holds out one game in ``k`` *whole* (:func:`is_val_game`) into a second series
    ``<out_dir>/<name>.val-NNNNN.npz`` (its own, smaller shuffle pool); ``shards.load_split`` uses that
    series as the game-disjoint validation set.
    """
    if isinstance(pgn_paths, (str, Path)):
        pgn_paths = [pgn_paths]
    if val_every < 0:
        raise ValueError("val_every must be >= 0 (0 = no game-level hold-out)")
    filt = game_filter or GameFilter(min_elo=min_elo, skip_openings=skip_openings)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    pool = ShufflePool(shuffle_buffer, shard_size, rng)
    val_pool = ShufflePool(max(shard_size, shuffle_buffer // max(val_every, 1)), shard_size, rng)
    stats = BuildStats()
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
        pending_writes.append(writer.submit(_write_shard_from_arrays, path, arrays))
        while len(pending_writes) > 2:  # bound memory: at most 2 shards in flight
            pending_writes.pop(0).result()

    def blocks() -> Iterator[str]:
        seen = 0
        for block in iter_game_blocks(pgn_paths, chunk_chars):
            if max_games is not None:
                n = block.count("[Event ")
                if seen + n >= max_games:
                    yield _truncate_block(block, max_games - seen)
                    return
                seen += n
            yield block

    bar = None
    if progress:
        from tqdm import tqdm
        bar = tqdm(unit="game", dynamic_ncols=True, desc=f"build-shards {name}")
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as ex:
            in_flight: list[Future] = []
            stop = False

            def consume(fut: Future) -> None:
                nonlocal stop
                arrays, val_arrays, st = fut.result()
                stats.games_seen += st["games_seen"]
                stats.games_kept += st["games_kept"]
                stats.games_error += st["games_error"]
                stats.games_short += st["games_short"]
                stats.positions += st["positions"]
                stats.games_val += st["games_val"]
                stats.positions_val += st["positions_val"]
                if bar is not None:
                    bar.update(st["games_seen"])
                    bar.set_postfix(kept=stats.games_kept, positions=stats.positions,
                                    shards=stats.shards, refresh=False)
                for full in pool.add(arrays):
                    flush(full)
                for full in val_pool.add(val_arrays):
                    flush(full, val=True)
                if max_positions is not None and stats.positions >= max_positions:
                    stop = True

            for block in blocks():
                in_flight.append(ex.submit(process_block, block, filt, val_every))
                while len(in_flight) >= 2 * workers:
                    consume(in_flight.pop(0))
                if stop:
                    break
            for fut in in_flight:
                consume(fut)
        for rest in pool.drain():
            flush(rest)
        for rest in val_pool.drain():
            flush(rest, val=True)
        for fut in pending_writes:
            fut.result()
    finally:
        writer.shutdown(wait=True)
        if bar is not None:
            bar.close()
    stats.seconds = time.perf_counter() - t0
    return stats


# ---------------------------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------------------------
def download_months(months: Iterable[str], out_dir: str | Path = PGN_DIR, force: bool = False) -> list[Path]:
    """Download ``lichess_db_standard_rated_<YYYY-MM>.pgn.zst`` dumps (CC0) with a progress bar."""
    import requests
    from tqdm import tqdm

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for month in months:
        url = LICHESS_URL.format(month=month)
        dest = out_dir / url.rsplit("/", 1)[-1]
        paths.append(dest)
        if dest.exists() and not force:
            continue
        tmp = dest.with_suffix(".part")
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", 0)) or None
            with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
                    bar.update(len(chunk))
        tmp.replace(dest)
    return paths


__all__ = [
    "LICHESS_URL",
    "BuildStats",
    "GameArrays",
    "GameFilter",
    "Position",
    "ShufflePool",
    "build_shards",
    "download_months",
    "game_arrays",
    "is_val_game",
    "iter_game_blocks",
    "iter_game_texts",
    "open_pgn_text",
    "parse_headers",
    "positions_from_pgn",
    "process_block",
    "split_games",
]
