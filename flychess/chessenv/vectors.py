"""Cross-language test vectors for the encoding (SPEC §3.5): ``tests/vectors/encoding.json``.

Each vector is a dict with
    ``fen``               position FEN (always present; for move-list vectors it is the final position)
    ``move_list``         optional list of UCI moves from the initial position (repetition cases)
    ``planes_sha256``     sha256 of the float32 little-endian C-order bytes of ``encode_board``
    ``planes_nonzero``    ``[[plane, rank, file, value], ...]`` for planes 0-17
    ``halfmove_plane``    scalar value of plane 18
    ``repetition_plane``  scalar value of plane 19
    ``legal_indices``     sorted move indices of all legal moves
    ``moves``             ``{uci: idx}`` for every legal move
Deterministic: run ``python -m flychess.chessenv.vectors`` to regenerate.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import chess
import numpy as np

from ..paths import VECTORS_DIR
from .encoding import encode_board, move_to_index

DEFAULT_PATH = VECTORS_DIR / "encoding.json"

CURATED_FENS: list[str] = [
    chess.STARTING_FEN,
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",                # black to move (mirror)
    "r3k2r/pppqbppp/2npbn2/4p3/4P3/2NPBN2/PPPQBPPP/R3K2R w KQkq - 4 8",          # both sides may castle both ways
    "r3k2r/pppqbppp/2npbn2/4p3/4P3/2NPBN2/PPPQBPPP/R3K2R b KQkq - 4 8",
    "r3k2r/8/8/8/8/8/8/R3K2R w Kq - 0 1",                                       # asymmetric rights
    "r3k2r/8/8/8/8/8/8/R3K2R b Kq - 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R w - - 0 1",                                        # king/rooks in place, no rights
    "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",             # white en passant
    "rnbqkbnr/pppp1ppp/8/8/3Pp3/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 3",             # black en passant
    "8/8/8/K1pP3r/8/8/8/6k1 w - c6 0 1",                                        # ep square set, capture illegal (pin)
    "8/P7/8/8/8/8/8/k6K w - - 0 1",                                             # push promotion
    "1n6/P7/8/8/8/8/8/k6K w - - 0 1",                                           # push + capture-right promotion
    "n1n5/1P6/8/8/8/8/8/k6K w - - 0 1",                                         # both capture promotions
    "k6K/8/8/8/8/8/p7/1N6 b - - 0 1",                                           # black push + capture promotion
    "k6K/8/8/8/8/8/6p1/5N1N b - - 0 1",                                         # black capture-left/right promotions
    "4k3/8/8/8/8/8/4r3/4K3 w - - 0 1",                                          # white king in check
    "rnbqkb1r/ppp2ppp/5n2/3pp1B1/3P4/8/PPP1PPPP/RN1QKBNR b KQkq - 1 4",
    "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",            # checkmate, no legal moves
    "4k3/8/8/8/8/8/8/4K2R b K - 0 1",                                           # black to move, opponent castles K
    "8/8/8/4k3/8/8/4K3/7R w - - 57 100",                                        # halfmove clock 57
    "8/8/8/4k3/8/8/4K3/7R b - - 120 80",                                        # halfmove clock clipped
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",                                # kiwipete-ish, many pins
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",     # kiwipete
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b KQkq - 0 1",
]

# Move lists from the initial position: repetition and castling-rights subtleties.
CURATED_MOVE_LISTS: list[list[str]] = [
    ["g1f3", "g8f6", "f3g1"],                                    # not yet repeated (side to move differs)
    ["g1f3", "g8f6", "f3g1", "f6g8"],                            # start position repeated
    ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3"],                    # position after 1.Nf3 repeated
    ["e2e4", "e7e5", "e1e2", "e8e7", "e2e1", "e7e8"],            # same pieces but castling rights lost: no repetition
    ["e2e4", "e7e5", "e1e2", "e8e7", "e2e1", "e7e8", "e1e2", "e8e7"],  # now repeated (rights already gone)
    ["e2e4", "e7e5", "g1f3", "b8c6", "f3g1", "c6b8", "g1f3", "b8c6"],  # repeated with black to move
]


def _random_fens(n: int, seed: int = 1234) -> list[str]:
    """Positions sampled from random playouts: plain positions plus playouts stopped at the first
    position offering a promotion / en-passant capture / check, so rare move types are covered."""
    rng = random.Random(seed)
    conditions = {
        "plain": lambda b: False,
        "promotion": lambda b: any(m.promotion for m in b.legal_moves),
        "en_passant": lambda b: b.has_legal_en_passant(),
        "check": lambda b: b.is_check(),
    }
    modes = list(conditions)
    fens: list[str] = []
    while len(fens) < n:
        mode = modes[len(fens) % len(modes)]
        stop = conditions[mode]
        board = chess.Board()
        target = rng.randint(1, 120) if mode == "plain" else 400
        for _ in range(target):
            moves = list(board.legal_moves)
            if not moves or board.is_game_over():
                break
            # bias towards captures/promotions so that rare move types show up
            special = [m for m in moves if m.promotion or board.is_capture(m)]
            move = rng.choice(special) if special and rng.random() < 0.5 else rng.choice(moves)
            board.push(move)
            if mode != "plain" and stop(board):
                break
        if mode == "plain" or stop(board):
            fens.append(board.fen())
    return fens


def vector_for(board: chess.Board, moves: list[str] | None = None) -> dict:
    planes = encode_board(board)
    nonzero = [[int(p), int(r), int(f), float(planes[p, r, f])]
               for p, r, f in zip(*np.nonzero(planes[:18]))]
    move_index = {m.uci(): move_to_index(m, board) for m in board.legal_moves}
    vec: dict = {"fen": board.fen()}
    if moves is not None:
        vec["move_list"] = list(moves)
    vec.update({
        "planes_sha256": planes_sha256(planes),
        "planes_nonzero": nonzero,
        "halfmove_plane": float(planes[18, 0, 0]),
        "repetition_plane": float(planes[19, 0, 0]),
        "legal_indices": sorted(move_index.values()),
        "moves": dict(sorted(move_index.items())),
    })
    return vec


def planes_sha256(planes: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(planes, dtype="<f4").tobytes(order="C")).hexdigest()


def build_vectors(n: int = 60, seed: int = 1234) -> list[dict]:
    vectors = [vector_for(chess.Board(fen)) for fen in CURATED_FENS]
    for moves in CURATED_MOVE_LISTS:
        board = chess.Board()
        for uci in moves:
            board.push_uci(uci)
        vectors.append(vector_for(board, moves))
    n_random = max(0, n - len(vectors))
    vectors.extend(vector_for(chess.Board(fen)) for fen in _random_fens(n_random, seed))
    return vectors


def write_encoding_vectors(path: str | Path = DEFAULT_PATH, n: int = 60, seed: int = 1234) -> Path:
    """Write ``n`` deterministic vectors (curated first, then random playouts) to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vectors = build_vectors(n, seed)
    with open(path, "w", encoding="utf-8") as f:  # one compact vector per line
        f.write("[\n")
        f.writelines(json.dumps(vec, separators=(",", ":")) + (",\n" if i + 1 < len(vectors) else "\n") for i, vec in enumerate(vectors))
        f.write("]\n")
    return path


if __name__ == "__main__":
    out = write_encoding_vectors()
    print(f"wrote {out}")
