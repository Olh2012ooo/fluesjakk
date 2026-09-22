"""Board -> planes and move <-> index encoding (docs/SPEC.md §3).

Everything is expressed from the side to move's perspective: when black is to move the board is
mirrored (colours swapped, ranks flipped -- ``chess.Board.mirror()`` semantics, squares ``sq ^ 56``)
so that the mover is always "white" pushing pawns up the board.  ``web/engine/encoding.js`` is a
line-by-line port of this module; ``tests/vectors/encoding.json`` pins both down.

Planes (``[plane, rank, file]``, a1 = ``[0, 0]``):
    0-5   mover's P N B R Q K        6-11  opponent's P N B R Q K
    12/13 mover castles K/Q          14/15 opponent castles K/Q           (all-ones planes)
    16    en-passant target square (single 1, only when an en-passant capture is legal)
    17    constant ones              18    halfmove clock / 100 clipped to 1 (all squares)
    19    1 if the current position already occurred earlier in the game (all squares)

Move index (perspective coordinates):
    from * 64 + to                                  non-promotion or queen promotion   (0 .. 4095)
    4096 + (from_file * 3 + dir) * 3 + piece        under-promotion, dir 0/1/2 = capture-left / push /
                                                    capture-right, piece 0/1/2 = N / B / R (4096 .. 4167)
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence

import chess
import numpy as np

NUM_PLANES = 20
FLAT_INPUT = NUM_PLANES * 64  # 1280
NUM_MOVES = 4096 + 72  # 4168
UNDERPROMOTION_BASE = 4096

# Feature vector layout produced by `board_features` (one row per position, dtype uint64):
#   0-11  piece bitboards in perspective (mover P N B R Q K, opponent P N B R Q K)
#   12-15 castling flags (mover K, mover Q, opponent K, opponent Q)
#   16    en-passant square in perspective coordinates, or NO_EP
#   17    halfmove clock
#   18    repetition flag
NUM_FEATURES = 19
NO_EP = 64

_PIECE_TYPES = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)
_UNDERPROMOTION_PIECES = {chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2}
_UNDERPROMOTION_BY_CODE = (chess.KNIGHT, chess.BISHOP, chess.ROOK)
_BIT_SHIFTS = np.arange(64, dtype=np.uint64)


# ---------------------------------------------------------------------------------------------
# Position identity / repetition
# ---------------------------------------------------------------------------------------------
def transposition_key(board: chess.Board) -> tuple:
    """Hashable identity of a position for repetition purposes.

    Pieces, side to move, (cleaned) castling rights and the en-passant square only if an
    en-passant capture is actually legal -- the same notion python-chess uses for
    ``is_repetition`` and the one ``chess.js``'s ``fen()`` (first four fields) exposes.
    """
    return (
        board.pawns, board.knights, board.bishops, board.rooks, board.queens, board.kings,
        board.occupied_co[chess.WHITE], board.occupied_co[chess.BLACK],
        board.turn, board.clean_castling_rights(),
        board.ep_square if board.has_legal_en_passant() else None,
    )


def position_repeated(board: chess.Board) -> bool:
    """True if the current position already occurred earlier in ``board``'s move stack."""
    return board.is_repetition(2)


# ---------------------------------------------------------------------------------------------
# Board -> features -> planes
# ---------------------------------------------------------------------------------------------
def board_features(board: chess.Board, repeated: bool | None = None) -> np.ndarray:
    """Compact perspective features of ``board`` as a ``uint64[NUM_FEATURES]`` row.

    ``repeated`` overrides the repetition flag (callers replaying a game can track it themselves);
    ``None`` computes it from the board's move stack.
    """
    mover = board.turn
    opp = not mover
    if mover == chess.WHITE:
        flip = lambda bb: bb
    else:
        flip = chess.flip_vertical
    feats = np.zeros(NUM_FEATURES, dtype=np.uint64)
    for i, pt in enumerate(_PIECE_TYPES):
        feats[i] = flip(board.pieces_mask(pt, mover))
        feats[6 + i] = flip(board.pieces_mask(pt, opp))
    feats[12] = board.has_kingside_castling_rights(mover)
    feats[13] = board.has_queenside_castling_rights(mover)
    feats[14] = board.has_kingside_castling_rights(opp)
    feats[15] = board.has_queenside_castling_rights(opp)
    if board.ep_square is not None and board.has_legal_en_passant():
        feats[16] = board.ep_square if mover == chess.WHITE else board.ep_square ^ 56
    else:
        feats[16] = NO_EP
    feats[17] = board.halfmove_clock
    feats[18] = position_repeated(board) if repeated is None else bool(repeated)
    return feats


def planes_from_features(feats: np.ndarray, quantized: bool = False) -> np.ndarray:
    """Expand ``(B, NUM_FEATURES) uint64`` features into ``(B, 20, 8, 8)`` planes.

    ``quantized=False`` -> float32 exactly as the model consumes them.
    ``quantized=True``  -> uint8 shard storage: planes 0-17 and 19 are 0/1, plane 18 is
    ``(min(halfmove, 100) * 255 + 50) // 100`` (round half up of halfmove / 100 * 255).
    """
    feats = np.asarray(feats, dtype=np.uint64)
    if feats.ndim == 1:
        feats = feats[None]
    b = feats.shape[0]
    out = np.zeros((b, NUM_PLANES, 8, 8), dtype=np.uint8 if quantized else np.float32)
    bits = (feats[:, :12, None] >> _BIT_SHIFTS[None, None, :]) & np.uint64(1)  # (B, 12, 64)
    out[:, :12] = bits.reshape(b, 12, 8, 8)
    out[:, 12:16] = (feats[:, 12:16] != 0)[:, :, None, None]
    ep = feats[:, 16].astype(np.int64)
    has_ep = ep < NO_EP
    rows = np.nonzero(has_ep)[0]
    out[rows, 16, ep[rows] >> 3, ep[rows] & 7] = 1
    out[:, 17] = 1
    halfmove = np.minimum(feats[:, 17].astype(np.int64), 100)
    if quantized:  # integer round-half-up of halfmove / 100 * 255
        out[:, 18] = ((halfmove * 255 + 50) // 100).astype(np.uint8)[:, None, None]
    else:
        out[:, 18] = (halfmove / 100.0).astype(np.float32)[:, None, None]
    out[:, 19] = (feats[:, 18] != 0)[:, None, None]
    return out


def encode_board(board: chess.Board, repeated: bool | None = None) -> np.ndarray:
    """``float32[NUM_PLANES, 8, 8]`` planes of ``board`` from the mover's perspective."""
    return planes_from_features(board_features(board, repeated)[None])[0]


def encode_boards(boards: Sequence[chess.Board] | Iterable[chess.Board]) -> np.ndarray:
    """Batch version of ``encode_board`` -> ``float32[B, NUM_PLANES, 8, 8]`` (single vectorised expand)."""
    boards = list(boards)
    if not boards:
        return np.zeros((0, NUM_PLANES, 8, 8), dtype=np.float32)
    feats = np.stack([board_features(b) for b in boards])
    return planes_from_features(feats)


def flatten_planes(planes: np.ndarray) -> np.ndarray:
    """``(..., 20, 8, 8)`` -> ``(..., 1280)`` in C order (plane-major)."""
    planes = np.ascontiguousarray(planes)
    return planes.reshape(*planes.shape[:-3], FLAT_INPUT)


# ---------------------------------------------------------------------------------------------
# Move <-> index
# ---------------------------------------------------------------------------------------------
def _perspective_square(square: int, turn: chess.Color) -> int:
    return square if turn == chess.WHITE else square ^ 56


def move_to_index(move: chess.Move, board: chess.Board) -> int:
    """Index in ``[0, NUM_MOVES)`` of ``move`` played on ``board`` (perspective of ``board.turn``)."""
    frm = _perspective_square(move.from_square, board.turn)
    to = _perspective_square(move.to_square, board.turn)
    promo = move.promotion
    if promo is None or promo == chess.QUEEN:
        return frm * 64 + to
    from_file = frm & 7
    direction = (to & 7) - from_file + 1
    if direction not in (0, 1, 2) or (frm >> 3) != 6 or (to >> 3) != 7:
        raise ValueError(f"not an encodable under-promotion: {move.uci()}")
    return UNDERPROMOTION_BASE + (from_file * 3 + direction) * 3 + _UNDERPROMOTION_PIECES[promo]


def index_to_move(idx: int, board: chess.Board) -> chess.Move:
    """Inverse of ``move_to_index``. Queen promotion is inferred from a pawn reaching the last rank."""
    idx = int(idx)
    if not 0 <= idx < NUM_MOVES:
        raise ValueError(f"move index out of range: {idx}")
    turn = board.turn
    if idx < UNDERPROMOTION_BASE:
        frm, to = divmod(idx, 64)
        promotion = None
        if (to >> 3) == 7 and (frm >> 3) == 6:
            piece = board.piece_type_at(_perspective_square(frm, turn))
            if piece == chess.PAWN:
                promotion = chess.QUEEN
        return chess.Move(_perspective_square(frm, turn), _perspective_square(to, turn), promotion)
    rem = idx - UNDERPROMOTION_BASE
    from_file, rest = divmod(rem, 9)
    direction, piece_code = divmod(rest, 3)
    frm = 6 * 8 + from_file
    to = 7 * 8 + from_file + direction - 1
    if not 0 <= (from_file + direction - 1) <= 7:
        raise ValueError(f"under-promotion index off the board: {idx}")
    return chess.Move(_perspective_square(frm, turn), _perspective_square(to, turn),
                      _UNDERPROMOTION_BY_CODE[piece_code])


def legal_move_indices(board: chess.Board) -> np.ndarray:
    """Sorted ``int64`` indices of every legal move."""
    return np.array(sorted(move_to_index(m, board) for m in board.legal_moves), dtype=np.int64)


def legal_move_mask(board: chess.Board) -> np.ndarray:
    """``bool[NUM_MOVES]`` with True at every legal move index."""
    mask = np.zeros(NUM_MOVES, dtype=bool)
    for move in board.legal_moves:
        mask[move_to_index(move, board)] = True
    return mask


__all__ = [
    "FLAT_INPUT",
    "NO_EP",
    "NUM_FEATURES",
    "NUM_MOVES",
    "NUM_PLANES",
    "UNDERPROMOTION_BASE",
    "board_features",
    "encode_board",
    "encode_boards",
    "flatten_planes",
    "index_to_move",
    "legal_move_indices",
    "legal_move_mask",
    "move_to_index",
    "planes_from_features",
    "position_repeated",
    "transposition_key",
]
