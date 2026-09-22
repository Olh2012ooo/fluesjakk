"""Chess environment and the shared board/move encoding (docs/SPEC.md §3)."""
from .encoding import (  # noqa: F401
    FLAT_INPUT,
    NUM_MOVES,
    NUM_PLANES,
    encode_board,
    encode_boards,
    index_to_move,
    legal_move_indices,
    legal_move_mask,
    move_to_index,
    position_repeated,
)
from .env import ChessEnv  # noqa: F401
