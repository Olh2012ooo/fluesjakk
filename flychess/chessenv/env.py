"""ChessEnv: a thin gym-style wrapper around ``chess.Board`` using the SPEC §3 encoding.

Observations are ``float32[20, 8, 8]`` planes from the side to move's perspective, actions are
either ``chess.Move`` objects, UCI strings or move indices (``move_to_index``), rewards/values are
always from the perspective of the player who just moved (+1 win, 0 draw, -1 loss).
"""
from __future__ import annotations

import chess
import chess.pgn
import numpy as np

from .encoding import encode_board, index_to_move, legal_move_mask, move_to_index

_RESULT_VALUE = {"1-0": 1.0, "0-1": -1.0, "1/2-1/2": 0.0}


class ChessEnv:
    """Single-game chess environment.

    ``step`` returns ``(obs, reward, done, info)`` where ``reward`` is the game outcome from the
    perspective of the player who just made the move (0 while the game is running) and ``obs`` is
    the encoding for the *next* side to move.  Game end follows ``chess.Board.is_game_over(claim_draw=True)``
    (checkmate, stalemate, insufficient material, 75-move / fivefold rules and claimable 50-move /
    threefold draws), so self-play games always terminate.
    """

    def __init__(self, fen: str | None = None, claim_draw: bool = True) -> None:
        self.claim_draw = claim_draw
        self.board = chess.Board()
        self.start_fen = chess.STARTING_FEN
        self.reset(fen)

    # ---- lifecycle -----------------------------------------------------------------------
    def reset(self, fen: str | None = None) -> np.ndarray:
        """Start a new game (from ``fen`` or the standard start) and return the first observation."""
        self.board = chess.Board(fen) if fen else chess.Board()
        self.start_fen = self.board.fen()
        return self.observe()

    def observe(self) -> np.ndarray:
        return encode_board(self.board)

    def legal_mask(self) -> np.ndarray:
        return legal_move_mask(self.board)

    def legal_moves(self) -> list[chess.Move]:
        return list(self.board.legal_moves)

    def to_move(self, action: chess.Move | str | int | np.integer) -> chess.Move:
        """Normalise an action (Move / UCI / index) into a ``chess.Move`` for the current board."""
        if isinstance(action, chess.Move):
            return action
        if isinstance(action, str):
            return chess.Move.from_uci(action)
        return index_to_move(int(action), self.board)

    def step(self, action: chess.Move | str | int | np.integer) -> tuple[np.ndarray, float, bool, dict]:
        """Play ``action``; returns ``(obs, reward_for_mover, done, info)``."""
        if self.is_over:
            raise RuntimeError("game is over; call reset()")
        move = self.to_move(action)
        if move not in self.board.legal_moves:
            raise ValueError(f"illegal move {move.uci()} in {self.board.fen()}")
        mover = self.board.turn
        index = move_to_index(move, self.board)
        san = self.board.san(move)
        self.board.push(move)
        done = self.is_over
        reward = self.result_value(mover) if done else 0.0
        info = {"san": san, "uci": move.uci(), "index": index, "fen": self.board.fen(),
                "result": self.result() if done else "*"}
        return self.observe(), reward, done, info

    # ---- state ---------------------------------------------------------------------------
    @property
    def turn(self) -> chess.Color:
        return self.board.turn

    @property
    def is_over(self) -> bool:
        return self.board.is_game_over(claim_draw=self.claim_draw)

    def result(self) -> str:
        """``'1-0'``, ``'0-1'``, ``'1/2-1/2'`` or ``'*'`` while running."""
        return self.board.result(claim_draw=self.claim_draw)

    def result_value(self, perspective: chess.Color) -> float:
        """+1 if ``perspective`` won, -1 if it lost, 0 for a draw or an unfinished game."""
        value = _RESULT_VALUE.get(self.result(), 0.0)
        return value if perspective == chess.WHITE else -value

    @property
    def ply(self) -> int:
        return len(self.board.move_stack)

    def san_history(self) -> list[str]:
        board = chess.Board(self.start_fen)
        sans = []
        for move in self.board.move_stack:
            sans.append(board.san(move))
            board.push(move)
        return sans

    def uci_history(self) -> list[str]:
        return [m.uci() for m in self.board.move_stack]

    def pgn(self, headers: dict[str, str] | None = None) -> str:
        """PGN text of the game so far (with a FEN/SetUp header when not started from the initial position)."""
        game = chess.pgn.Game()
        if self.start_fen != chess.STARTING_FEN:
            game.setup(chess.Board(self.start_fen))
        game.headers["Result"] = self.result()
        for k, v in (headers or {}).items():
            game.headers[k] = v
        node = game
        for move in self.board.move_stack:
            node = node.add_variation(move)
        return str(game)

    def __repr__(self) -> str:
        return f"ChessEnv(fen={self.board.fen()!r}, ply={self.ply}, result={self.result()!r})"
