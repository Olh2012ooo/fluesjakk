"""Players for evaluation matches.

Only :class:`BrainPolicyPlayer` is the fly: it picks moves exclusively from the FlyBrain policy.
:class:`RandomPlayer`, :class:`GreedyMaterialPlayer` and :class:`StockfishPlayer` are *opponents*
used to estimate the fly's strength; they are never presented as the fly.
"""
from __future__ import annotations

import shutil
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import chess
import numpy as np
import torch

from flychess.chessenv.encoding import (
    NUM_MOVES,
    encode_boards,
    flatten_planes,
    index_to_move,
    legal_move_mask,
)
from flychess.model.flybrain import FlyBrain

PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


@runtime_checkable
class Player(Protocol):
    """Anything that can pick a legal move for the side to move."""

    name: str

    def choose(self, board: chess.Board) -> chess.Move: ...


def _no_moves(board: chess.Board) -> ValueError:
    return ValueError(f"no legal moves in {board.fen()} (game over?)")


# ---------------------------------------------------------------------------------------------------
# Opponents
# ---------------------------------------------------------------------------------------------------
class RandomPlayer:
    """Uniformly random legal move (seeded)."""

    def __init__(self, seed: int = 0, name: str = "random") -> None:
        self.name = name
        self.rng = np.random.default_rng(seed)

    def choose(self, board: chess.Board) -> chess.Move:
        moves = list(board.legal_moves)
        if not moves:
            raise _no_moves(board)
        return moves[int(self.rng.integers(len(moves)))]


def material_balance(board: chess.Board, color: chess.Color) -> int:
    """Material of ``color`` minus material of the opponent (P=1, N=B=3, R=5, Q=9)."""
    total = 0
    for piece_type, value in PIECE_VALUES.items():
        if value:
            total += value * (len(board.pieces(piece_type, color)) - len(board.pieces(piece_type, not color)))
    return total


class GreedyMaterialPlayer:
    """1-ply greedy: the move maximising the mover's material balance after it is played; random tie-break."""

    def __init__(self, seed: int = 0, name: str = "material") -> None:
        self.name = name
        self.rng = np.random.default_rng(seed)

    def choose(self, board: chess.Board) -> chess.Move:
        moves = list(board.legal_moves)
        if not moves:
            raise _no_moves(board)
        mover = board.turn
        best, best_moves = None, []
        for move in moves:
            board.push(move)
            score = material_balance(board, mover)
            board.pop()
            if best is None or score > best:
                best, best_moves = score, [move]
            elif score == best:
                best_moves.append(move)
        return best_moves[int(self.rng.integers(len(best_moves)))]


class StockfishPlayer:
    """Stockfish via ``chess.engine`` (only when the binary is installed; see :meth:`available`)."""

    def __init__(self, depth: int = 1, skill: int = 0, path: str | None = None, name: str | None = None,
                 time_limit: float | None = None) -> None:
        import chess.engine

        path = path or shutil.which("stockfish")
        if not path:
            raise FileNotFoundError("stockfish binary not found on PATH")
        self.name = name or f"stockfish-d{depth}-s{skill}"
        self.depth, self.time_limit = depth, time_limit
        self.engine = chess.engine.SimpleEngine.popen_uci(path)
        try:
            self.engine.configure({"Skill Level": int(skill)})
        except Exception:  # noqa: BLE001, S110 - older builds lack the option
            pass

    @staticmethod
    def available() -> bool:
        return shutil.which("stockfish") is not None

    def choose(self, board: chess.Board) -> chess.Move:
        import chess.engine

        if not any(board.legal_moves):
            raise _no_moves(board)
        limit = chess.engine.Limit(depth=self.depth, time=self.time_limit)
        result = self.engine.play(board, limit)
        assert result.move is not None
        return result.move

    def close(self) -> None:
        try:
            self.engine.quit()
        except Exception:  # noqa: BLE001, S110 - engine already gone
            pass

    def __del__(self) -> None:  # pragma: no cover
        self.close()


# ---------------------------------------------------------------------------------------------------
# The fly
# ---------------------------------------------------------------------------------------------------
class BrainPolicyPlayer:
    """The fly brain choosing moves from its policy head alone (no search).

    ``temperature == 0`` plays the arg-max legal move; otherwise a legal move is sampled from
    ``softmax(logits / temperature)``. :meth:`choose_many` evaluates many boards in one forward pass
    (used by :func:`flychess.eval.elo.play_match` to play all games of a match in lockstep).
    """

    def __init__(self, model: FlyBrain, device: str | torch.device | None = None, temperature: float = 0.0,
                 name: str = "fly-policy", seed: int = 0, amp: bool = False) -> None:
        self.model = model
        self.device = torch.device(device) if device is not None else next(model.parameters()).device
        self.temperature = float(temperature)
        self.name = name
        self.amp = bool(amp)
        self.generator = torch.Generator(device="cpu").manual_seed(int(seed))

    @torch.inference_mode()
    def policy(self, boards: Sequence[chess.Board]) -> tuple[np.ndarray, np.ndarray]:
        """``(legal-masked probabilities float32[B, NUM_MOVES], values float32[B])`` for ``boards``."""
        if not boards:
            return np.zeros((0, NUM_MOVES), np.float32), np.zeros(0, np.float32)
        x = torch.from_numpy(flatten_planes(encode_boards(boards))).to(self.device, non_blocking=True)
        mask = torch.from_numpy(np.stack([legal_move_mask(b) for b in boards])).to(self.device)
        if not bool(mask.any(dim=1).all()):
            raise _no_moves(boards[int((~mask.any(dim=1)).nonzero()[0])])
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.autocast(self.device.type, dtype=torch.bfloat16,
                                enabled=self.amp and self.device.type == "cuda"):
                logits, value = self.model(x)[:2]
        finally:
            self.model.train(was_training)
        logits = logits.float().masked_fill(~mask, float("-inf"))
        if self.temperature > 0:
            logits = logits / self.temperature
        probs = torch.softmax(logits, dim=-1)
        return probs.cpu().numpy(), value.float().view(-1).cpu().numpy()

    def choose_many(self, boards: Sequence[chess.Board]) -> list[chess.Move]:
        probs, _ = self.policy(boards)
        if self.temperature > 0:
            idx = torch.multinomial(torch.from_numpy(probs), 1, generator=self.generator).view(-1).tolist()
        else:
            idx = probs.argmax(axis=1).tolist()
        return [index_to_move(i, b) for i, b in zip(idx, boards, strict=True)]

    def choose(self, board: chess.Board) -> chess.Move:
        return self.choose_many([board])[0]


def make_opponent(spec: str, seed: int = 0) -> Player:
    """``'random'`` / ``'material'`` / ``'stockfish[:depth[:skill]]'`` → a Player."""
    if spec == "random":
        return RandomPlayer(seed=seed)
    if spec == "material":
        return GreedyMaterialPlayer(seed=seed)
    if spec.startswith("stockfish"):
        parts = spec.split(":")
        depth = int(parts[1]) if len(parts) > 1 else 1
        skill = int(parts[2]) if len(parts) > 2 else 0
        return StockfishPlayer(depth=depth, skill=skill)
    raise ValueError(f"unknown opponent {spec!r}")


__all__ = [
    "PIECE_VALUES",
    "BrainPolicyPlayer",
    "GreedyMaterialPlayer",
    "Player",
    "RandomPlayer",
    "StockfishPlayer",
    "make_opponent",
    "material_balance",
]
