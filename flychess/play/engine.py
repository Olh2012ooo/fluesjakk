"""FlyEngine: the fly brain choosing moves at three difficulties (docs/SPEC.md §9).

* ``larva``     sample the legal-masked policy with temperature 1.2 (no search)
* ``fly``       argmax policy with a 1-ply value check: the three most likely moves are each played on
                a scratch board, the resulting positions are evaluated by the brain in one batch and the
                move that leaves the opponent with the *lowest* value is chosen (a mate is a -1 for the
                opponent without asking the network; a stalemate / dead draw is 0)
* ``superfly``  PUCT MCTS with ``sims`` (200) simulations, every leaf evaluated by the brain

All three use the fly brain and nothing else; ``info['mood']`` turns the value head into the fly's
mood (``smug > 0.7``, ``confident > 0.3``, ``nervous < -0.3``, ``panicking < -0.7``, else ``focused``).
"""
from __future__ import annotations

import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import chess
import numpy as np
import torch

from flychess.chessenv.encoding import (
    board_features,
    index_to_move,
    move_to_index,
    planes_from_features,
)
from flychess.connectome.graph import BrainGraph
from flychess.model.config import BrainConfig
from flychess.model.flybrain import FlyBrain
from flychess.train.mcts import BatchedMCTS, leaf_status

DIFFICULTIES = ("larva", "fly", "superfly")
LARVA_TEMPERATURE = 1.2
FLY_CANDIDATES = 3
SUPERFLY_SIMS = 200


def mood_from_value(value: float) -> str:
    """The fly's mood as a function of the value head (mover's perspective)."""
    if value > 0.7:
        return "smug"
    if value > 0.3:
        return "confident"
    if value < -0.7:
        return "panicking"
    if value < -0.3:
        return "nervous"
    return "focused"


def board_with_history(board: chess.Board, history: Iterable[chess.Move | str] | None) -> chess.Board:
    """``board`` itself, or a board rebuilt from ``history`` (needed for the repetition plane) when given."""
    if history is None:
        return board
    b = chess.Board()
    for mv in history:
        b.push(mv if isinstance(mv, chess.Move) else chess.Move.from_uci(mv))
    if b.board_fen() != board.board_fen() or b.turn != board.turn:
        raise ValueError("history does not lead to the given board")
    return b


def masked_probs(logits: np.ndarray, legal: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Softmax of ``logits[legal] / temperature`` (float64, sums to 1) — the policy restricted to legal moves."""
    z = logits[legal].astype(np.float64) / max(float(temperature), 1e-6)
    z -= z.max()
    p = np.exp(z)
    return p / p.sum()


class _FlyPlayer:
    """``Player``-protocol adapter (``name``, ``choose(board) -> Move``) around a :class:`FlyEngine`."""

    def __init__(self, engine: FlyEngine, difficulty: str) -> None:
        self.engine = engine
        self.difficulty = difficulty
        self.name = f"fly-{difficulty}"
        self.last_info: dict[str, Any] | None = None

    def choose(self, board: chess.Board) -> chess.Move:
        move, info = self.engine.choose_move(board, self.difficulty)
        self.last_info = info
        return move

    def choose_many(self, boards: list[chess.Board]) -> list[chess.Move]:
        """Batched ``choose`` (used by ``flychess.eval.elo.play_match`` to evaluate all boards at once)."""
        out = self.engine.choose_moves(list(boards), self.difficulty)
        if out:
            self.last_info = out[-1][1]
        return [mv for mv, _ in out]

    @property
    def rng(self) -> np.random.Generator:
        return self.engine.rng

    @rng.setter
    def rng(self, value: np.random.Generator) -> None:  # play_match reseeds players through this
        self.engine.rng = value
        self.engine.mcts.rng = value

    def __repr__(self) -> str:
        return f"FlyPlayer({self.name})"


class FlyEngine:
    """Move chooser built on a :class:`FlyBrain`.

    ``sims`` is the number of MCTS simulations used by ``superfly`` (200 by default; tests use fewer).
    """

    def __init__(self, model: FlyBrain, graph: BrainGraph | None = None, device: str | torch.device | None = None,
                 sims: int = SUPERFLY_SIMS, seed: int | None = None, amp: bool = False) -> None:
        self.model = model.eval()
        self.graph = graph
        self.device = torch.device(device) if device is not None else next(model.parameters()).device
        if next(model.parameters()).device != self.device:
            self.model.to(self.device)
        self.sims = int(sims)
        self.rng = np.random.default_rng(seed)
        self.amp = bool(amp) and self.device.type == "cuda"
        self.mcts = BatchedMCTS(self.model, self.device, sims=self.sims, c_puct=1.5, dirichlet_alpha=0.0,
                                temperature=0.0, rng=self.rng, amp=self.amp)

    # ---- raw network ---------------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, boards: list[chess.Board]) -> tuple[np.ndarray, np.ndarray]:
        """Policy logits ``(B, NUM_MOVES)`` and values ``(B,)`` (mover's perspective) for ``boards``."""
        feats = np.stack([board_features(b) for b in boards])
        x = torch.from_numpy(planes_from_features(feats).reshape(len(boards), -1)).to(self.device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.amp):
            logits, value = self.model(x)
        return logits.float().cpu().numpy(), value.float().view(-1).cpu().numpy()

    def policy(self, board: chess.Board, temperature: float = 1.0
               ) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        """``(legal_idx, probs over legal_idx, value, raw logits)`` — the legal-masked softmax at ``temperature``."""
        _, legal = leaf_status(board)
        if legal is None:
            raise ValueError("game is over: no legal moves")
        logits, values = self.evaluate([board])
        return legal, masked_probs(logits[0], legal, temperature), float(values[0]), logits[0]

    # ---- choosing ------------------------------------------------------------------------------
    def choose_move(self, board: chess.Board, difficulty: str = "fly",
                    history: Iterable[chess.Move | str] | None = None) -> tuple[chess.Move, dict[str, Any]]:
        """Pick a move for the side to move; returns ``(move, info)``.

        ``info``: ``policy_top`` (top-5 ``[(uci, prob)]`` of the raw policy), ``value`` (before the move,
        mover's perspective), ``sims``, ``think_ms``, ``mood``, ``difficulty``, ``move_prob`` (policy
        probability of the chosen move), plus ``search_top`` / ``search_value`` for ``superfly`` and
        ``candidates`` (``[(uci, opponent value)]``) for ``fly``.
        """
        return self.choose_moves([board_with_history(board, history)], difficulty)[0]

    def choose_moves(self, boards: list[chess.Board], difficulty: str = "fly"
                     ) -> list[tuple[chess.Move, dict[str, Any]]]:
        """Batched :meth:`choose_move`: one network call for all boards (plus one for the ``fly`` value check;
        ``superfly`` searches all boards in lockstep). Every board must have a legal move."""
        if difficulty not in DIFFICULTIES:
            raise ValueError(f"difficulty must be one of {DIFFICULTIES}, got {difficulty!r}")
        if not boards:
            return []
        t0 = time.perf_counter()
        status = [leaf_status(b) for b in boards]
        if any(legal is None for _, legal in status):
            raise ValueError("game is over: no legal moves")
        legal = [lg for _, lg in status]
        logits, values = self.evaluate(boards)
        probs = [masked_probs(logits[i], legal[i], 1.0) for i in range(len(boards))]
        orders = [np.argsort(-p) for p in probs]
        infos: list[dict[str, Any]] = []
        for i, board in enumerate(boards):
            top = [(index_to_move(int(legal[i][k]), board).uci(), float(probs[i][k])) for k in orders[i][:5]]
            infos.append({"difficulty": difficulty, "policy_top": top, "value": float(values[i]), "sims": 0,
                          "mood": mood_from_value(float(values[i]))})
        moves: list[chess.Move]
        if difficulty == "larva":
            moves = []
            for i, board in enumerate(boards):
                p_t = masked_probs(logits[i], legal[i], LARVA_TEMPERATURE)
                moves.append(index_to_move(int(legal[i][self.rng.choice(len(legal[i]), p=p_t)]), board))
        elif difficulty == "fly":
            cands = [[index_to_move(int(legal[i][k]), b) for k in orders[i][:FLY_CANDIDATES]] for i, b in enumerate(boards)]
            moves = []
            for (mv, ranked), info in zip(self._value_check_many(boards, cands), infos, strict=True):
                moves.append(mv)
                info["candidates"] = ranked
        else:
            results = self.mcts.search(boards, add_noise=False, sims=self.sims)
            moves = []
            for board, res, info in zip(boards, results, infos, strict=True):
                moves.append(index_to_move(res.best_index(), board))
                info["sims"] = int(res.sims)
                info["search_top"] = res.top(board, 5)
                info["search_value"] = float(res.root_value)
        think_ms = (time.perf_counter() - t0) * 1000.0
        out = []
        for i, (board, mv, info) in enumerate(zip(boards, moves, infos, strict=True)):
            info["think_ms"] = think_ms
            info["move_prob"] = self._prob_of(legal[i], probs[i], mv, board)
            out.append((mv, info))
        return out

    @staticmethod
    def _prob_of(legal: np.ndarray, probs: np.ndarray, move: chess.Move, board: chess.Board) -> float:
        idx = move_to_index(move, board)
        hit = np.nonzero(legal == idx)[0]
        return float(probs[hit[0]]) if hit.size else 0.0

    def _value_check(self, board: chess.Board, candidates: list[chess.Move]
                     ) -> tuple[chess.Move, list[tuple[str, float]]]:
        """Play each candidate, evaluate the replies' positions with the brain, keep the worst for the opponent."""
        return self._value_check_many([board], [candidates])[0]

    def _value_check_many(self, boards: list[chess.Board], candidates: list[list[chess.Move]]
                          ) -> list[tuple[chess.Move, list[tuple[str, float]]]]:
        """Batched 1-ply value check: one forward for every non-terminal candidate position of every board.

        A candidate that mates is a -1 for the opponent, a stalemate / dead draw a 0 — without asking
        the network (it never sees a position without legal moves).
        """
        opp_values: list[list[float | None]] = []
        to_eval: list[tuple[int, int, chess.Board]] = []
        for i, (board, cands) in enumerate(zip(boards, candidates, strict=True)):
            row: list[float | None] = []
            for j, mv in enumerate(cands):
                b = board.copy(stack=True)
                b.push(mv)
                tv, _ = leaf_status(b)
                row.append(tv)
                if tv is None:
                    to_eval.append((i, j, b))
            opp_values.append(row)
        if to_eval:
            _, values = self.evaluate([b for _, _, b in to_eval])
            for (i, j, _), v in zip(to_eval, values, strict=True):
                opp_values[i][j] = float(v)
        out = []
        for cands, vals in zip(candidates, opp_values, strict=True):
            ranked = sorted(((mv.uci(), float(v)) for mv, v in zip(cands, vals, strict=True)), key=lambda t: t[1])
            best = min(range(len(cands)), key=lambda j: vals[j])
            out.append((cands[best], ranked))
        return out

    # ---- adapters ------------------------------------------------------------------------------
    def as_player(self, difficulty: str = "fly") -> _FlyPlayer:
        """A ``Player`` (``name='fly-<difficulty>'``, ``choose(board) -> Move``) for the evaluation code."""
        if difficulty not in DIFFICULTIES:
            raise ValueError(f"difficulty must be one of {DIFFICULTIES}, got {difficulty!r}")
        return _FlyPlayer(self, difficulty)

    @classmethod
    def load(cls, run_or_ckpt: str | Path, device: str | torch.device | None = None, **kw: Any) -> FlyEngine:
        """Build an engine from a run name or checkpoint path (see :func:`load_checkpoint`)."""
        model, graph, _ = load_checkpoint(run_or_ckpt, device)
        return cls(model, graph, device, **kw)


# ------------------------------------------------------------------------------------------------
# Checkpoint loading
# ------------------------------------------------------------------------------------------------
def resolve_checkpoint(run_or_ckpt: str | Path) -> Path:
    """A run name → ``runs/<run>/latest.pt``; a path is returned as is (directories → ``latest.pt`` inside)."""
    from flychess import paths

    p = Path(run_or_ckpt)
    if p.is_file():
        return p
    if p.is_dir():
        return p / "latest.pt"
    return paths.run_dir(str(run_or_ckpt)) / "latest.pt"


def load_checkpoint(run_or_ckpt: str | Path, device: str | torch.device | None = None
                    ) -> tuple[FlyBrain, BrainGraph, dict[str, Any]]:
    """``(model, graph, ckpt)`` from a run name or checkpoint path (SPEC §6 checkpoint format).

    Delegates to ``flychess.train.trainer.load_checkpoint`` when that module is available and falls
    back to reading the shared checkpoint format directly otherwise.
    """
    device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        from flychess.train.trainer import load_checkpoint as _trainer_load
    except ImportError:
        _trainer_load = None
    if _trainer_load is not None:
        model, graph, ckpt = _trainer_load(run_or_ckpt, device)
        return model.to(device).eval(), graph, ckpt
    path = resolve_checkpoint(run_or_ckpt)
    if not path.exists():
        raise FileNotFoundError(f"no checkpoint at {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    graph = BrainGraph.load(ckpt["graph_path"])
    bc = ckpt["brain_config"]
    config = bc if isinstance(bc, BrainConfig) else BrainConfig.from_dict(dict(bc))
    model = FlyBrain.from_checkpoint(ckpt["model"], config, graph).to(device).eval()
    return model, graph, ckpt


__all__ = ["DIFFICULTIES", "FlyEngine", "board_with_history", "load_checkpoint", "masked_probs", "mood_from_value",
           "resolve_checkpoint"]
