"""Batched PUCT Monte-Carlo tree search driven by the fly brain (docs/SPEC.md §6 stage 2, §9 "superfly").

Many games are searched **in lockstep**: every simulation step selects one leaf in each active tree,
encodes all of those leaves at once and runs exactly ONE forward pass of :class:`FlyBrain` for the
whole batch (terminal leaves take the game result instead — a position without legal moves is never
sent to the network). Values are always from the perspective of the player to move at the node and
are negated at every backup step. Because each tree contributes a single leaf per step no virtual
loss is needed.

Trees are memory-light: a :class:`Node` stores its children as flat arrays (move index in the mover's
perspective, prior, visit count, accumulated value) plus a list of child ``Node`` objects created on
first visit. Boards are *not* stored per node — moves are pushed on the root board along the selected
path and popped afterwards, which also keeps the repetition plane (SPEC §3.2 plane 19) exact.

Subtree reuse: :meth:`BatchedMCTS.search` accepts previously returned root nodes (``roots=``) and
:func:`advance_root` returns the child node reached by a move, so a self-play game can keep the
part of the tree below the chosen move. Dirichlet noise is applied to a *copy* of the root priors,
so re-used roots get fresh noise each search. ``web/engine/mcts.js`` implements the same search
(c_puct 1.5, same value convention) for a single game.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import chess
import numpy as np
import torch

from flychess.chessenv.encoding import (
    NUM_MOVES,
    board_features,
    index_to_move,
    move_to_index,
    planes_from_features,
)
from flychess.model.flybrain import FlyBrain

TemperatureFn = Callable[[int], float]


# ------------------------------------------------------------------------------------------------
# Game termination (cheap version of ``Board.is_game_over(claim_draw=True)``)
# ------------------------------------------------------------------------------------------------
def terminal_value(board: chess.Board) -> float | None:
    """``None`` while the game goes on, else the outcome from the perspective of the side to move.

    Checkmate → -1 (the mover is mated); stalemate, insufficient material, the 50-move rule and a
    threefold repetition (``Board.is_repetition``: occupancy fast path, no full replay unless the
    occupancy really repeated) → 0. ``can_claim_threefold_repetition`` is deliberately avoided: it
    replays the whole move stack on every call, which dominated the search when it was used.
    """
    if board.is_checkmate():
        return -1.0
    if board.is_stalemate() or board.is_insufficient_material():
        return 0.0
    if board.halfmove_clock >= 100:
        return 0.0
    if board.halfmove_clock >= 8 and board.is_repetition(3):
        return 0.0
    return None


def leaf_status(board: chess.Board) -> tuple[float | None, np.ndarray | None]:
    """``(terminal_value, legal_move_indices)`` with a single legal-move generation.

    Same semantics as :func:`terminal_value`; ``legal`` is ``None`` when the position is terminal and
    an ``int64`` array of move indices (mover's perspective, generation order) otherwise.
    """
    moves = list(board.generate_legal_moves())
    if not moves:
        return (-1.0 if board.is_check() else 0.0), None
    if board.is_insufficient_material() or board.halfmove_clock >= 100:
        return 0.0, None
    if board.halfmove_clock >= 8 and board.is_repetition(3):
        return 0.0, None
    return None, np.fromiter((move_to_index(m, board) for m in moves), dtype=np.int64, count=len(moves))


def result_string(board: chess.Board) -> str:
    """``'1-0'`` / ``'0-1'`` / ``'1/2-1/2'`` for a position where :func:`terminal_value` is not ``None``."""
    tv = terminal_value(board)
    if tv is None:
        return "*"
    if tv < 0:
        return "0-1" if board.turn == chess.WHITE else "1-0"
    return "1/2-1/2"


# ------------------------------------------------------------------------------------------------
# Tree
# ------------------------------------------------------------------------------------------------
class Node:
    """A search node; the player to move here is *its* mover.

    ``moves``     int16 (k,) move indices (perspective of the node's mover) of the k legal moves
    ``prior``     float32 (k,) network policy over those moves (sums to 1)
    ``N``, ``W``  int32 / float32 (k,) visits and summed values of each child edge, values from the
                  perspective of THIS node's mover (i.e. the one choosing among the children)
    ``children``  list of ``Node | None`` (created when first visited)
    ``terminal``  ``None`` or the game result for this node's mover (-1 / 0)
    ``value``     the network's value estimate for this node's mover (set at expansion)
    """

    __slots__ = ("N", "W", "children", "moves", "prior", "terminal", "value")

    def __init__(self) -> None:
        self.moves: np.ndarray | None = None
        self.prior: np.ndarray | None = None
        self.N: np.ndarray | None = None
        self.W: np.ndarray | None = None
        self.children: list[Node | None] | None = None
        self.terminal: float | None = None
        self.value: float = 0.0

    @property
    def expanded(self) -> bool:
        return self.moves is not None

    def expand(self, legal_idx: np.ndarray, prior: np.ndarray, value: float) -> None:
        k = int(legal_idx.shape[0])
        self.moves = legal_idx.astype(np.int16, copy=False)
        self.prior = prior.astype(np.float32, copy=False)
        self.N = np.zeros(k, dtype=np.int32)
        self.W = np.zeros(k, dtype=np.float32)
        self.children = [None] * k
        self.value = float(value)

    @property
    def visits(self) -> int:
        return int(self.N.sum()) if self.N is not None else 0

    def q_values(self) -> np.ndarray:
        """Mean value of each child edge (0 for unvisited), from this node's mover's perspective."""
        return np.divide(self.W, self.N, out=np.zeros_like(self.W), where=self.N > 0)

    def visit_counts(self) -> np.ndarray:
        """Visit counts scattered over the full move space: ``float32[NUM_MOVES]``."""
        out = np.zeros(NUM_MOVES, dtype=np.float32)
        if self.moves is not None:
            out[self.moves.astype(np.int64)] = self.N
        return out


def advance_root(root: Node | None, move: chess.Move, board: chess.Board) -> Node | None:
    """The child of ``root`` reached by playing ``move`` on ``board`` (position *before* the move), if it exists."""
    if root is None or root.moves is None:
        return None
    idx = move_to_index(move, board)
    hits = np.nonzero(root.moves == idx)[0]
    if hits.size == 0:
        return None
    child = root.children[int(hits[0])]
    return child if child is not None and child.expanded else None


@dataclass
class SearchResult:
    """Outcome of one search on one board.

    ``visits``      float32 ``[NUM_MOVES]`` root visit counts (all zero if the root was terminal)
    ``root_value``  visit-weighted mean value of the root children (mover's perspective; the network
                    value if nothing was searched)
    ``prior_value`` the raw network value at the root (mover's perspective; the terminal result if terminal)
    ``root``        the root node (pass back as ``roots=`` to reuse the subtree)
    ``sims``        simulations actually run
    """

    visits: np.ndarray
    root_value: float
    prior_value: float
    root: Node
    sims: int

    @property
    def terminal(self) -> bool:
        return self.root.terminal is not None

    def policy(self) -> np.ndarray:
        """Normalised visit distribution ``float32[NUM_MOVES]`` (zeros if terminal)."""
        total = float(self.visits.sum())
        return self.visits / total if total > 0 else self.visits

    def ranking(self) -> np.ndarray:
        """Child slots of the root ordered by visits, then mean value, then prior (all descending)."""
        root = self.root
        if root.N is None:
            return np.zeros(0, dtype=np.int64)
        return np.lexsort((-root.prior, -root.q_values(), -root.N))

    def best_index(self) -> int:
        """Move index of the most visited root move (ties broken by Q, then prior); -1 if terminal."""
        order = self.ranking()
        return int(self.root.moves[order[0]]) if order.size else -1

    def top(self, board: chess.Board, k: int = 5) -> list[tuple[str, float]]:
        """``[(uci, visit share)]`` of the ``k`` most visited moves (same order as :meth:`ranking`)."""
        root = self.root
        total = float(root.N.sum()) if root.N is not None else 0.0
        if total <= 0:
            return []
        return [(index_to_move(int(root.moves[j]), board).uci(), float(root.N[j] / total))
                for j in self.ranking()[:k] if root.N[j] > 0]


def select_move_index(visits: np.ndarray, temperature: float, rng: np.random.Generator) -> int:
    """Pick a move index from root visit counts: argmax at T=0, else sample ∝ ``visits^(1/T)``."""
    if temperature <= 0:
        best = int(np.argmax(visits))
        return best
    nz = np.nonzero(visits > 0)[0]
    logits = np.log(visits[nz].astype(np.float64)) / temperature
    logits -= logits.max()
    p = np.exp(logits)
    p /= p.sum()
    return int(nz[rng.choice(nz.shape[0], p=p)])


# ------------------------------------------------------------------------------------------------
# Batched search
# ------------------------------------------------------------------------------------------------
class BatchedMCTS:
    """PUCT search over several boards with one network call per simulation step.

    Parameters
    ----------
    model            the fly brain (policy + value heads); it is the only evaluator
    device           where the model lives
    sims             simulations per search (each = one leaf per tree)
    c_puct           exploration constant
    dirichlet_alpha  root noise concentration (``<= 0`` disables noise)
    dirichlet_eps    mixing weight of the noise into the root priors
    temperature      a float or a ``ply -> float`` function used by :meth:`choose`
    rng              numpy generator for noise / sampling
    amp              run the dense parts under bf16 autocast on CUDA
    """

    def __init__(
        self,
        model: FlyBrain,
        device: str | torch.device | None = None,
        sims: int = 64,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_eps: float = 0.25,
        temperature: float | TemperatureFn = 0.0,
        rng: np.random.Generator | None = None,
        amp: bool = False,
    ) -> None:
        self.model = model
        self.device = torch.device(device) if device is not None else next(model.parameters()).device
        self.sims = int(sims)
        self.c_puct = float(c_puct)
        self.dirichlet_alpha = float(dirichlet_alpha)
        self.dirichlet_eps = float(dirichlet_eps)
        self.temperature = temperature
        self.rng = rng if rng is not None else np.random.default_rng()
        self.amp = bool(amp) and self.device.type == "cuda"
        self.evaluations = 0   # positions sent to the network (for throughput reporting)
        self.forwards = 0      # network calls

    # ---- network -------------------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, feats: np.ndarray, legal: Sequence[np.ndarray]) -> tuple[list[np.ndarray], np.ndarray]:
        """One forward pass for ``L`` positions.

        ``feats``: ``uint64 (L, NUM_FEATURES)`` from :func:`board_features`; ``legal[i]``: int64 legal move
        indices of position ``i`` (never empty). Returns ``(priors, values)`` with ``priors[i]`` the
        masked softmax restricted to ``legal[i]`` (float32, sums to 1) and ``values`` float32 ``(L,)``
        from each mover's perspective.

        The legal mask and the padded gather index are built with numpy and copied together with the
        planes (three H2D copies per step, no device-side scatter); probabilities of the legal moves and
        the values come back in a single D2H transfer. A non-finite policy or value (a diverged network)
        raises ``RuntimeError`` — there is no silent fallback: the fly's policy is the only prior.
        """
        L = feats.shape[0]
        lens = np.fromiter((len(li) for li in legal), dtype=np.int64, count=L)
        if L == 0 or int(lens.min()) == 0:
            raise ValueError("evaluate() needs at least one position, each with at least one legal move")
        kmax = int(lens.max())
        planes = planes_from_features(feats).reshape(L, -1)
        idx = np.zeros((L, kmax), dtype=np.int64)
        mask = np.zeros((L, NUM_MOVES), dtype=bool)
        for i, li in enumerate(legal):
            idx[i, : lens[i]] = li
            mask[i, li] = True
        x = torch.from_numpy(planes).to(self.device, non_blocking=True)
        mask_t = torch.from_numpy(mask).to(self.device, non_blocking=True)
        idx_t = torch.from_numpy(idx).to(self.device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.amp):
            logits, value = self.model(x)[:2]
        # every row has a legal move (checked above on the host), so the masked log-softmax is safe
        probs = torch.softmax(logits.float().masked_fill(~mask_t, float("-inf")), dim=-1)
        out = torch.cat([probs.gather(1, idx_t), value.float().view(L, 1)], dim=1).cpu().numpy()
        self.evaluations += L
        self.forwards += 1
        if not np.isfinite(out).all():
            raise RuntimeError("the fly brain returned a non-finite policy/value (diverged weights?)")
        values = np.ascontiguousarray(out[:, kmax], dtype=np.float32)
        priors = []
        for i in range(L):
            p = out[i, : lens[i]].astype(np.float32)
            s = float(p.sum())
            if not s > 0:
                raise RuntimeError("the fly brain gave zero probability to every legal move")
            priors.append(p / s)
        return priors, values

    def _expand_batch(self, pending: list[tuple[Node, np.ndarray, np.ndarray]]) -> list[float]:
        """Expand ``(node, feats, legal_idx)`` triples with one forward; returns each node's value."""
        feats = np.stack([p[1] for p in pending])
        legal = [p[2] for p in pending]
        priors, values = self.evaluate(feats, legal)
        for (node, _, li), pr, v in zip(pending, priors, values, strict=True):
            node.expand(li, pr, float(v))
        return [float(v) for v in values]

    # ---- selection / backup --------------------------------------------------------------------
    def _select(self, node: Node, prior: np.ndarray) -> int:
        n = node.N
        q = np.divide(node.W, n, out=np.zeros_like(node.W), where=n > 0)
        u = (self.c_puct * math.sqrt(max(1, int(n.sum())))) * prior / (1.0 + n)
        return int(np.argmax(q + u))

    @staticmethod
    def _backup(path: list[tuple[Node, int]], leaf_value: float) -> None:
        """``leaf_value`` is from the leaf mover's perspective; the edge into the leaf belongs to the opponent."""
        v = -leaf_value
        for node, k in reversed(path):
            node.N[k] += 1
            node.W[k] += v
            v = -v

    def _noisy_prior(self, root: Node) -> np.ndarray:
        prior = root.prior
        if self.dirichlet_alpha <= 0 or self.dirichlet_eps <= 0 or prior.shape[0] < 2:
            return prior
        noise = self.rng.dirichlet(np.full(prior.shape[0], self.dirichlet_alpha))
        return ((1.0 - self.dirichlet_eps) * prior + self.dirichlet_eps * noise).astype(np.float32)

    # ---- public API ----------------------------------------------------------------------------
    def search(
        self,
        boards: Sequence[chess.Board],
        roots: Sequence[Node | None] | None = None,
        add_noise: bool = True,
        sims: int | None = None,
    ) -> list[SearchResult]:
        """Run ``sims`` lockstep simulations on every board; boards are left exactly as given."""
        boards = list(boards)
        n = len(boards)
        sims = self.sims if sims is None else int(sims)
        trees: list[Node] = [(roots[i] if roots is not None and roots[i] is not None else Node()) for i in range(n)]

        # 1. expand the roots (one batched forward for those the network has not seen yet)
        pending: list[tuple[Node, np.ndarray, np.ndarray]] = []
        for board, root in zip(boards, trees, strict=True):
            if root.expanded or root.terminal is not None:
                continue
            tv, legal = leaf_status(board)
            if tv is not None:
                root.terminal = tv
                root.value = tv
                continue
            pending.append((root, board_features(board), legal))
        if pending:
            self._expand_batch(pending)
        root_prior = [(self._noisy_prior(r) if add_noise else r.prior) if r.expanded else None for r in trees]
        active = [i for i, r in enumerate(trees) if r.terminal is None]

        # 2. simulations
        for _ in range(sims):
            leaves: list[tuple[Node, np.ndarray, np.ndarray]] = []
            leaf_paths: list[list[tuple[Node, int]]] = []
            for i in active:
                board = boards[i]
                node = trees[i]
                path: list[tuple[Node, int]] = []
                depth = 0
                while node.expanded and node.terminal is None:
                    k = self._select(node, root_prior[i] if depth == 0 else node.prior)
                    board.push(index_to_move(int(node.moves[k]), board))
                    depth += 1
                    child = node.children[k]
                    if child is None:
                        child = Node()
                        node.children[k] = child
                    path.append((node, k))
                    node = child
                if node.terminal is None and not node.expanded:
                    tv, legal = leaf_status(board)
                    if tv is not None:
                        node.terminal = tv
                        node.value = tv
                    else:
                        leaves.append((node, board_features(board), legal))
                        leaf_paths.append(path)
                if node.terminal is not None:
                    self._backup(path, node.terminal)
                for _ in range(depth):
                    board.pop()
            if leaves:
                values = self._expand_batch(leaves)
                for path, v in zip(leaf_paths, values, strict=True):
                    self._backup(path, v)

        # 3. results
        results = []
        for root in trees:
            visits = root.visit_counts()
            if root.terminal is not None:
                results.append(SearchResult(visits, root.terminal, root.terminal, root, 0))
                continue
            total = int(root.N.sum())
            q = float(root.W.sum() / total) if total > 0 else root.value
            results.append(SearchResult(visits, q, root.value, root, sims if total > 0 else 0))
        return results

    def search_one(self, board: chess.Board, root: Node | None = None, add_noise: bool = False,
                   sims: int | None = None) -> SearchResult:
        """Single-board convenience (no root noise by default: for play, not for self-play)."""
        return self.search([board], roots=[root], add_noise=add_noise, sims=sims)[0]

    def temperature_at(self, ply: int) -> float:
        t = self.temperature
        return float(t(ply)) if callable(t) else float(t)

    def choose(self, board: chess.Board, root: Node | None = None, add_noise: bool = False,
               ply: int | None = None) -> tuple[chess.Move | None, SearchResult]:
        """Search then pick a move with the configured temperature; ``None`` if the game is over."""
        res = self.search_one(board, root=root, add_noise=add_noise)
        if res.terminal or res.visits.sum() == 0:
            return None, res
        ply = len(board.move_stack) if ply is None else ply
        temperature = self.temperature_at(ply)
        idx = res.best_index() if temperature <= 0 else select_move_index(res.visits, temperature, self.rng)
        return index_to_move(idx, board), res


__all__ = [
    "BatchedMCTS",
    "Node",
    "SearchResult",
    "advance_root",
    "leaf_status",
    "result_string",
    "select_move_index",
    "terminal_value",
]
