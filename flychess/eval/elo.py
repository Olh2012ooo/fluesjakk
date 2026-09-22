"""Match play and Elo estimation (docs/SPEC.md eval/elo.py).

``play_match`` plays every game of a match in lockstep so that a :class:`BrainPolicyPlayer` can
evaluate all boards where it is to move in a single batched forward pass; players without
``choose_many`` are simply called board by board.  Games truncated at ``max_plies`` count as draws.

Game termination follows the project-wide rule in :func:`flychess.train.mcts.terminal_value` (the one
used by self-play, ``fly play`` and the web engine): checkmate, stalemate, insufficient material,
50-move rule and threefold repetition of the *current* position. ``Board.is_game_over(claim_draw=True)``
is deliberately not used — python-chess evaluates claimable draws *before* the side to move plays, so
a player with a mate in one would be awarded a draw instead of delivering it.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chess
import chess.pgn

from flychess.eval.opponents import BrainPolicyPlayer, Player, make_opponent
from flychess.train.mcts import result_string, terminal_value

RESULT_SCORE = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}


@dataclass
class MatchResult:
    """Outcome of a match from ``player_a``'s point of view."""

    wins: int = 0
    draws: int = 0
    losses: int = 0
    pgns: list[str] = field(default_factory=list)      # PGN of games 0 .. keep_pgns-1 (game order)
    results: list[str] = field(default_factory=list)   # '1-0' / '0-1' / '1/2-1/2' per game (game order, white's view)
    plies: list[int] = field(default_factory=list)     # plies per game (game order)
    truncated: int = 0                                 # games stopped at max_plies (counted as draws)
    player_a: str = ""
    player_b: str = ""

    @property
    def games(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def score(self) -> float:
        """Points per game for player_a in [0, 1]."""
        return (self.wins + 0.5 * self.draws) / max(self.games, 1)

    @property
    def elo(self) -> float:
        return elo_estimate(self.wins, self.draws, self.losses)

    def to_dict(self) -> dict[str, Any]:
        return {
            "player_a": self.player_a, "player_b": self.player_b, "games": self.games, "wins": self.wins,
            "draws": self.draws, "losses": self.losses, "score": self.score, "elo_estimate": self.elo,
            "truncated": self.truncated, "mean_plies": (sum(self.plies) / len(self.plies)) if self.plies else 0.0,
        }

    def __str__(self) -> str:
        return (f"{self.player_a} vs {self.player_b}: +{self.wins} ={self.draws} -{self.losses} "
                f"(score {self.score:.3f}, Elo {self.elo:+.0f})")


def elo_estimate(wins: int, draws: int, losses: int, opponent_elo: float | None = None) -> float:
    """Elo difference implied by a score (logistic model), clamped so 100 % / 0 % stay finite.

    ``p`` is clamped to ``[1/(2n+2), 1 - 1/(2n+2)]`` — a Laplace-style bound, so a perfect 20-game
    match reads ~+660 rather than +inf. With ``opponent_elo`` the absolute rating is returned instead.
    """
    n = wins + draws + losses
    if n <= 0:
        return 0.0 if opponent_elo is None else float(opponent_elo)
    p = (wins + 0.5 * draws) / n
    eps = 1.0 / (2.0 * n + 2.0)
    p = min(max(p, eps), 1.0 - eps)
    diff = 400.0 * math.log10(p / (1.0 - p))
    return diff if opponent_elo is None else float(opponent_elo) + diff


def board_pgn(board: chess.Board, white: str, black: str, result: str, headers: dict[str, str] | None = None) -> str:
    game = chess.pgn.Game.from_board(board)
    game.headers["Event"] = "fly-chess evaluation"
    game.headers["White"], game.headers["Black"], game.headers["Result"] = white, black, result
    for k, v in (headers or {}).items():
        game.headers[k] = v
    return str(game)


def _game_result(board: chess.Board, max_plies: int) -> str | None:
    """Final result string, or None while the game is still running (see module doc for the rule)."""
    if terminal_value(board) is not None:
        return result_string(board)
    if len(board.move_stack) >= max_plies:
        return "1/2-1/2"
    return None


def _choose_all(player: Player, boards: list[chess.Board]) -> list[chess.Move]:
    choose_many = getattr(player, "choose_many", None)
    if choose_many is not None:
        return list(choose_many(boards))
    return [player.choose(b) for b in boards]


def play_match(
    player_a: Player,
    player_b: Player,
    games: int,
    max_plies: int = 200,
    seed: int = 0,
    on_game: Callable[[int, str, str], None] | None = None,
    keep_pgns: int = 3,
    start_fens: Sequence[str] | None = None,
) -> MatchResult:
    """``games`` games between ``player_a`` and ``player_b`` with alternating colours (a is white in game 0).

    All games run in lockstep (see module doc). ``on_game(index, result, pgn)`` is called as each game
    ends. ``seed`` reseeds players that expose ``rng``/``generator`` so that matches are reproducible.
    """
    for k, player in enumerate((player_a, player_b)):
        rng = getattr(player, "rng", None)
        if rng is not None:
            import numpy as np

            player.rng = np.random.default_rng(seed * 2 + k)
        gen = getattr(player, "generator", None)
        if gen is not None:
            gen.manual_seed(seed * 2 + k)

    result = MatchResult(player_a=player_a.name, player_b=player_b.name, results=[""] * games,
                         plies=[0] * games, pgns=[""] * min(max(keep_pgns, 0), games))
    boards = [chess.Board(start_fens[i % len(start_fens)]) if start_fens else chess.Board() for i in range(games)]
    a_is_white = [i % 2 == 0 for i in range(games)]
    done: list[str | None] = [None] * games
    n_done = 0

    def finish(i: int, res: str) -> None:
        nonlocal n_done
        done[i] = res
        n_done += 1
        score_white = RESULT_SCORE[res]
        score_a = score_white if a_is_white[i] else 1.0 - score_white
        if score_a == 1.0:
            result.wins += 1
        elif score_a == 0.0:
            result.losses += 1
        else:
            result.draws += 1
        if len(boards[i].move_stack) >= max_plies and terminal_value(boards[i]) is None:
            result.truncated += 1
        result.results[i] = res
        result.plies[i] = len(boards[i].move_stack)
        pgn = ""
        if i < len(result.pgns) or on_game is not None:
            white, black = (player_a.name, player_b.name) if a_is_white[i] else (player_b.name, player_a.name)
            pgn = board_pgn(boards[i], white, black, res, {"Round": str(i + 1)})
            if i < len(result.pgns):
                result.pgns[i] = pgn
        if on_game is not None:
            on_game(i, res, pgn)

    for i in range(games):  # games that are over before a single move (custom start FENs)
        res = _game_result(boards[i], max_plies)
        if res is not None:
            finish(i, res)

    while n_done < games:
        for player, plays_white in ((player_a, True), (player_b, False)):
            idx = [i for i in range(games) if done[i] is None
                   and (boards[i].turn == chess.WHITE) == (a_is_white[i] == plays_white)]
            if not idx:
                continue
            moves = _choose_all(player, [boards[i] for i in idx])
            for i, move in zip(idx, moves, strict=True):
                board = boards[i]
                if move not in board.legal_moves:
                    raise ValueError(f"{player.name} played illegal move {move.uci()} in {board.fen()}")
                board.push(move)
                res = _game_result(board, max_plies)
                if res is not None:
                    finish(i, res)
    return result


def round_robin(players: Sequence[Player], games: int, max_plies: int = 200, seed: int = 0,
                keep_pgns: int = 1) -> list[dict[str, Any]]:
    """Every pair plays ``games`` games; returns one row per player sorted by score (desc).

    Row: ``{name, games, wins, draws, losses, score, elo_vs: {opponent: elo_diff}, matches: [...]}``.
    """
    rows = {p.name: {"name": p.name, "games": 0, "wins": 0, "draws": 0, "losses": 0, "elo_vs": {}, "matches": []}
            for p in players}
    for ia in range(len(players)):
        for ib in range(ia + 1, len(players)):
            a, b = players[ia], players[ib]
            m = play_match(a, b, games, max_plies=max_plies, seed=seed + ia * 1000 + ib, keep_pgns=keep_pgns)
            for name, w, d, l_, opp in ((a.name, m.wins, m.draws, m.losses, b.name),
                                        (b.name, m.losses, m.draws, m.wins, a.name)):
                row = rows[name]
                row["games"] += m.games
                row["wins"] += w
                row["draws"] += d
                row["losses"] += l_
                row["elo_vs"][opp] = elo_estimate(w, d, l_)
                row["matches"].append({**m.to_dict(), "player_a": name, "player_b": opp, "wins": w,
                                       "draws": d, "losses": l_, "elo_estimate": elo_estimate(w, d, l_),
                                       "score": (w + 0.5 * d) / max(m.games, 1)})
    table = list(rows.values())
    for row in table:
        row["score"] = (row["wins"] + 0.5 * row["draws"]) / max(row["games"], 1)
    table.sort(key=lambda r: (r["score"], r["wins"]), reverse=True)
    return table


def _resolve_player(spec: str, run: str | None, device: str | None, temperature: float, seed: int) -> Player:
    """'random' / 'material' / 'stockfish' / a run name / a checkpoint path / 'ckpt-N' inside ``run``."""
    from flychess import paths
    from flychess.train.trainer import load_checkpoint

    if spec in ("random", "material") or spec.startswith("stockfish"):
        return make_opponent(spec, seed=seed)
    candidates = [Path(spec)]
    if run is not None:
        candidates.append(paths.run_dir(run) / spec)
        candidates.append(paths.run_dir(run) / f"{spec}.pt")
    for c in candidates:
        if c.exists():
            model, _, ckpt = load_checkpoint(c, device=device or "cpu")
            return BrainPolicyPlayer(model, device=device, temperature=temperature,
                                     name=f"{c.stem}@{ckpt.get('step', 0)}", seed=seed)
    model, _, ckpt = load_checkpoint(spec, device=device or "cpu")   # another run name (raises if unknown)
    return BrainPolicyPlayer(model, device=device, temperature=temperature,
                             name=f"{spec}@{ckpt.get('step', 0)}", seed=seed)


def evaluate_run(
    run: str,
    games: int = 20,
    opponents: Sequence[str] = ("random", "material"),
    device: str | None = None,
    max_plies: int = 200,
    temperature: float = 0.0,
    seed: int = 0,
    log: bool = True,
    verbose: bool = True,
) -> dict[str, dict[str, Any]]:
    """Play the run's latest brain against ``opponents`` and return ``{opponent: MatchResult dict + pgn}``.

    Opponents: ``'random'``, ``'material'``, ``'stockfish[:depth[:skill]]'``, another run name, a
    checkpoint path, or ``'ckpt-<step>'`` — an earlier checkpoint of the *same* run (previous-checkpoint
    tournament). With ``log=True`` an ``elo`` record per opponent is appended to the run's metrics.
    """
    import torch

    from flychess import paths
    from flychess.train.metrics import MetricsLogger
    from flychess.train.trainer import load_checkpoint

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, _, ckpt = load_checkpoint(run, device=device)
    step = int(ckpt.get("step", 0))
    fly = BrainPolicyPlayer(model, device=device, temperature=temperature, name=f"{run}@{step}", seed=seed)
    out: dict[str, dict[str, Any]] = {}
    logger = MetricsLogger(paths.run_dir(run)) if log else None
    try:
        for k, spec in enumerate(opponents):
            opp = _resolve_player(spec, run, device, temperature, seed + 100 + k)
            m = play_match(fly, opp, games, max_plies=max_plies, seed=seed + k)
            out[spec] = {**m.to_dict(), "pgn": m.pgns[0] if m.pgns else ""}
            if verbose:
                print(m)
            if logger is not None:
                logger.log_elo(step, opponent=spec, games=m.games, wins=m.wins, draws=m.draws, losses=m.losses,
                               elo_estimate=m.elo, source="evaluate_run")
            close = getattr(opp, "close", None)
            if close is not None:
                close()
    finally:
        if logger is not None:
            logger.close()
    return out


__all__ = [
    "MatchResult",
    "board_pgn",
    "elo_estimate",
    "evaluate_run",
    "play_match",
    "round_robin",
]
