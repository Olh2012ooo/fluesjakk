"""Evaluation: opponents (§ eval/opponents.py) and match play / Elo estimation (§ eval/elo.py)."""
from flychess.eval.elo import MatchResult, elo_estimate, evaluate_run, play_match, round_robin
from flychess.eval.opponents import (
    BrainPolicyPlayer,
    GreedyMaterialPlayer,
    Player,
    RandomPlayer,
    StockfishPlayer,
    make_opponent,
)

__all__ = [
    "BrainPolicyPlayer",
    "GreedyMaterialPlayer",
    "MatchResult",
    "Player",
    "RandomPlayer",
    "StockfishPlayer",
    "elo_estimate",
    "evaluate_run",
    "make_opponent",
    "play_match",
    "round_robin",
]
