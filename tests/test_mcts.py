"""Tests for flychess.train.mcts (batched PUCT search) and flychess.train.selfplay (games, buffer, stage)."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import chess
import numpy as np
import pytest
import torch

from flychess.chessenv.encoding import NUM_MOVES, index_to_move, legal_move_mask, move_to_index
from flychess.connectome.graph import BrainGraph, toy_graph
from flychess.model import BrainConfig, FlyBrain
from flychess.train.mcts import (
    BatchedMCTS,
    advance_root,
    leaf_status,
    result_string,
    select_move_index,
    terminal_value,
)
from flychess.train.metrics import MetricsLogger, read_metrics
from flychess.train.selfplay import (
    Game,
    ReplayBuffer,
    _evaluate_vs_previous,
    completed_iterations,
    elo_from_score,
    generate_games,
    opening_fens,
    paired_start_fens,
    play_match,
    run_selfplay_stage,
    selfplay_loss,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TINY = Path(__file__).resolve().parent.parent / "data" / "brain" / "tiny.npz"
MATE_IN_ONE = "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1"  # Ra8# is the only mate
# Simulation budget for the "the search finds an obvious mate" assertions. A randomly initialised
# model can give the mating move an arbitrarily low prior, so the budget has to be well above the
# point where PUCT first visits it (see test_batched_search_is_per_board); a smaller number makes
# the test assert a property of torch's initialisation rather than of the search.
MATE_SIMS = 64


def _config(**over):
    """Minimal self-play config (the fields of the shared TrainConfig contract that this stage reads)."""
    base = dict(  # noqa: C408
        seed=0, mcts_sims=4, c_puct=1.5, dirichlet_alpha=0.3, dirichlet_eps=0.25, temperature_plies=30,
        max_game_plies=20, replay_buffer_size=2000, selfplay_batch_size=16, selfplay_train_steps_per_iter=3,
        selfplay_lr=2e-4, selfplay_iters=1, selfplay_games_per_iter=2, selfplay_workers=1, value_loss_weight=1.0,
        grad_clip=1.0, amp=False, log_every=1, elo_every=3, elo_games=2, weight_decay=1e-4,
        selfplay_eval_every_iters=1,
        selfplay_human_mix=0.0, selfplay_gate=False,  # v1 semantics (no shards here); v2 is tested in test_selfplay_v2.py
    )
    base.update(over)
    try:  # use the real TrainConfig when the trainer module exists
        from flychess.train.config import TrainConfig

        cfg = TrainConfig.tiny() if hasattr(TrainConfig, "tiny") else TrainConfig(run="t")
        for k, v in base.items():  # plain attributes: also the fields TrainConfig does not define yet
            setattr(cfg, k, v)
        return cfg
    except Exception:  # noqa: BLE001 - not written yet / different shape → duck-typed namespace
        return SimpleNamespace(run="t", **base)


@pytest.fixture(scope="module")
def graph():
    if TINY.exists():
        return BrainGraph.load(TINY)
    return toy_graph(n=300, nnz=3000, n_in=32, n_out=32, seed=3)


@pytest.fixture(scope="module")
def model(graph):
    torch.manual_seed(0)
    cfg = BrainConfig(graph_path=str(TINY), steps=4)
    return FlyBrain(graph, cfg).to(DEVICE).eval()


# ---- helpers --------------------------------------------------------------------------------------
def test_terminal_value_and_leaf_status():
    b = chess.Board()
    assert terminal_value(b) is None
    tv, legal = leaf_status(b)
    assert tv is None and legal is not None and legal.shape[0] == 20
    assert set(legal.tolist()) == set(np.nonzero(legal_move_mask(b))[0].tolist())
    mate = chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
    assert mate.is_checkmate() and terminal_value(mate) == -1.0 and result_string(mate) == "1-0"
    assert leaf_status(mate) == (-1.0, None)
    stale = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    assert stale.is_stalemate() and terminal_value(stale) == 0.0 and result_string(stale) == "1/2-1/2"
    dead = chess.Board("7k/8/8/8/8/8/8/K7 w - - 0 1")
    assert terminal_value(dead) == 0.0
    fifty = chess.Board("7k/8/8/8/8/8/8/R6K w - - 100 80")
    assert terminal_value(fifty) == 0.0


def test_select_move_index():
    rng = np.random.default_rng(0)
    v = np.zeros(NUM_MOVES, np.float32)
    v[[10, 20, 30]] = [5, 1, 2]
    assert select_move_index(v, 0.0, rng) == 10
    picks = {select_move_index(v, 1.0, rng) for _ in range(200)}
    assert picks <= {10, 20, 30} and 10 in picks


# ---- search ---------------------------------------------------------------------------------------
def test_search_returns_legal_moves_and_counts(model):
    mcts = BatchedMCTS(model, DEVICE, sims=16, rng=np.random.default_rng(0))
    board = chess.Board()
    res = mcts.search_one(board, add_noise=True)
    assert res.visits.shape == (NUM_MOVES,) and res.visits.sum() == 16 and res.sims == 16
    assert board.move_stack == []  # the board is left untouched
    legal = legal_move_mask(board)
    assert not res.visits[~legal].any()
    assert -1.0 <= res.root_value <= 1.0 and -1.0 <= res.prior_value <= 1.0
    move, _ = mcts.choose(board)
    assert move in board.legal_moves
    assert mcts.forwards >= 2 and mcts.evaluations >= 17


def test_mate_in_one_from_terminal_values(model):
    """Values at mates come from the tree, not the (untrained) network: the mating move must dominate."""
    mcts = BatchedMCTS(model, DEVICE, sims=120, dirichlet_alpha=0.0, rng=np.random.default_rng(1))
    board = chess.Board(MATE_IN_ONE)
    res = mcts.search_one(board)
    pol = res.policy()
    best = index_to_move(res.best_index(), board)
    assert best == chess.Move.from_uci("a1a8"), res.top(board)
    assert pol[move_to_index(best, board)] > 0.5
    assert res.root_value > 0.5  # the search knows it is winning


def test_batched_search_is_per_board(model):
    # The budget must be large enough that PUCT reaches the mating move whatever priors the
    # (randomly initialised) fixture model happens to give it: Ra8# is a quiet rook move, and with
    # 12 simulations the search never even visits it, so the test would be asserting a property of
    # torch's initialisation rather than of the search. Measured on this fixture: the mating move
    # is first visited at ~24 simulations and dominates from ~50 on.
    mcts = BatchedMCTS(model, DEVICE, sims=MATE_SIMS, rng=np.random.default_rng(0))
    boards = [chess.Board(), chess.Board(MATE_IN_ONE), chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1"),
              chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")]
    fens = [b.fen() for b in boards]
    results = mcts.search(boards, add_noise=False)
    assert len(results) == 4
    for b, f, r in zip(boards, fens, results, strict=True):
        assert b.fen() == f and not b.move_stack  # the boards are left untouched
        if r.terminal:
            assert r.visits.sum() == 0 and r.root_value == -1.0
        else:
            assert r.visits.sum() == MATE_SIMS
            assert not r.visits[~legal_move_mask(b)].any()
    assert results[2].terminal and not results[0].terminal
    assert index_to_move(results[1].best_index(), boards[1]).uci() == "a1a8"
    # batched == what we get searching alone (search is deterministic without noise)
    single = mcts.search_one(chess.Board(), add_noise=False)
    assert np.array_equal(single.visits, results[0].visits)


def test_subtree_reuse(model):
    mcts = BatchedMCTS(model, DEVICE, sims=10, dirichlet_alpha=0.0, rng=np.random.default_rng(0))
    board = chess.Board()
    res = mcts.search_one(board)
    move = index_to_move(int(np.argmax(res.visits)), board)
    child = advance_root(res.root, move, board)
    assert child is not None and child.expanded
    board.push(move)
    before = child.visits
    res2 = mcts.search_one(board, root=child)
    assert res2.root is child and res2.visits.sum() == before + 10
    assert advance_root(None, move, board) is None


# ---- self-play games ------------------------------------------------------------------------------
def test_generate_games_shapes(model, graph):
    cfg = _config(mcts_sims=4, max_game_plies=20)
    seen = []
    games = generate_games(model, graph, cfg, 2, DEVICE, np.random.default_rng(0), on_game=seen.append)
    assert len(games) == 2 and len(seen) == 2
    for g in games:
        assert isinstance(g, Game) and 1 <= g.plies <= 20
        assert g.planes.shape == (g.plies, 20, 8, 8) and g.planes.dtype == np.uint8
        assert g.pi.shape == (g.plies, NUM_MOVES) and g.pi.dtype == np.float16
        assert np.allclose(g.pi.astype(np.float32).sum(1), 1.0, atol=1e-2)
        assert g.z.shape == (g.plies,) and g.z.dtype == np.int8 and set(g.z.tolist()) <= {-1, 0, 1}
        assert g.result in ("1-0", "0-1", "1/2-1/2") and len(g.moves) == g.plies
        assert "[Result" in g.pgn and len(g.legal) == g.plies
        if g.result == "1/2-1/2":
            assert not g.z.any()
        else:  # alternating signs from each mover's perspective
            assert abs(int(g.z[0])) == 1 and all(g.z[i] == -g.z[i + 1] for i in range(g.plies - 1))
        # the played move always had visits, and pi is supported on legal moves only
        b = chess.Board()
        for t, uci in enumerate(g.moves):
            mv = chess.Move.from_uci(uci)
            assert mv in b.legal_moves
            assert g.pi[t, move_to_index(mv, b)] > 0
            assert set(np.nonzero(g.pi[t])[0].tolist()) <= set(g.legal[t].astype(int).tolist())
            b.push(mv)
        assert g.capped == (g.plies == 20 and terminal_value(b) is None)


def test_generate_games_threaded(model, graph):
    cfg = _config(mcts_sims=2, max_game_plies=6)
    games = generate_games(model, graph, cfg, 3, DEVICE, np.random.default_rng(0), workers=2)
    assert len(games) == 3 and all(g.plies >= 1 for g in games)


def test_replay_buffer(model, graph):
    cfg = _config(mcts_sims=4, max_game_plies=12)
    games = generate_games(model, graph, cfg, 2, DEVICE, np.random.default_rng(0))
    buf = ReplayBuffer(size=15, k=8, lmax=16)
    total = sum(buf.add(g) for g in games)
    assert total == sum(g.plies for g in games) and len(buf) == min(15, total) and buf.games == 2
    batch = buf.sample(6)
    assert batch["planes"].shape == (6, 1280) and batch["planes"].dtype == torch.float32
    assert batch["pi"].shape == (6, NUM_MOVES) and torch.allclose(batch["pi"].sum(1), torch.ones(6), atol=1e-3)
    assert batch["z"].shape == (6,) and batch["legal_mask"].shape == (6, NUM_MOVES)
    assert bool((batch["legal_mask"] | (batch["pi"] == 0)).all())  # pi lives inside the mask
    assert bool(batch["legal_mask"].any(1).all())
    # planes 0-17 are 0/1 and plane 17 is all ones
    p = batch["planes"].view(6, 20, 8, 8)
    assert bool((p[:, 17] == 1).all()) and bool(((p[:, :17] == 0) | (p[:, :17] == 1)).all())
    with pytest.raises(ValueError):
        ReplayBuffer(4).sample(1)


def test_selfplay_loss(model):
    B = 3
    logits = torch.randn(B, NUM_MOVES, device=DEVICE, requires_grad=True)
    value = torch.tanh(torch.randn(B, 1, device=DEVICE, requires_grad=True))
    mask = torch.zeros(B, NUM_MOVES, dtype=torch.bool, device=DEVICE)
    mask[:, :5] = True
    pi = torch.zeros(B, NUM_MOVES, device=DEVICE)
    pi[:, 0] = 0.5
    pi[:, 1] = 0.5
    z = torch.tensor([1.0, 0.0, -1.0], device=DEVICE)
    loss, met = selfplay_loss(logits, value, pi, z, mask, value_weight=2.0)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all() and not logits.grad[:, 5:].any()  # masked moves get no gradient
    assert set(met) == {"loss", "policy_loss", "value_loss", "top1", "top3"}
    assert abs(float(met["loss"]) - (float(met["policy_loss"]) + 2.0 * float(met["value_loss"]))) < 1e-5


def test_elo_helpers_and_match(model, graph):
    assert elo_from_score(0.5) == 0.0 and elo_from_score(0.75) > 150 and elo_from_score(0.25) < -150
    from flychess.play.engine import FlyEngine

    a = FlyEngine(model, graph, DEVICE, sims=2, seed=0).as_player("larva")
    b = FlyEngine(model, graph, DEVICE, sims=2, seed=1).as_player("larva")
    out = play_match(a, b, games=2, max_plies=6)
    assert out["games"] == 2 and out["wins"] + out["draws"] + out["losses"] == 2 and "pgn" in out


def test_run_selfplay_stage_smoke(model, graph, tmp_path):
    cfg = _config(selfplay_iters=1, selfplay_games_per_iter=2, mcts_sims=4, selfplay_train_steps_per_iter=3,
                  max_game_plies=10, elo_every=3, elo_games=2, selfplay_batch_size=8)
    saved = []

    def save_checkpoint(step: int, stage: str) -> Path:
        p = tmp_path / f"ckpt-{step}.pt"
        torch.save({"model": model.state_dict(), "step": step, "stage": stage}, p)
        saved.append((step, stage))
        return p

    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    with MetricsLogger(tmp_path / "run", rate_limits={}) as logger:
        step = run_selfplay_stage(model, graph, cfg, logger, start_step=100, save_checkpoint=save_checkpoint,
                                  device=DEVICE)
    assert step == 103
    assert saved == [(103, "selfplay")] and (tmp_path / "ckpt-103.pt").exists()
    assert not model.training  # left in eval mode
    assert any(not torch.equal(before[k], v) for k, v in model.state_dict().items() if v.dtype.is_floating_point)
    recs = read_metrics(tmp_path / "run")
    kinds = {r["kind"] for r in recs}
    assert {"train", "game", "selfplay", "status", "elo"} <= kinds
    train = [r for r in recs if r["kind"] == "train"]
    assert train[-1]["step"] == 103 and train[-1]["stage"] == "selfplay"
    assert all(k in train[-1] for k in ("loss", "policy_loss", "value_loss", "top1", "top3", "lr", "pos_per_sec"))
    game = next(r for r in recs if r["kind"] == "game")
    assert game["source"] == "selfplay" and "1." in game["pgn"]
    elos = {r["opponent"]: r for r in recs if r["kind"] == "elo"}
    assert set(elos) == {"previous-iteration", "random"}
    for r in elos.values():
        assert r["games"] == 2 and r["wins"] + r["draws"] + r["losses"] == 2 and r["openings"] >= 1 and r["iteration"] == 1
    status = [r for r in recs if r["kind"] == "status" and r.get("iteration")]
    assert status[-1]["iteration"] == 1 and completed_iterations(tmp_path / "run", 103) == 1
    assert completed_iterations(tmp_path / "run", 102) == 0


# ---- regression tests for the review findings -----------------------------------------------------
def test_evaluate_uses_distinct_openings(model, graph):
    """Two deterministic ``fly`` players from the start position would replay one game per colour; the
    evaluation must vary the openings so that every game of the match is a different game."""
    cfg = _config(mcts_sims=2, max_game_plies=16)
    games = generate_games(model, graph, cfg, 3, DEVICE, np.random.default_rng(3))
    rng = np.random.default_rng(0)
    openings = opening_fens(games, 2, rng, min_ply=2, max_ply=4)
    assert 1 <= len(openings) <= 2 and len(set(openings)) == len(openings)
    for fen in openings:
        assert terminal_value(chess.Board(fen)) is None
    fens = paired_start_fens(openings, 4)
    assert len(fens) == 4 and fens[0] == fens[1] and fens[2] == fens[3]
    assert paired_start_fens([], 4) is None and paired_start_fens(openings, 0) is None
    same_weights = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
    out = _evaluate_vs_previous(model, same_weights, graph, cfg, torch.device(DEVICE), 4, fens, seed=1)
    assert out is not None and out["games"] == 4 and out["openings"] == len(set(fens)) and len(out["pgns"]) == 4
    # identical brains: a paired opening gives mirrored games, and different openings give different games
    assert all(f'[FEN "{f}"]' in pgn for f, pgn in zip(fens, out["pgns"], strict=True))
    without = _evaluate_vs_previous(model, same_weights, graph, cfg, torch.device(DEVICE), 4, None, seed=1)
    assert without is not None and without["openings"] == 1
    moves = [p.split("\n\n", 1)[1] for p in without["pgns"]]
    assert moves[0] == moves[2] and moves[1] == moves[3]  # the old behaviour: 4 games, 2 distinct


def test_fallback_play_match_supports_start_fens(model, graph):
    from flychess.play.engine import FlyEngine

    a = FlyEngine(model, graph, DEVICE, sims=2, seed=0).as_player("fly")
    b = FlyEngine(model, graph, DEVICE, sims=2, seed=1).as_player("fly")
    fen = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2").fen()
    out = play_match(a, b, games=2, max_plies=4, start_fens=[fen], keep_pgns=2)
    assert out["games"] == 2 and len(out["pgns"]) == 2 and all(f'[FEN "{fen}"]' in p for p in out["pgns"])


def test_non_finite_network_output_is_an_error(model):
    """A diverged network must not be silently replaced by uniform priors."""
    mcts = BatchedMCTS(model, DEVICE, sims=2, rng=np.random.default_rng(0))
    head = model.policy_head.bias
    saved = head.detach().clone()
    try:
        with torch.no_grad():
            head.fill_(float("nan"))
        with pytest.raises(RuntimeError, match="non-finite"):
            mcts.search_one(chess.Board())
    finally:
        with torch.no_grad():
            head.copy_(saved)
    res = mcts.search_one(chess.Board())  # healthy again
    assert res.visits.sum() == 2


def test_non_finite_loss_stops_the_stage(model, graph, tmp_path, monkeypatch):
    import flychess.train.selfplay as sp

    cfg = _config(selfplay_iters=2, selfplay_games_per_iter=1, mcts_sims=2, selfplay_train_steps_per_iter=2,
                  max_game_plies=6, elo_games=0)
    saved = []

    real = sp.selfplay_loss

    def bad_loss(logits, value, pi, z, legal_mask=None, value_weight=1.0):
        loss, met = real(logits, value, pi, z, legal_mask, value_weight)
        return loss * float("nan"), met

    monkeypatch.setattr(sp, "selfplay_loss", bad_loss)
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    with MetricsLogger(tmp_path / "run", rate_limits={}) as logger, pytest.raises(RuntimeError, match="non-finite"):
        run_selfplay_stage(model, graph, cfg, logger, 0, lambda step, stage: saved.append((step, stage)) or tmp_path,
                           device=DEVICE)
    assert saved == [(0, "selfplay")]  # a checkpoint of the untouched weights was written
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert not model.training


def test_eval_interval_is_iterations_not_steps(model, graph, tmp_path):
    """``selfplay_eval_every_iters`` (default 5) sets the evaluation cadence; the last iteration always evaluates."""
    cfg = _config(selfplay_iters=3, selfplay_games_per_iter=1, mcts_sims=2, selfplay_train_steps_per_iter=1,
                  max_game_plies=6, elo_games=2, elo_every=1, selfplay_eval_every_iters=2)
    with MetricsLogger(tmp_path / "run", rate_limits={}) as logger:
        run_selfplay_stage(model, graph, cfg, logger, 0, lambda step, stage: tmp_path, device=DEVICE)
    elos = read_metrics(tmp_path / "run", kinds=["elo"])
    assert sorted({r["iteration"] for r in elos}) == [2, 3]
    assert {r["opponent"] for r in elos} == {"previous-iteration", "random"}


def test_resume_skips_finished_iterations_and_restores_optimizer(model, graph, tmp_path):
    cfg = _config(selfplay_iters=2, selfplay_games_per_iter=1, mcts_sims=2, selfplay_train_steps_per_iter=2,
                  max_game_plies=6, elo_games=0, selfplay_lr=1e-3)
    run_dir = tmp_path / "run"
    calls = []

    def save(step: int, stage: str, optim=None) -> Path:  # the trainer's save() accepts optim=
        state = optim.state_dict() if isinstance(optim, torch.optim.Optimizer) else optim
        calls.append((step, stage, state is not None and len(state["state"]) > 0))
        p = run_dir / "latest.pt"
        torch.save({"model": model.state_dict(), "optim": state, "step": step, "stage": stage}, p)
        return p

    with MetricsLogger(run_dir, rate_limits={}) as logger:
        step = run_selfplay_stage(model, graph, cfg, logger, 0, save, device=DEVICE)
    assert step == 4 and [c[:2] for c in calls] == [(2, "selfplay"), (4, "selfplay")] and all(c[2] for c in calls)
    assert completed_iterations(run_dir, 4) == 2
    # resuming a finished stage runs nothing more
    with MetricsLogger(run_dir, rate_limits={}) as logger:
        assert run_selfplay_stage(model, graph, cfg, logger, 4, save, device=DEVICE) == 4
    assert len(calls) == 2
    # one more iteration requested: exactly one more runs, with the optimizer state restored from latest.pt
    cfg.selfplay_iters = 3
    saved_state = torch.load(run_dir / "latest.pt", map_location="cpu", weights_only=False)["optim"]
    seen: dict = {}
    real_load = torch.optim.AdamW.load_state_dict

    def spy(self, state):
        seen["restored"] = True
        return real_load(self, state)

    torch.optim.AdamW.load_state_dict = spy  # type: ignore[method-assign]
    try:
        with MetricsLogger(run_dir, rate_limits={}) as logger:
            step = run_selfplay_stage(model, graph, cfg, logger, 4, save, device=DEVICE)
    finally:
        torch.optim.AdamW.load_state_dict = real_load  # type: ignore[method-assign]
    assert step == 6 and seen.get("restored") and [c[0] for c in calls] == [2, 4, 6]
    assert saved_state["param_groups"][0]["lr"] == 1e-3
    status = [r for r in read_metrics(run_dir, kinds=["status"]) if "resuming" in r["message"]]
    assert status and status[-1]["message"].endswith("after iteration 2/3")
    # explicit completed_iters wins over the metrics
    with MetricsLogger(run_dir, rate_limits={}) as logger:
        assert run_selfplay_stage(model, graph, cfg, logger, 6, save, device=DEVICE, completed_iters=3) == 6
