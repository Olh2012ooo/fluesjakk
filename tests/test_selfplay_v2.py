"""Tests for self-play v2 (flychess.train.selfplay): human-mixed batches, value blend, gating, best.pt, resume.

Tiny graph + the fixture PGN built into shards in a temp dir; small enough for the CPU (or a busy GPU).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from flychess import paths
from flychess.chessenv.encoding import NUM_MOVES
from flychess.connectome.graph import BrainGraph, toy_graph
from flychess.data.lichess import build_shards
from flychess.data.shards import list_shards
from flychess.model import BrainConfig, FlyBrain
from flychess.train import selfplay as sp
from flychess.train.config import TrainConfig
from flychess.train.metrics import MetricsLogger, read_metrics
from flychess.train.selfplay import (
    GATE_OPPONENT,
    Game,
    HumanStream,
    ReplayBuffer,
    completed_iterations,
    fixed_openings,
    gate_candidate,
    generate_games,
    human_targets,
    make_mixed_batch,
    match_openings,
    mixed_loss,
    run_selfplay_stage,
)
from flychess.train.trainer import save_checkpoint

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TINY = paths.BRAIN_DIR / "tiny.npz"
FIXTURE = Path(__file__).parent / "fixtures" / "small.pgn"


@pytest.fixture(scope="module")
def shards(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("shards")
    stats = build_shards(FIXTURE, out, name="fix", workers=2, shard_size=500, shuffle_buffer=700, progress=False)
    assert stats.positions > 1000 and len(list_shards(out, "fix")) >= 3
    return out


@pytest.fixture(scope="module")
def graph():
    if TINY.exists():
        return BrainGraph.load(TINY)
    return toy_graph(n=300, nnz=3000, n_in=32, n_out=32, seed=3)


@pytest.fixture
def model(graph):
    torch.manual_seed(0)
    return FlyBrain(graph, BrainConfig(graph_path=str(TINY), steps=4, value_hidden=32)).to(DEVICE).eval()


def v2_cfg(shards: Path, **over) -> TrainConfig:
    base = dict(  # noqa: C408
        shards_dir=str(shards), shard_name="fix", graph=str(TINY), device=DEVICE, num_workers=0, log_every=1,
        selfplay_iters=1, selfplay_games_per_iter=8, mcts_sims=8, selfplay_train_steps_per_iter=6,
        selfplay_batch_size=16, selfplay_human_mix=0.5, selfplay_value_blend=0.5, selfplay_gate=True,
        selfplay_gate_games=4, selfplay_gate_sims=4, selfplay_gate_threshold=0.55, max_game_plies=24,
        elo_games=2, selfplay_eval_every_iters=1, replay_buffer_size=4000,
    )
    base.update(over)
    return TrainConfig.tiny(**base)


def _state(model: FlyBrain) -> dict[str, torch.Tensor]:
    return {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}


def _same(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> bool:
    return all(torch.equal(a[k].cpu(), b[k].cpu()) for k in a)


def _saver(run_dir: Path, model: FlyBrain, cfg: TrainConfig):
    """The trainer's checkpoint writer (ckpt-<step>.pt + latest.pt), bound like ``train()`` binds it."""
    def save(step: int, stage: str, optim=None) -> Path:
        return save_checkpoint(run_dir, model, cfg, TINY, step, stage, optim=optim)
    return save


# ---- config ----------------------------------------------------------------------------------------
def test_v2_config_defaults_and_validation():
    cfg = TrainConfig()
    assert (cfg.selfplay_games_per_iter, cfg.mcts_sims, cfg.selfplay_train_steps_per_iter) == (512, 100, 400)
    assert cfg.selfplay_lr == 1e-4 and cfg.selfplay_batch_size == 384 and cfg.selfplay_lockstep_games == 512
    assert cfg.selfplay_human_mix == 0.5 and cfg.selfplay_value_blend == 0.5
    assert cfg.selfplay_gate and cfg.selfplay_gate_games == 40 and cfg.selfplay_gate_sims == 100
    assert cfg.selfplay_gate_threshold == 0.55
    for bad in ({"selfplay_human_mix": 1.5}, {"selfplay_value_blend": -0.1}, {"selfplay_gate_threshold": 2},
                {"selfplay_gate_sims": 0}, {"selfplay_lockstep_games": 0}):
        with pytest.raises(ValueError):
            TrainConfig(**bad)
    with pytest.raises(TypeError):
        TrainConfig(selfplay_gate="yes")
    assert TrainConfig(selfplay_gate_games=0).selfplay_gate_games == 0  # 0 games = no gate
    for name in ("default", "v3", "tiny"):
        y = TrainConfig.from_yaml(Path(__file__).parent.parent / "configs" / f"{name}.yaml")
        assert 0.0 <= y.selfplay_human_mix <= 1.0 and isinstance(y.selfplay_gate, bool)
    # old checkpoints / yamls without the v2 keys get the defaults
    old = {k: v for k, v in TrainConfig().to_dict().items() if not k.startswith("selfplay_")}
    assert TrainConfig.from_dict(old).selfplay_human_mix == 0.5


# ---- value blend and the human mix ------------------------------------------------------------------
def test_value_blend_math():
    T = 5
    game = Game(planes=np.zeros((T, 20, 8, 8), np.uint8), pi=np.full((T, NUM_MOVES), 0, np.float16),
                z=np.array([1, -1, 1, -1, 0], np.int8), legal=[np.array([0, 1], np.int16)] * T,
                moves=["a"] * T, pgn="", result="1-0", plies=T, q=np.array([0.2, -0.4, 0.6, 0.0, 0.5], np.float32))
    game.pi[:, 0] = 0.75
    game.pi[:, 1] = 0.25
    buf = ReplayBuffer(size=100, k=4, lmax=8)
    assert buf.add(game) == T
    rng = np.random.default_rng(0)
    b = buf.sample(64, rng, value_blend=0.5)
    expected = 0.5 * b["outcome"] + 0.5 * b["q"]
    assert torch.allclose(b["z"], expected, atol=1e-3)
    assert set(b["outcome"].tolist()) <= {-1.0, 0.0, 1.0}
    q_by_outcome = {(float(o), round(float(q), 2)) for o, q in zip(b["outcome"], b["q"], strict=True)}
    assert q_by_outcome <= {(1.0, 0.2), (-1.0, -0.4), (1.0, 0.6), (-1.0, 0.0), (0.0, 0.5)}
    b1 = buf.sample(16, np.random.default_rng(7), value_blend=1.0)
    assert torch.equal(b1["z"], b1["outcome"])                      # blend 1 = the outcome only (v1 target)
    b0 = buf.sample(16, np.random.default_rng(7), value_blend=0.0)
    assert torch.allclose(b0["z"], b0["q"], atol=1e-3)               # blend 0 = the search value only
    # a game recorded without q (older Game objects) falls back to the outcome
    old = Game(planes=game.planes, pi=game.pi, z=game.z, legal=game.legal, moves=game.moves, pgn="", result="1-0", plies=T)
    buf2 = ReplayBuffer(size=100, k=4, lmax=8)
    buf2.add(old)
    b2 = buf2.sample(8, np.random.default_rng(0), value_blend=0.3)
    assert torch.equal(b2["z"], b2["outcome"]) and torch.equal(b2["q"], b2["outcome"])


def test_generated_games_carry_search_values(model, graph, shards):
    cfg = v2_cfg(shards, mcts_sims=4, max_game_plies=10)
    games = generate_games(model, graph, cfg, 2, DEVICE, np.random.default_rng(0))
    for g in games:
        assert g.q is not None and g.q.shape == (g.plies,) and g.q.dtype == np.float32
        assert bool((np.abs(g.q) <= 1.0).all())


def test_human_mix_batch_composition(model, graph, shards):
    cfg = v2_cfg(shards, mcts_sims=4, max_game_plies=10)
    games = generate_games(model, graph, cfg, 2, DEVICE, np.random.default_rng(0))
    buf = ReplayBuffer(size=1000, k=8, lmax=32)
    for g in games:
        buf.add(g)
    human = HumanStream(cfg, batch=8, device=torch.device(DEVICE), seed=3)
    assert len(human.files) >= 2 and all(f.name.startswith("fix-") for f in human.files)
    # the human targets: one-hot on the stored move, unmasked, value = the stored value
    hb = human.next()
    ht = human_targets(hb)
    assert ht["pi"].shape == (8, NUM_MOVES) and torch.equal(ht["pi"].argmax(1), hb["move"])
    assert torch.equal(ht["pi"].sum(1), torch.ones(8)) and bool(ht["legal_mask"].all())
    assert torch.equal(ht["z"], hb["value"]) and set(ht["z"].tolist()) <= {-1.0, 0.0, 1.0}
    assert ht["planes"].shape == (8, 1280) and bool((ht["planes"].view(8, 20, 8, 8)[:, 17] == 1).all())
    # the mixed batch: 8 self-play rows first, then 8 human rows
    batch = make_mixed_batch(buf, human, 16, human_mix=0.5, value_blend=0.5, rng=np.random.default_rng(0))
    assert batch["n_self"] == 8 and batch["n_human"] == 8
    assert batch["planes"].shape == (16, 1280) and batch["pi"].shape == (16, NUM_MOVES) and batch["z"].shape == (16,)
    assert torch.allclose(batch["pi"].sum(1), torch.ones(16), atol=1e-3)
    self_rows, human_rows = batch["legal_mask"][:8], batch["legal_mask"][8:]
    assert bool(human_rows.all()) and not bool(self_rows.all(dim=1).any())  # masked search rows, unmasked human rows
    assert bool(((batch["pi"][8:] == 0) | (batch["pi"][8:] == 1)).all())     # one-hot human policy targets
    assert bool((batch["z"][8:].abs() <= 1).all()) and bool((batch["z"][:8].abs() <= 1).all())
    # other mixes
    b0 = make_mixed_batch(buf, human, 16, human_mix=0.0, rng=np.random.default_rng(0))
    assert b0["n_self"] == 16 and b0["n_human"] == 0 and not bool(b0["legal_mask"].all())
    human8 = HumanStream(cfg, batch=16, device=torch.device(DEVICE), seed=4)
    b1 = make_mixed_batch(buf, human8, 16, human_mix=1.0, rng=np.random.default_rng(0))
    assert b1["n_self"] == 0 and b1["n_human"] == 16 and bool(b1["legal_mask"].all())
    with pytest.raises(ValueError, match="human stream"):
        make_mixed_batch(buf, human8, 16, human_mix=0.5, rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match="empty"):
        make_mixed_batch(ReplayBuffer(10), human, 16, human_mix=0.5)
    # the mixed loss reports both halves
    x = batch["planes"].to(DEVICE)
    with torch.no_grad():
        logits, value = model(x)[:2]
    targets = {k: v.to(DEVICE) for k, v in batch.items() if isinstance(v, torch.Tensor)}
    targets.update(n_self=8, n_human=8)
    loss, met = mixed_loss(logits, value, targets, value_weight=1.0)
    assert torch.isfinite(loss)
    for k in ("policy_loss_self", "policy_loss_human", "value_loss_self", "value_loss_human", "top1_self", "top1_human"):
        assert k in met and torch.isfinite(met[k])
    assert abs(float(met["policy_loss"]) - 0.5 * (float(met["policy_loss_self"]) + float(met["policy_loss_human"]))) < 1e-4
    # the stream cycles through the shards forever
    n_pos = sum(int(np.load(f)["move"].shape[0]) for f in human.files)
    for _ in range(n_pos // 8 + 2):
        human.next()
    assert human.epoch >= 1


def test_lockstep_batches_and_openings(model, graph, shards):
    cfg = v2_cfg(shards, mcts_sims=2, max_game_plies=6)
    seen = []
    games = generate_games(model, graph, cfg, 5, DEVICE, np.random.default_rng(0), on_game=seen.append, lockstep=2)
    assert len(games) == 5 and len(seen) == 5 and all(g.plies >= 1 for g in games)
    assert len(fixed_openings()) == 20 == len(set(fixed_openings()))
    fens = match_openings(games, 8, np.random.default_rng(0))
    assert len(fens) == 8 == len(set(fens))           # games too short: topped up from the fixed set
    assert match_openings([], 3, np.random.default_rng(0)) == list(fixed_openings()[:3])


# ---- the gate --------------------------------------------------------------------------------------
def test_gate_candidate_match(model, graph, shards):
    cfg = v2_cfg(shards, max_game_plies=8)
    best = _state(model)
    fens = sp.paired_start_fens(list(fixed_openings()[:2]), 4)
    out = gate_candidate(model, best, graph, cfg, torch.device(DEVICE), games=4, sims=2, threshold=0.55,
                         start_fens=fens, seed=1)
    assert out is not None and out["games"] == 4 and out["wins"] + out["draws"] + out["losses"] == 4
    assert out["score"] == (out["wins"] + 0.5 * out["draws"]) / 4 and out["threshold"] == 0.55
    assert out["promoted"] == (out["score"] >= 0.55) and out["openings"] == 2 and out["sims"] == 2
    assert out["player_a"] == "candidate" and out["player_b"] == "best" and len(out["pgns"]) == 4
    assert not model.training


def test_v2_iteration_smoke_writes_best_and_gate_record(model, graph, shards, tmp_path):
    cfg = v2_cfg(shards)
    run_dir = tmp_path / "run"
    before = _state(model)
    with MetricsLogger(run_dir, rate_limits={"game": 3600.0}) as logger:   # one sample game per hour ...
        step = run_selfplay_stage(model, graph, cfg, logger, start_step=10, save_checkpoint=_saver(run_dir, model, cfg),
                                  device=DEVICE)
    assert step == 16 and not model.training
    assert (run_dir / "ckpt-16.pt").exists() and (run_dir / "latest.pt").exists() and (run_dir / "best.pt").exists()
    best = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    latest = torch.load(run_dir / "latest.pt", map_location="cpu", weights_only=False)
    assert best["step"] == 16 and best["stage"] == "selfplay" and _same(best["model"], latest["model"])
    assert _same(best["model"], _state(model))       # the model holds the best network
    recs = read_metrics(run_dir)
    kinds = {r["kind"] for r in recs}
    assert {"train", "game", "selfplay", "status", "elo"} <= kinds
    gate = [r for r in recs if r["kind"] == "elo" and r["opponent"] == GATE_OPPONENT]
    assert len(gate) == 1
    g = gate[0]
    assert g["games"] == 4 and g["wins"] + g["draws"] + g["losses"] == 4 and g["iteration"] == 1
    assert g["threshold"] == 0.55 and g["sims"] == 4 and g["openings"] >= 1 and isinstance(g["promoted"], bool)
    assert g["promoted"] == (g["score"] >= 0.55)
    assert 0 <= g["fixed_openings_used"] <= g["openings"] and 0 <= g["truncated"] <= 4 and g["mean_plies"] > 0
    games = [r for r in recs if r["kind"] == "game"]
    assert [x["source"] for x in games] == ["selfplay", "eval"]   # ... but the gate game bypasses the limit
    assert games[1]["opponent"] == GATE_OPPONENT and games[1]["moves"] >= 1
    if g["promoted"]:
        assert not _same(before, _state(model))
    else:
        assert _same(before, _state(model))
    elos = {r["opponent"] for r in recs if r["kind"] == "elo"}
    assert elos == {GATE_OPPONENT, "random"}          # the gate replaces the 'previous-iteration' eval
    train = [r for r in recs if r["kind"] == "train"]
    assert len(train) == 6 and train[-1]["step"] == 16 and train[-1]["stage"] == "selfplay"
    assert train[-1]["n_self"] == 8 and train[-1]["n_human"] == 8 and train[-1]["human_mix"] == 0.5
    assert {"policy_loss_self", "policy_loss_human", "value_loss_human", "top1_human"} <= set(train[-1])
    sel = next(r for r in recs if r["kind"] == "selfplay")
    assert sel["games"] == 8 and sel["sims"] == 8 and sel["buffer"] == sel["positions"]
    status = [r for r in recs if r["kind"] == "status"]
    assert any("candidate promoted" in s["message"] or "candidate rejected" in s["message"] for s in status)
    assert any(f"8/{cfg.selfplay_games_per_iter} games" in s["message"] for s in status)  # generation progress
    done = [s for s in status if s.get("iteration")]
    assert done[-1]["iteration"] == 1 and done[-1]["promoted"] == g["promoted"] and done[-1]["gated"] is True
    assert completed_iterations(run_dir, 16) == 1 and completed_iterations(run_dir, 15) == 0


def test_gate_rejection_reverts_the_weights(model, graph, shards, tmp_path, monkeypatch):
    """A candidate that loses the gate is discarded: the model, latest.pt and best.pt keep the best weights,
    the replay buffer is kept and the iteration still counts."""
    cfg = v2_cfg(shards, selfplay_iters=2, max_game_plies=10)
    run_dir = tmp_path / "run"
    real_train = sp._train_iteration
    real_match = sp._run_match

    def random_candidate(model_, optim, buffer, human, p, rng, logger, step, it):
        step, last = real_train(model_, optim, buffer, human, p, rng, logger, step, it)
        with torch.no_grad():  # the candidate becomes a random-weight copy
            for v in model_.parameters():
                v.copy_(torch.randn_like(v))
        return step, last

    def losing_match(player_a, player_b, games, max_plies, start_fens, seed):
        if player_a.name != "candidate":  # the periodic evaluation vs random: not under test here
            return real_match(player_a, player_b, games, max_plies, start_fens, seed)
        assert player_b.name == "best" and player_a.difficulty == player_b.difficulty == "superfly"
        assert start_fens is not None and len(start_fens) == games and start_fens[0] == start_fens[1]
        return {"player_a": player_a.name, "player_b": player_b.name, "games": games, "wins": 0, "draws": 1,
                "losses": games - 1, "score": 0.5 / games, "elo_estimate": -400.0, "pgns": [], "pgn": "",
                "openings": len(set(start_fens))}

    monkeypatch.setattr(sp, "_train_iteration", random_candidate)
    monkeypatch.setattr(sp, "_run_match", losing_match)
    before = _state(model)
    with MetricsLogger(run_dir, rate_limits={}) as logger:
        step = run_selfplay_stage(model, graph, cfg, logger, 0, _saver(run_dir, model, cfg), device=DEVICE)
    assert step == 12
    assert _same(before, _state(model))                                  # reverted, both iterations
    best = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    assert _same(best["model"], before) and best["step"] == 12
    latest = torch.load(run_dir / "latest.pt", map_location="cpu", weights_only=False)
    assert _same(latest["model"], before)
    # the AdamW moments of the rejected candidates went with them: the pre-training snapshot (empty) is back
    assert latest["optim"]["state"] == {} and best["optim"]["state"] == {}
    assert all(tuple(g["betas"]) == (0.9, 0.95) for g in latest["optim"]["param_groups"])
    recs = read_metrics(run_dir)
    gate = [r for r in recs if r["kind"] == "elo" and r["opponent"] == GATE_OPPONENT]
    assert [g["promoted"] for g in gate] == [False, False] and [g["iteration"] for g in gate] == [1, 2]
    assert gate[0]["wins"] == 0 and gate[0]["losses"] == 3 and gate[0]["draws"] == 1 and gate[0]["score"] == 0.125
    sel = [r for r in recs if r["kind"] == "selfplay"]
    assert len(sel) == 2 and sel[1]["buffer"] == sel[0]["positions"] + sel[1]["positions"]   # buffer kept
    done = [s for s in recs if s["kind"] == "status" and s.get("iteration")]
    assert [s["iteration"] for s in done] == [1, 2] and all(s["promoted"] is False for s in done)
    assert any("candidate rejected" in s["message"] for s in recs if s["kind"] == "status")
    assert completed_iterations(run_dir, 12) == 2


def test_gate_promotion_keeps_the_candidate(model, graph, shards, tmp_path, monkeypatch):
    cfg = v2_cfg(shards, max_game_plies=10)
    run_dir = tmp_path / "run"

    real_match = sp._run_match

    def winning_match(player_a, player_b, games, max_plies, start_fens, seed):
        if player_a.name != "candidate":
            return real_match(player_a, player_b, games, max_plies, start_fens, seed)
        return {"player_a": player_a.name, "player_b": player_b.name, "games": games, "wins": games, "draws": 0,
                "losses": 0, "score": 1.0, "elo_estimate": 800.0, "pgns": [], "pgn": "", "openings": 2}

    monkeypatch.setattr(sp, "_run_match", winning_match)
    before = _state(model)
    with MetricsLogger(run_dir, rate_limits={}) as logger:
        run_selfplay_stage(model, graph, cfg, logger, 0, _saver(run_dir, model, cfg), device=DEVICE)
    after = _state(model)
    assert not _same(before, after)
    best = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    assert _same(best["model"], after)
    assert best["optim"]["state"] and all(int(v["step"]) == 6 for v in best["optim"]["state"].values())  # moments kept
    gate = [r for r in read_metrics(run_dir, kinds=["elo"]) if r["opponent"] == GATE_OPPONENT]
    assert gate[0]["promoted"] is True and gate[0]["score"] == 1.0


def test_gate_disabled_keeps_v1_evaluation(model, graph, shards, tmp_path):
    cfg = v2_cfg(shards, selfplay_gate=False, max_game_plies=10)
    run_dir = tmp_path / "run"
    before = _state(model)
    with MetricsLogger(run_dir, rate_limits={}) as logger:
        run_selfplay_stage(model, graph, cfg, logger, 0, _saver(run_dir, model, cfg), device=DEVICE)
    assert not _same(before, _state(model))
    elos = {r["opponent"] for r in read_metrics(run_dir, kinds=["elo"])}
    assert elos == {"previous-iteration", "random"}
    assert (run_dir / "best.pt").exists()
    done = [s for s in read_metrics(run_dir, kinds=["status"]) if s.get("iteration")]
    assert done[-1]["promoted"] is True and done[-1]["gated"] is False


def test_resume_after_one_iteration_loads_best(model, graph, shards, tmp_path):
    cfg = v2_cfg(shards, max_game_plies=10)
    run_dir = tmp_path / "run"
    save = _saver(run_dir, model, cfg)
    with MetricsLogger(run_dir, rate_limits={}) as logger:
        step = run_selfplay_stage(model, graph, cfg, logger, 0, save, device=DEVICE)
    assert step == 6 and completed_iterations(run_dir, 6) == 1
    best = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)["model"]
    # the stage is finished: nothing more runs
    with MetricsLogger(run_dir, rate_limits={}) as logger:
        assert run_selfplay_stage(model, graph, cfg, logger, 6, save, device=DEVICE) == 6
    # perturb the live weights (as if latest.pt held an ungated candidate): resuming reloads best.pt
    with torch.no_grad():
        for v in model.parameters():
            v.add_(0.1)
    assert not _same(best, _state(model))
    cfg.selfplay_iters = 2
    loaded = {}
    real = sp._load_best

    def spy(run_dir_, model_):
        loaded["step"] = real(run_dir_, model_)
        loaded["weights"] = _state(model_)
        return loaded["step"]

    sp._load_best = spy
    try:
        with MetricsLogger(run_dir, rate_limits={}) as logger:
            step = run_selfplay_stage(model, graph, cfg, logger, 6, save, device=DEVICE)
    finally:
        sp._load_best = real
    assert step == 12 and loaded["step"] == 6 and _same(best, loaded["weights"])
    status = [r for r in read_metrics(run_dir, kinds=["status"]) if "resuming" in r["message"]]
    assert status and status[-1]["message"].endswith("after iteration 1/2") and status[-1]["best_loaded_step"] == 6
    done = [s for s in read_metrics(run_dir, kinds=["status"]) if s.get("iteration")]
    assert [s["iteration"] for s in done] == [1, 2]
    assert completed_iterations(run_dir, 12) == 2


@pytest.mark.parametrize("failure", ["raises", "none"])
def test_gate_failure_rejects_the_candidate_and_stops(model, graph, shards, tmp_path, monkeypatch, failure):
    """A gate that cannot run never promotes: the candidate is discarded (weights + optimizer), best.pt /
    latest.pt hold the best weights, the iteration does not count and the stage stops with RuntimeError."""
    cfg = v2_cfg(shards, selfplay_iters=2, max_game_plies=10)
    run_dir = tmp_path / "run"

    def broken_match(player_a, player_b, games, max_plies, start_fens, seed):
        if failure == "raises":
            raise torch.OutOfMemoryError("CUDA out of memory (simulated)")
        # else: None — FlyEngine unimportable / an unexpected match result

    monkeypatch.setattr(sp, "_run_match", broken_match)
    before = _state(model)
    with MetricsLogger(run_dir, rate_limits={}) as logger, pytest.raises(RuntimeError, match="gate could not run"):
        run_selfplay_stage(model, graph, cfg, logger, 0, _saver(run_dir, model, cfg), device=DEVICE)
    assert _same(before, _state(model)) and not model.training
    for name in ("best.pt", "latest.pt", "ckpt-6.pt"):
        ck = torch.load(run_dir / name, map_location="cpu", weights_only=False)
        assert _same(ck["model"], before) and ck["step"] == 6 and ck["optim"]["state"] == {}
    recs = read_metrics(run_dir)
    assert not [r for r in recs if r["kind"] == "elo"]                     # no gate record, no evaluation
    failed = [r for r in recs if r["kind"] == "status" and r.get("gate_error")]
    assert len(failed) == 1 and failed[0]["promoted"] is False and failed[0]["gated"] is False
    assert "iteration" not in failed[0] and "candidate rejected" in failed[0]["message"]
    assert ("OutOfMemoryError" in failed[0]["gate_error"]) == (failure == "raises")
    assert completed_iterations(run_dir, 6) == 0                             # --resume redoes the iteration


def test_runtime_error_during_training_restores_best(model, graph, shards, tmp_path, monkeypatch):
    cfg = v2_cfg(shards, max_game_plies=10)
    run_dir = tmp_path / "run"
    real_train = sp._train_iteration

    def exploding_train(model_, optim, buffer, human, p, rng, logger, step, it):
        real_train(model_, optim, buffer, human, p, rng, logger, step, it)
        with torch.no_grad():
            for v in model_.parameters():
                v.add_(1.0)
        raise RuntimeError("self-play loss became non-finite (simulated)")

    monkeypatch.setattr(sp, "_train_iteration", exploding_train)
    before = _state(model)
    with MetricsLogger(run_dir, rate_limits={}) as logger, pytest.raises(RuntimeError, match="non-finite"):
        run_selfplay_stage(model, graph, cfg, logger, 0, _saver(run_dir, model, cfg), device=DEVICE)
    assert _same(before, _state(model)) and not model.training
    for name in ("best.pt", "latest.pt"):
        ck = torch.load(run_dir / name, map_location="cpu", weights_only=False)
        assert _same(ck["model"], before) and ck["optim"]["state"] == {}   # weights and moments of the best
    assert completed_iterations(run_dir, ck["step"]) == 0


def test_resume_applies_the_config_lr_and_betas(model, graph, shards, tmp_path):
    cfg = v2_cfg(shards, selfplay_gate=False, max_game_plies=10, selfplay_lr=1e-4)
    run_dir = tmp_path / "run"
    save = _saver(run_dir, model, cfg)
    with MetricsLogger(run_dir, rate_limits={}) as logger:
        assert run_selfplay_stage(model, graph, cfg, logger, 0, save, device=DEVICE) == 6
    first = [r for r in read_metrics(run_dir, kinds=["train"])]
    assert first and all(r["lr"] == 1e-4 for r in first)
    ck = torch.load(run_dir / "latest.pt", map_location="cpu", weights_only=False)
    assert ck["stage"] == "selfplay" and ck["optim"]["param_groups"][0]["lr"] == 1e-4
    # resume with a lower lr: the restored optimizer state keeps its moments but takes the config's lr / betas
    cfg2 = v2_cfg(shards, selfplay_gate=False, max_game_plies=10, selfplay_lr=5e-5, selfplay_iters=2)
    with MetricsLogger(run_dir, rate_limits={}) as logger:
        assert run_selfplay_stage(model, graph, cfg2, logger, 6, _saver(run_dir, model, cfg2), device=DEVICE) == 12
    second = [r for r in read_metrics(run_dir, kinds=["train"]) if r["step"] > 6]
    assert len(second) == 6 and all(r["lr"] == 5e-5 for r in second)
    ck = torch.load(run_dir / "latest.pt", map_location="cpu", weights_only=False)
    assert all(g["lr"] == 5e-5 and tuple(g["betas"]) == (0.9, 0.95) for g in ck["optim"]["param_groups"])
    assert all(int(v["step"]) == 12 for v in ck["optim"]["state"].values())   # the moments were restored


def test_run_match_follows_the_evaluator_signature(monkeypatch):
    class P:
        name = "x"
    calls = []

    def small(a, b, games):   # an evaluator without start_fens / seed / keep_pgns
        calls.append(games)
        return {"wins": 1, "draws": 0, "losses": games - 1, "games": games, "elo_estimate": 0.0}

    monkeypatch.setattr(sp, "_match_fn", lambda: small)
    out = sp._run_match(P(), P(), 4, 10, ["fen-a", "fen-a", "fen-b", "fen-b"], seed=3)
    assert calls == [4] and out["openings"] == 1                            # played from the start position

    def full(a, b, games, max_plies=200, seed=0, on_game=None, keep_pgns=3, start_fens=None):
        assert (max_plies, seed, keep_pgns, list(start_fens)) == (10, 3, 4, ["fen-a", "fen-a", "fen-b", "fen-b"])
        return {"wins": 2, "draws": 0, "losses": 2, "games": 4, "elo_estimate": 0.0}

    monkeypatch.setattr(sp, "_match_fn", lambda: full)
    assert sp._run_match(P(), P(), 4, 10, ["fen-a", "fen-a", "fen-b", "fen-b"], seed=3)["openings"] == 2

    def inner_type_error(a, b, games, **kwargs):   # a bug inside the match must not be replayed with defaults
        raise TypeError("bug in the engine")

    monkeypatch.setattr(sp, "_match_fn", lambda: inner_type_error)
    with pytest.raises(TypeError, match="bug in the engine"):
        sp._run_match(P(), P(), 4, 10, None, seed=0)


def test_missing_shards_is_a_clear_error(model, graph, tmp_path):
    cfg = v2_cfg(tmp_path / "nowhere")
    with MetricsLogger(tmp_path / "run", rate_limits={}) as logger, pytest.raises(FileNotFoundError, match="fix"):
        run_selfplay_stage(model, graph, cfg, logger, 0, lambda step, stage: tmp_path, device=DEVICE)
