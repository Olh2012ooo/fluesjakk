"""Tests for flychess.train (config, imitation stage, trainer / checkpoints). Fast: tiny graph + fixture PGN."""
from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from flychess import paths
from flychess.connectome.graph import BrainGraph
from flychess.data.lichess import build_shards
from flychess.data.shards import list_shards
from flychess.model.config import BrainConfig
from flychess.train.config import TrainConfig, load_config, resolve_graph_path
from flychess.train.imitation import adapt_optim_state, lr_at, param_groups, planned_steps
from flychess.train.metrics import read_metrics, read_run_json
from flychess.train.trainer import cuda_mem_gb, load_checkpoint, resolve_checkpoint, save_checkpoint, train

FIXTURE = Path(__file__).parent / "fixtures" / "small.pgn"
TINY_GRAPH = paths.BRAIN_DIR / "tiny.npz"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
needs_tiny = pytest.mark.skipif(not TINY_GRAPH.exists(), reason="data/brain/tiny.npz not built")


@pytest.fixture(scope="module")
def shards(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("shards")
    stats = build_shards(FIXTURE, out, name="fix", workers=2, shard_size=500, shuffle_buffer=700, progress=False)
    assert stats.positions > 1000 and len(list_shards(out, "fix")) >= 3
    return out


@pytest.fixture
def runs_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setattr(paths, "RUNS_DIR", tmp_path / "runs")
    return tmp_path / "runs"


def tiny_cfg(shards: Path, **overrides) -> TrainConfig:
    return TrainConfig.tiny(shards_dir=str(shards), shard_name="fix", graph=str(TINY_GRAPH), device=DEVICE,
                            **overrides)


# ---- config --------------------------------------------------------------------------------------
def test_config_roundtrip_and_overrides(tmp_path):
    cfg = TrainConfig(run="x", graph="full", brain={"steps": 3})
    assert cfg.brain["steps"] == 3 and cfg.brain["activation"] == "relu"  # defaults filled in
    assert resolve_graph_path(cfg) == paths.BRAIN_DIR / "full.npz"
    assert resolve_graph_path(TrainConfig(graph="/tmp/some/graph.npz")) == Path("/tmp/some/graph.npz")
    bc = cfg.brain_config("/g.npz")
    assert isinstance(bc, BrainConfig) and bc.steps == 3 and bc.graph_path == "/g.npz"
    y = cfg.to_yaml(tmp_path / "c.yaml")
    assert TrainConfig.from_yaml(y) == cfg
    assert TrainConfig.from_dict(cfg.to_dict()) == cfg
    c2 = cfg.replace(**{"brain.steps": 5, "lr": 3e-4, "max_steps": 7})
    assert c2.brain["steps"] == 5 and c2.lr == 3e-4 and c2.max_steps == 7 and cfg.brain["steps"] == 3
    assert cfg.replace(brain={"activation": "tanh"}).brain["activation"] == "tanh"
    assert load_config(str(y), lr=1.0).lr == 1.0 and load_config(None).run == "fly1"
    with pytest.raises(KeyError):
        TrainConfig.from_dict({"nope": 1})
    with pytest.raises(KeyError):
        TrainConfig(brain={"bogus": 1})
    tiny = TrainConfig.tiny()
    assert tiny.graph == "tiny" and tiny.batch_size == 32
    repo_yaml = Path(__file__).parent.parent / "configs" / "tiny.yaml"
    if repo_yaml.exists():  # the yaml pins device=cuda; the preset picks the available device
        assert TrainConfig.from_yaml(repo_yaml).replace(device=tiny.device).to_dict() == tiny.to_dict()
    default_yaml = repo_yaml.with_name("default.yaml")
    if default_yaml.exists():
        d = TrainConfig.from_yaml(default_yaml)
        assert d.graph == "full" and d.batch_size == 256 and d.lr == 1e-3 and d.epochs == 1
        # the yaml must not drift from the built-in defaults (`fly train --run fly1` without --config),
        # in particular shard_name must accept whatever `fly build-shards` wrote (default name "lichess")
        assert d.shard_name is None
        # `device` is the one field the yaml and the built-in defaults legitimately disagree on: the
        # yaml pins cuda, the preset asks the machine (same normalisation as for tiny.yaml above).
        # Comparing without it keeps the drift check meaningful on a CPU-only box.
        assert d.replace(device=TrainConfig().device).to_dict() == TrainConfig().to_dict()
        assert d.device == "cuda", "default.yaml trains on GPU unless --device overrides it"


def test_lr_schedule_and_planned_steps():
    assert lr_at(0, 1.0, 10, 100) == pytest.approx(0.1)
    assert lr_at(9, 1.0, 10, 100) == pytest.approx(1.0)
    assert lr_at(10, 1.0, 10, 100) == pytest.approx(1.0)
    assert lr_at(55, 1.0, 10, 100) == pytest.approx(0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * 0.5)))
    assert lr_at(100, 1.0, 10, 100) == pytest.approx(0.1)
    assert lr_at(500, 1.0, 10, 100) == pytest.approx(0.1)  # never below the floor
    cfg = TrainConfig(batch_size=10, epochs=2)
    assert planned_steps(cfg, 105) == (20, 10)
    assert planned_steps(cfg.replace(max_steps=7), 105) == (7, 10)
    assert planned_steps(cfg.replace(epochs=0, max_steps=99), 105) == (99, 10)


@needs_tiny
def test_param_groups_exclude_biases_and_leaks():
    from flychess.model.flybrain import FlyBrain

    graph = BrainGraph.load(TINY_GRAPH)
    model = FlyBrain(graph, BrainConfig(graph_path=str(TINY_GRAPH), steps=2, value_hidden=8))
    groups = param_groups(model, 0.1)
    decay = {id(p) for p in groups[0]["params"]}
    no_decay = {id(p) for p in groups[1]["params"]}
    assert id(model.bias) in no_decay and id(model.leak_logit) in no_decay and id(model.b_in) in no_decay
    assert id(model.policy_head.bias) in no_decay
    assert id(model.w_in) in decay and id(model.policy_head.weight) in decay
    assert len(decay) + len(no_decay) == len(list(model.parameters()))
    # Dale: |w| = softplus(syn_gain), so decaying the logit would pull synapses toward ln 2, not 0 -> no decay
    assert model.dale and id(model.syn_gain) in no_decay
    assert id(model.syn_gain) in {id(p) for p in param_groups(model, 0.1, decay_syn_gain=True)[0]["params"]}
    free = FlyBrain(graph, BrainConfig(graph_path=str(TINY_GRAPH), steps=2, value_hidden=8, dale=False))
    assert id(free.syn_gain) in {id(p) for p in param_groups(free, 0.1)[0]["params"]}  # w = gain: decay is real


@needs_tiny
def test_adapt_optim_state_remaps_legacy_layouts():
    """Checkpoints written with syn_gain in the decayed group (or self-play's single group) stay resumable."""
    from flychess.model.flybrain import FlyBrain

    graph = BrainGraph.load(TINY_GRAPH)
    model = FlyBrain(graph, BrainConfig(graph_path=str(TINY_GRAPH), steps=2, value_hidden=8))
    legacy_groups = [param_groups(model, 0.1, decay_syn_gain=True),
                     [{"params": list(model.parameters()), "weight_decay": 0.1}]]
    for groups in legacy_groups:
        old = torch.optim.AdamW(groups, lr=1e-3)
        for p in model.parameters():
            p.grad = torch.randn_like(p)
        old.step()
        saved = old.state_dict()
        new = torch.optim.AdamW(param_groups(model, 0.1), lr=1e-3)
        with pytest.raises(ValueError):
            new.load_state_dict(saved)  # positional group sizes differ -> torch refuses the raw state
        adapted = adapt_optim_state(saved, model, 0.1)
        assert [len(g["params"]) for g in adapted["param_groups"]] == [len(g["params"]) for g in new.param_groups]
        assert [g["weight_decay"] for g in adapted["param_groups"]] == [0.1, 0.0]
        new.load_state_dict(adapted)
        for p in model.parameters():  # every parameter keeps its own moments
            torch.testing.assert_close(new.state[p]["exp_avg"], old.state[p]["exp_avg"])
            torch.testing.assert_close(new.state[p]["exp_avg_sq"], old.state[p]["exp_avg_sq"])
        assert saved["param_groups"] is not adapted["param_groups"]  # the checkpoint dict itself is untouched
    # a state that already matches, or an unknown layout, passes through unchanged
    fresh = torch.optim.AdamW(param_groups(model, 0.1), lr=1e-3).state_dict()
    assert adapt_optim_state(fresh, model, 0.1) is fresh
    odd = {"state": {}, "param_groups": [{"params": [0, 1], "lr": 1e-3}]}
    assert adapt_optim_state(odd, model, 0.1) is odd


# ---- imitation + trainer -------------------------------------------------------------------------
@needs_tiny
def test_imitation_stage_trains_and_checkpoints(shards, runs_dir):
    cfg = tiny_cfg(shards, max_steps=40, log_every=1, eval_every=20, checkpoint_every=20, elo_every=20,
                   elo_games=2, lr=2e-3)
    latest = train("imit", stage="imitation", config=cfg)
    run_dir = runs_dir / "imit"
    assert latest == run_dir / "latest.pt" and latest.exists()
    assert (run_dir / "ckpt-20.pt").exists() and (run_dir / "ckpt-40.pt").exists()
    assert (run_dir / "metrics.jsonl").exists()
    info = read_run_json(run_dir)
    assert info and info["config"]["run"] == "imit" and info["graph_meta"]["n"] == 2000
    assert info["graph_path"] == str(TINY_GRAPH)

    train_recs = read_metrics(run_dir, kinds=["train"])
    assert len(train_recs) == 40 and train_recs[-1]["step"] == 40
    for key in ("loss", "policy_loss", "value_loss", "top1", "top3", "lr", "pos_per_sec", "gpu_mem_gb", "epoch"):
        assert key in train_recs[0]
    assert train_recs[0]["stage"] == "imitation" and train_recs[0]["pos_per_sec"] > 0
    first = sum(r["loss"] for r in train_recs[:5]) / 5
    last = sum(r["loss"] for r in train_recs[-5:]) / 5
    assert last < first, (first, last)
    evals = read_metrics(run_dir, kinds=["eval"])
    assert [e["step"] for e in evals] == [0, 20, 40]
    assert evals[-1]["val_loss"] < evals[0]["val_loss"]
    assert all(k in evals[0] for k in ("val_loss", "val_top1", "val_top3", "val_value_mse"))
    elo = read_metrics(run_dir, kinds=["elo"])
    assert {e["opponent"] for e in elo} == {"random", "material"}
    assert all(e["games"] == 2 == e["wins"] + e["draws"] + e["losses"] for e in elo)
    games = read_metrics(run_dir, kinds=["game"])
    assert games and games[0]["source"] == "eval" and "1." in games[0]["pgn"]
    act = read_metrics(run_dir, kinds=["activity"])
    assert act and len(act[0]["neuron_idx"]) == 2000 == len(act[0]["values"])  # tiny graph has < 2048 neurons
    status = read_metrics(run_dir, kinds=["status"])
    assert status[-1]["total_steps"] == 40 and status[-1]["eta_s"] == 0

    # checkpoint round trip
    model, graph, ckpt = load_checkpoint("imit", device=DEVICE)
    assert ckpt["step"] == 40 and ckpt["stage"] == "imitation" and ckpt["optim"] is not None
    assert set(ckpt["elo"]) == {"random", "material"}
    assert ckpt["config"]["run"] == "imit" and ckpt["brain_config"]["steps"] == 4
    assert graph.n == model.n == 2000 and not model.training
    raw = torch.load(latest, map_location="cpu", weights_only=False)
    for k, v in raw["model"].items():
        assert torch.equal(model.state_dict()[k].cpu(), v)
    assert resolve_checkpoint(run_dir) == latest and resolve_checkpoint(str(latest)) == latest
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint("no-such-run")
    x = torch.zeros(2, 1280, device=DEVICE)
    with torch.no_grad():
        logits, value = model(x)
    assert logits.shape == (2, 4168) and value.shape == (2, 1)


@needs_tiny
def test_resume_continues_step_counter(shards, runs_dir):
    cfg = tiny_cfg(shards, max_steps=12, log_every=4, eval_every=100, checkpoint_every=100, elo_every=0)
    train("res", stage="imitation", config=cfg)
    _, _, ckpt = load_checkpoint("res")
    assert ckpt["step"] == 12 and ckpt["elo"] == {}
    latest = train("res", stage="imitation", config=cfg.replace(max_steps=20), resume=True)
    _, _, ckpt2 = load_checkpoint(latest)
    assert ckpt2["step"] == 20
    steps = [r["step"] for r in read_metrics(runs_dir / "res", kinds=["train"])]
    assert steps == [4, 8, 12, 16, 20]
    info = read_run_json(runs_dir / "res")
    assert info["resumed_from_step"] == 12
    # resuming a finished stage is a no-op that keeps the checkpoint
    assert train("res", stage="imitation", config=cfg.replace(max_steps=20), resume=True) == latest
    assert load_checkpoint("res")[2]["step"] == 20


@needs_tiny
def test_save_checkpoint_direct_and_helpers(tmp_path):
    from flychess.model.flybrain import FlyBrain

    graph = BrainGraph.load(TINY_GRAPH)
    cfg = TrainConfig.tiny(graph=str(TINY_GRAPH))
    model = FlyBrain(graph, cfg.brain_config())
    p = save_checkpoint(tmp_path / "r", model, cfg, TINY_GRAPH, 5, "selfplay", optim=None, elo={"random": 12.5})
    assert p == tmp_path / "r" / "ckpt-5.pt" and (tmp_path / "r" / "latest.pt").exists()
    m2, g2, ck = load_checkpoint(p, graph=graph)
    assert ck["stage"] == "selfplay" and ck["elo"] == {"random": 12.5} and ck["optim"] is None and g2 is graph
    assert torch.equal(m2.syn_gain, model.syn_gain)
    assert cuda_mem_gb() >= 0.0


def test_selfplay_stage_missing_module_is_clear(shards, runs_dir, monkeypatch):
    import builtins

    if not TINY_GRAPH.exists():
        pytest.skip("tiny graph missing")
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "flychess.train.selfplay":
            raise ImportError("simulated: module under construction")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    cfg = tiny_cfg(shards, max_steps=2, eval_every=100, checkpoint_every=100, elo_every=0)
    with pytest.raises(RuntimeError, match="self-play stage"):
        train("sp", stage="selfplay", config=cfg)


@needs_tiny
def test_keyboard_interrupt_saves_checkpoint(shards, runs_dir, monkeypatch):
    from flychess.train import imitation

    real_loss = imitation.policy_value_loss
    calls = {"n": 0}

    def interrupting_loss(*a, **k):
        if torch.is_grad_enabled():  # training calls only (evaluation runs under no_grad)
            calls["n"] += 1
            if calls["n"] == 8:  # 7 optimizer steps done, interrupted while computing the 8th
                raise KeyboardInterrupt
        return real_loss(*a, **k)

    monkeypatch.setattr(imitation, "policy_value_loss", interrupting_loss)
    cfg = tiny_cfg(shards, max_steps=50, log_every=100, eval_every=100, checkpoint_every=100, elo_every=0)
    latest = train("kbi", stage="imitation", config=cfg)
    assert latest.exists()
    _, _, ckpt = load_checkpoint(latest)
    assert ckpt["step"] == 7 and ckpt["stage"] == "imitation"
    status = read_metrics(runs_dir / "kbi", kinds=["status"])
    assert any(s["message"] == "interrupted" for s in status)


# ---- regression tests (review findings) ---------------------------------------------------------
def test_config_rejects_zero_cadences():
    """log_every / eval_every / checkpoint_every == 0 used to pass validation and crash with ZeroDivisionError."""
    for name in ("log_every", "eval_every", "checkpoint_every", "eval_batches", "elo_games"):
        with pytest.raises(ValueError, match=name):
            TrainConfig(**{name: 0})
    with pytest.raises(ValueError, match="warmup_steps"):
        TrainConfig(warmup_steps=-1)
    with pytest.raises(ValueError, match="elo_every"):
        TrainConfig(elo_every=-5)
    assert TrainConfig(elo_every=0, warmup_steps=0, num_workers=0).elo_every == 0  # 0 = disabled is fine
    with pytest.raises(ValueError):
        TrainConfig(lr=0.0)


@needs_tiny
def test_resume_without_config_uses_checkpoint_config(shards, runs_dir):
    """`train(run, resume=True)` with config=None must continue with the run's own config, not the defaults."""
    cfg = tiny_cfg(shards, max_steps=6, log_every=3, eval_every=100, checkpoint_every=100, elo_every=0, lr=5e-4)
    train("rc", stage="imitation", config=cfg)
    # resume with no config at all, only a step-cap override: the tiny graph / batch / lr / shards must survive
    latest = train("rc", stage="imitation", resume=True, max_steps=9)
    _, _, ckpt = load_checkpoint(latest)
    assert ckpt["step"] == 9
    c = ckpt["config"]
    assert c["graph"] == str(TINY_GRAPH) and c["batch_size"] == 32 and c["lr"] == 5e-4 and c["max_steps"] == 9
    assert c["shards_dir"] == str(shards) and c["shard_name"] == "fix" and c["run"] == "rc"
    assert ckpt["brain_config"]["steps"] == 4
    steps = [r["step"] for r in read_metrics(runs_dir / "rc", kinds=["train"])]
    assert steps == [3, 6, 9]
    info = read_run_json(runs_dir / "rc")
    assert info["config"]["graph"] == str(TINY_GRAPH) and info["config"]["max_steps"] == 9


@needs_tiny
def test_resume_applies_new_weight_decay(shards, runs_dir):
    """optim.load_state_dict used to restore the checkpoint's weight_decay over a changed cfg.weight_decay."""
    cfg = tiny_cfg(shards, max_steps=4, log_every=100, eval_every=100, checkpoint_every=100, elo_every=0)
    train("wd", stage="imitation", config=cfg)
    raw = torch.load(runs_dir / "wd" / "latest.pt", map_location="cpu", weights_only=False)
    assert [g["weight_decay"] for g in raw["optim"]["param_groups"]] == [cfg.weight_decay, 0.0]
    latest = train("wd", stage="imitation", config=cfg.replace(weight_decay=0.5, max_steps=6), resume=True)
    raw = torch.load(latest, map_location="cpu", weights_only=False)
    assert raw["step"] == 6 and raw["config"]["weight_decay"] == 0.5
    assert [g["weight_decay"] for g in raw["optim"]["param_groups"]] == [0.5, 0.0]
    assert all(tuple(g["betas"]) == (0.9, 0.95) for g in raw["optim"]["param_groups"])


@needs_tiny
def test_resume_from_checkpoint_with_decayed_syn_gain_layout(shards, runs_dir, capsys):
    """A checkpoint saved when syn_gain was still in the decayed group (runs/fly1) resumes with its AdamW moments."""
    from flychess.model.flybrain import FlyBrain

    cfg = tiny_cfg(shards, max_steps=4, log_every=100, eval_every=100, checkpoint_every=100, elo_every=0)
    train("legacy", stage="imitation", config=cfg)
    latest = runs_dir / "legacy" / "latest.pt"
    raw = torch.load(latest, map_location="cpu", weights_only=False)
    graph = BrainGraph.load(TINY_GRAPH)
    model = FlyBrain(graph, cfg.brain_config(str(TINY_GRAPH)))
    current, legacy = param_groups(model, cfg.weight_decay), param_groups(model, cfg.weight_decay, decay_syn_gain=True)
    saved_index = {id(p): i for g, sg in zip(current, raw["optim"]["param_groups"], strict=True)
                   for p, i in zip(g["params"], sg["params"], strict=True)}
    raw["optim"]["param_groups"] = [dict(sg, params=[saved_index[id(p)] for p in g["params"]])
                                    for g, sg in zip(legacy, raw["optim"]["param_groups"], strict=True)]
    sizes = [len(g["params"]) for g in raw["optim"]["param_groups"]]  # latest.pt now has the pre-fix layout
    assert sizes != [len(g["params"]) for g in current]
    torch.save(raw, latest)
    capsys.readouterr()
    latest = train("legacy", stage="imitation", config=cfg.replace(max_steps=6), resume=True)
    assert "optimizer state not restored" not in capsys.readouterr().out
    raw = torch.load(latest, map_location="cpu", weights_only=False)
    assert raw["step"] == 6
    assert [g["weight_decay"] for g in raw["optim"]["param_groups"]] == [cfg.weight_decay, 0.0]
    assert all(int(st["step"]) == 6 for st in raw["optim"]["state"].values())  # moments continued, not reset


@needs_tiny
def test_interrupt_in_final_elo_keeps_final_weights(shards, runs_dir, monkeypatch):
    """Ctrl-C during the end-of-training eval/Elo used to lose every step since the last periodic checkpoint."""
    from flychess.train import imitation

    def interrupting_elo(model, cfg, device, logger, step, games=None):
        raise KeyboardInterrupt

    monkeypatch.setattr(imitation, "quick_elo", interrupting_elo)
    cfg = tiny_cfg(shards, max_steps=30, log_every=100, eval_every=100, checkpoint_every=20, elo_every=1000)
    latest = train("fin", stage="imitation", config=cfg)
    _, _, ckpt = load_checkpoint(latest)
    assert ckpt["step"] == 30, "the final weights must be saved even if the final Elo is interrupted"
    assert (runs_dir / "fin" / "ckpt-20.pt").exists() and (runs_dir / "fin" / "ckpt-30.pt").exists()


@needs_tiny
def test_throughput_window_excludes_eval_time(shards, runs_dir, monkeypatch):
    """pos_per_sec / ETA are measured over training steps only (eval/Elo/checkpoint time is excluded)."""
    import time

    from flychess.train import imitation

    real_eval = imitation.evaluate

    def slow_evaluate(*a, **k):
        time.sleep(0.4)
        return real_eval(*a, **k)

    monkeypatch.setattr(imitation, "evaluate", slow_evaluate)
    cfg = tiny_cfg(shards, max_steps=16, log_every=1, eval_every=8, checkpoint_every=100, elo_every=0)
    train("tp", stage="imitation", config=cfg)
    recs = {r["step"]: r for r in read_metrics(runs_dir / "tp", kinds=["train"])}
    others = sorted(r["pos_per_sec"] for s, r in recs.items() if s not in (1, 9))
    median = others[len(others) // 2]
    # step 9 is the first step after the eval at step 8: a 0.4 s eval inside a ~ms window would show a >50x dip
    assert recs[9]["pos_per_sec"] > 0.1 * median, (recs[9]["pos_per_sec"], median)


@needs_tiny
def test_resume_with_epochs_zero_continues_data_cycle(shards, runs_dir, monkeypatch):
    """epochs=0 (cycle until max_steps) used to restart the shard permutation at epoch 0 on resume."""
    from flychess.data.shards import ShardDataset, count_positions, load_split
    from flychess.train import imitation

    calls: list[int] = []
    real_set_epoch = ShardDataset.set_epoch

    def recording_set_epoch(self, epoch):
        calls.append(int(epoch))
        return real_set_epoch(self, epoch)

    monkeypatch.setattr(imitation.ShardDataset, "set_epoch", recording_set_epoch)
    base = tiny_cfg(shards, epochs=0, log_every=100, eval_every=1000, checkpoint_every=1000, elo_every=0)
    train_files, _ = load_split(base.shards_dir, base.val_fraction, base.seed, base.shard_name)
    spe = count_positions(train_files) // base.batch_size
    assert spe >= 2
    train("cyc", stage="imitation", config=base.replace(max_steps=spe + 1))
    assert calls == [0, 1]
    calls.clear()
    latest = train("cyc", stage="imitation", config=base.replace(max_steps=spe + 3), resume=True)
    assert calls == [1], calls  # continues in epoch 1, does not replay epoch 0
    assert load_checkpoint(latest)[2]["step"] == spe + 3


@needs_tiny
def test_resume_inside_epoch_continues_with_unseen_batches(shards, runs_dir, monkeypatch):
    """A resume at step S used to replay batches 1..S of the epoch (and, with epochs=1, never reach its tail)."""
    import hashlib

    from flychess.train import imitation

    seen: list[str] = []
    real_collate = imitation.collate

    def recording_collate(batch):  # num_workers=0 only: worker processes would not share `seen`
        out = real_collate(batch)
        seen.append(hashlib.md5(out["planes"].numpy().tobytes()).hexdigest())
        return out

    monkeypatch.setattr(imitation, "collate", recording_collate)
    base = tiny_cfg(shards, epochs=1, num_workers=0, eval_batches=1, log_every=100, eval_every=1000,
                    checkpoint_every=1000, elo_every=0)
    train("fresh", stage="imitation", config=base.replace(max_steps=12))
    fresh = [h for h in seen if h not in seen[:1]]  # drop the ValCache batch (first collate call)
    val_hash, seen[:] = seen[0], []
    assert len(fresh) == 12
    train("part", stage="imitation", config=base.replace(max_steps=6))
    first = [h for h in seen if h != val_hash]
    seen.clear()
    assert first == fresh[:6]
    latest = train("part", stage="imitation", config=base.replace(max_steps=12), resume=True)
    resumed = [h for h in seen if h != val_hash]
    assert load_checkpoint(latest)[2]["step"] == 12
    # the loader is replayed past the 6 already-trained batches, then the run trains on batches 7..12
    assert resumed == fresh, (resumed[:3], fresh[:3])
    status = read_metrics(runs_dir / "part", kinds=["status"])
    assert any("skipping 6" in s["message"] for s in status)
    steps = [r["step"] for r in read_metrics(runs_dir / "part", kinds=["train"])]
    assert steps == [6, 12]  # log_every=100: the two runs log their final step only; nothing trained twice
    fresh_last = read_metrics(runs_dir / "fresh", kinds=["train"])[-1]
    assert read_metrics(runs_dir / "part", kinds=["train"])[-1]["loss"] == fresh_last["loss"]  # same data, same loss
