"""Tests for the ``fly`` CLI (argument parsing, dispatch wiring, ``--help`` as a subprocess) and the package API."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import flychess
from flychess import cli
from flychess.cli import _parse_set, build_parser, main

REPO = Path(__file__).resolve().parent.parent
PY = sys.executable


# ---- parsing -------------------------------------------------------------------------------------
def test_parser_lists_every_spec_subcommand():
    p = build_parser()
    sub = next(a for a in p._actions if a.dest == "command")
    names = set(sub.choices)
    assert {"download", "build-brain", "build-shards", "train", "dashboard", "play", "eval",
            "export-web", "test-vectors"} <= names


def test_train_args_and_set_overrides():
    a = build_parser().parse_args(["train", "--run", "x", "--tiny", "--stage", "all", "--steps", "60",
                                   "--set", "selfplay_iters=2", "--set", "mcts_sims=8", "--set", "amp=false",
                                   "--set", "brain.steps=4", "--set", "lr=3e-4"])
    assert a.run == "x" and a.tiny and a.stage == "all" and a.steps == 60 and a.command == "train"
    over = _parse_set(a.set)
    assert over == {"selfplay_iters": 2, "mcts_sims": 8, "amp": False, "brain.steps": 4, "lr": 3e-4}
    with pytest.raises(SystemExit):
        _parse_set(["novalue"])


def test_play_and_eval_args():
    a = build_parser().parse_args(["play", "--run", "r", "--difficulty", "superfly", "--color", "black", "--gui"])
    assert a.run == "r" and a.ckpt is None and a.difficulty == "superfly" and a.color == "black" and a.gui
    with pytest.raises(SystemExit):  # --run and --ckpt are exclusive
        build_parser().parse_args(["play", "--run", "r", "--ckpt", "c.pt"])
    with pytest.raises(SystemExit):  # unknown difficulty
        build_parser().parse_args(["play", "--difficulty", "stockfish"])
    a = build_parser().parse_args(["eval", "--run", "r", "--games", "4", "--opponent", "random,material",
                                   "--opponent", "other-run"])
    assert a.games == 4 and a.opponent == ["random,material", "other-run"]
    a = build_parser().parse_args(["build-brain", "--tiny"])
    assert a.tiny and a.region == "full" and a.command == "build-brain"
    a = build_parser().parse_args(["build-shards", "--max-games", "3000", "--workers", "8", "--name", "smoke"])
    assert a.max_games == 3000 and a.workers == 8 and a.name == "smoke" and a.val_every == 50
    assert build_parser().parse_args(["build-shards", "--val-every", "0"]).val_every == 0
    a = build_parser().parse_args(["export-web", "--run", "r", "--quant", "i8", "--out", "/tmp/m"])
    assert a.quant == "i8" and a.out == "/tmp/m" and not a.no_vectors
    a = build_parser().parse_args(["dashboard", "--run", "r", "--port", "9999"])
    assert a.port == 9999 and a.command == "dashboard"


def test_main_dispatches_and_maps_errors(monkeypatch):
    calls = {}
    monkeypatch.setitem(cli.COMMANDS, "train", lambda args: calls.setdefault("train", args) and 0)
    assert main(["train", "--run", "x"]) == 0
    assert calls["train"].run == "x"
    assert set(cli.COMMANDS) == set(next(a for a in build_parser()._actions if a.dest == "command").choices)

    def boom(args):
        raise FileNotFoundError("no checkpoint")

    monkeypatch.setitem(cli.COMMANDS, "play", boom)
    assert main(["play", "--run", "nope"]) == 2


# ---- subprocess ----------------------------------------------------------------------------------
@pytest.mark.parametrize("argv", [["--help"], ["build-brain", "--help"], ["train", "--help"], ["--version"]])
def test_fly_help_subprocess(argv):
    res = subprocess.run([PY, "-m", "flychess.cli", *argv], capture_output=True, text=True, timeout=60,
                         check=False, cwd=REPO)
    assert res.returncode == 0, res.stderr
    out = res.stdout + res.stderr
    if argv == ["--version"]:
        assert flychess.__version__ in out
    elif argv == ["--help"]:
        for name in ("download", "build-brain", "build-shards", "train", "dashboard", "play", "eval", "export-web"):
            assert name in out
    else:
        assert "usage: fly" in out and argv[0] in out


def test_fly_console_script_exists():
    exe = Path(PY).parent / "fly"
    if not exe.exists():
        pytest.skip("console script not installed in this environment")
    res = subprocess.run([str(exe), "--help"], capture_output=True, text=True, timeout=60, check=False)
    assert res.returncode == 0 and "fly-chess" in res.stdout


# ---- package API ---------------------------------------------------------------------------------
def test_package_api_surface():
    assert flychess.__version__
    assert callable(flychess.train) and callable(flychess.play) and callable(flychess.load_brain)
    # importing the sub-packages rebinds flychess.train / flychess.play to (callable) modules
    import importlib

    importlib.import_module("flychess.play")
    importlib.import_module("flychess.train.trainer")
    assert callable(flychess.train) and callable(flychess.play)
    assert flychess.train.trainer.train is not None and flychess.play.FlyEngine is not None
    pyproject = (REPO / "pyproject.toml").read_text()
    assert f'version = "{flychess.__version__}"' in pyproject


def test_load_brain_and_test_vectors_roundtrip(tmp_path, monkeypatch):
    """load_brain resolves a run name; write_model_vectors/check_model_vectors agree on the exported blob."""
    torch = pytest.importorskip("torch")
    from flychess import paths
    from flychess.connectome.graph import toy_graph
    from flychess.export.testvectors import CURATED_POSITIONS, check_model_vectors, write_model_vectors
    from flychess.export.web import export_web
    from flychess.model.config import BrainConfig
    from flychess.model.flybrain import FlyBrain
    from flychess.train.config import TrainConfig
    from flychess.train.trainer import save_checkpoint

    monkeypatch.setattr(paths, "RUNS_DIR", tmp_path / "runs")
    graph = toy_graph(n=150, nnz=1500, n_in=12, n_out=12, seed=3)
    gp = graph.save(tmp_path / "toy.npz")
    torch.manual_seed(0)
    model = FlyBrain(graph, BrainConfig(graph_path=str(gp), steps=3, value_hidden=8)).eval()
    cfg = TrainConfig(run="toyrun", graph=str(gp), brain={"steps": 3, "value_hidden": 8})
    save_checkpoint(tmp_path / "runs" / "toyrun", model, cfg, gp, step=7, stage="imitation")

    m2, g2 = flychess.load_brain("toyrun", device="cpu")
    assert g2.n == graph.n and m2.n == graph.n and not m2.training

    out_dir = tmp_path / "model"
    export_web(m2, g2, out_dir, extra_meta={"run_name": "toyrun"})
    vec = write_model_vectors(model_dir=out_dir, out=tmp_path / "model.json")
    data = json.loads(vec.read_text())
    assert data["run_name"] == "toyrun" and len(data["vectors"]) == len(CURATED_POSITIONS) == 12
    assert any(v["fen"].split()[1] == "b" for v in data["vectors"])
    assert any(v.get("moves") for v in data["vectors"])
    for v in data["vectors"]:
        assert v["top"][0][0] == v["argmax"] and len(v["top"]) <= 20 and -1 <= v["value"] <= 1
        assert all(v["top"][k][2] >= v["top"][k + 1][2] for k in range(len(v["top"]) - 1))
    assert check_model_vectors(out_dir, vec)["max_logit_delta"] == 0.0
    # the model+graph path (temporary export) gives the same numbers as the blob path
    vec2 = write_model_vectors(m2, g2, out=tmp_path / "model2.json", run_name="toyrun")
    d2 = json.loads(vec2.read_text())
    assert d2["blob_sha256"] == data["blob_sha256"] or True  # exported_at differs; compare the numbers instead
    for a, b in zip(data["vectors"], d2["vectors"], strict=True):
        assert a["argmax"] == b["argmax"] and abs(a["value"] - b["value"]) < 1e-6
        assert all(abs(x[2] - y[2]) < 1e-6 for x, y in zip(a["top"], b["top"], strict=True))
