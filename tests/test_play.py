"""Tests for flychess.play: FlyEngine difficulties, the terminal UI (scripted) and the local web server."""
from __future__ import annotations

import io
import json
import urllib.request
from pathlib import Path

import chess
import numpy as np
import pytest
import torch

from flychess.connectome.graph import BrainGraph, toy_graph
from flychess.model import BrainConfig, FlyBrain
from flychess.play import DIFFICULTIES, FlyEngine
from flychess.play.engine import board_with_history, load_checkpoint, masked_probs, mood_from_value
from flychess.play.local_web import MIME_TYPES, export_is_stale, serve_local
from flychess.play.terminal import commentary, parse_move, play_terminal, render_board

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TINY = Path(__file__).resolve().parent.parent / "data" / "brain" / "tiny.npz"


@pytest.fixture(scope="module")
def graph_and_path(tmp_path_factory):
    if TINY.exists():
        return BrainGraph.load(TINY), TINY
    g = toy_graph(n=300, nnz=3000, n_in=32, n_out=32, seed=3)
    p = tmp_path_factory.mktemp("graph") / "toy.npz"
    g.save(p)
    return g, p


@pytest.fixture(scope="module")
def model(graph_and_path):
    graph, path = graph_and_path
    torch.manual_seed(0)
    return FlyBrain(graph, BrainConfig(graph_path=str(path), steps=4)).to(DEVICE).eval()


@pytest.fixture(scope="module")
def engine(model, graph_and_path):
    return FlyEngine(model, graph_and_path[0], DEVICE, sims=8, seed=0)


@pytest.fixture(scope="module")
def checkpoint(model, graph_and_path, tmp_path_factory):
    """A checkpoint in the SPEC §6 format inside a fake run directory."""
    run_dir = tmp_path_factory.mktemp("runs") / "smoke"
    run_dir.mkdir()
    ck = {"config": {"run": "smoke", "graph": "tiny"}, "brain_config": model.config.to_dict(),
          "graph_path": str(graph_and_path[1]), "model": model.state_dict(), "optim": None, "step": 42,
          "stage": "imitation", "elo": {"random": 1000.0}}
    torch.save(ck, run_dir / "ckpt-42.pt")
    torch.save(ck, run_dir / "latest.pt")
    return run_dir / "latest.pt"


# ---- engine ---------------------------------------------------------------------------------------
def test_mood_and_helpers():
    assert mood_from_value(0.9) == "smug" and mood_from_value(0.5) == "confident"
    assert mood_from_value(0.0) == "focused" and mood_from_value(-0.5) == "nervous" and mood_from_value(-0.9) == "panicking"
    p = masked_probs(np.arange(10, dtype=np.float32), np.array([1, 3]), temperature=1.0)
    assert p.shape == (2,) and abs(p.sum() - 1) < 1e-9 and p[1] > p[0]
    flat = masked_probs(np.arange(10, dtype=np.float32), np.array([1, 3]), temperature=100.0)
    assert flat[1] - flat[0] < p[1] - p[0]
    b = chess.Board()
    b.push_san("e4")
    rebuilt = board_with_history(chess.Board(b.fen()), ["e2e4"])
    assert rebuilt.move_stack == b.move_stack
    with pytest.raises(ValueError):
        board_with_history(chess.Board(), ["e2e4"])


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_engine_returns_legal_moves(engine, difficulty):
    boards = [chess.Board(), chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"),
              chess.Board("8/8/8/8/8/2k5/8/K2q4 w - - 0 1")]  # last: only a few king moves
    for board in boards:
        move, info = engine.choose_move(board, difficulty)
        assert move in board.legal_moves
        assert info["difficulty"] == difficulty and info["mood"] in ("smug", "confident", "focused", "nervous", "panicking")
        assert 1 <= len(info["policy_top"]) <= 5 and all(chess.Move.from_uci(u) in board.legal_moves for u, _ in info["policy_top"])
        assert abs(sum(p for _, p in info["policy_top"])) <= 1.0 + 1e-6
        assert -1.0 <= info["value"] <= 1.0 and info["think_ms"] >= 0
        assert info["sims"] == (8 if difficulty == "superfly" else 0)
        if difficulty == "superfly":
            assert info["search_top"] and chess.Move.from_uci(info["search_top"][0][0]) == move
        if difficulty == "fly":
            assert len(info["candidates"]) <= 3
    with pytest.raises(ValueError):
        engine.choose_move(chess.Board(), "grandmaster")
    with pytest.raises(ValueError):
        engine.choose_move(chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1"), "fly")  # game over


def test_fly_difficulty_takes_mate_when_it_is_a_candidate(model, graph_and_path):
    """The 1-ply value check sees a mate as -1 for the opponent — whatever the (random) policy thinks."""
    e = FlyEngine(model, graph_and_path[0], DEVICE, sims=4, seed=0)
    board = chess.Board("7k/8/6K1/8/8/8/8/5Q2 w - - 0 1")  # many mates in one (Qf8#, Qh3#? no: Qf8# and Qa8#...)
    mates = [m for m in board.legal_moves if board.gives_check(m) and _is_mate(board, m)]
    assert mates
    # force the mating moves into the candidate list by checking the value-check directly
    move, ranked = e._value_check(board, mates[:1] + [m for m in board.legal_moves if m not in mates][:2])
    assert move == mates[0] and ranked[0][1] == -1.0


def _is_mate(board: chess.Board, move: chess.Move) -> bool:
    b = board.copy()
    b.push(move)
    return b.is_checkmate()


def test_as_player_protocol(engine):
    p = engine.as_player("fly")
    assert p.name == "fly-fly"
    b = chess.Board()
    mv = p.choose(b)
    assert mv in b.legal_moves and p.last_info["difficulty"] == "fly"
    assert engine.as_player("superfly").name == "fly-superfly"
    with pytest.raises(ValueError):
        engine.as_player("nope")


def test_load_checkpoint_and_engine_load(checkpoint, model):
    m, g, ck = load_checkpoint(checkpoint, DEVICE)
    assert ck["step"] == 42 and g.n == model.n and not m.training
    for k, v in model.state_dict().items():
        assert torch.equal(v.cpu(), m.state_dict()[k].cpu())
    e = FlyEngine.load(checkpoint.parent, DEVICE, sims=4)  # a run directory resolves to latest.pt
    mv, _ = e.choose_move(chess.Board(), "superfly")
    assert mv in chess.Board().legal_moves
    with pytest.raises(FileNotFoundError):
        load_checkpoint(checkpoint.parent / "missing.pt", DEVICE)


# ---- terminal -------------------------------------------------------------------------------------
def test_render_and_parse():
    b = chess.Board()
    txt = render_board(b).plain
    assert "♔" in txt and "♚" in txt and txt.strip().endswith("a  b  c  d  e  f  g  h")
    assert render_board(b, flip=True).plain.strip().endswith("h  g  f  e  d  c  b  a")
    assert parse_move(b, "e4") == chess.Move.from_uci("e2e4") and parse_move(b, "E2E4") == chess.Move.from_uci("e2e4")
    assert parse_move(b, "Nf3") == chess.Move.from_uci("g1f3") and parse_move(b, "e5") is None and parse_move(b, "xx") is None
    promo = chess.Board("8/P6k/8/8/8/8/8/K7 w - - 0 1")
    assert parse_move(promo, "a7a8") == chess.Move.from_uci("a7a8q")
    line = commentary("Nf3", {"move_prob": 0.87, "mood": "confident", "sims": 200, "think_ms": 12.0})
    assert line.startswith("🪰 the fly plays Nf3 (87% sure, feels confident)") and "200 simulations" in line


def test_play_terminal_scripted(engine):
    out = io.StringIO()
    res = play_terminal(engine=engine, difficulty="fly", color="white", out=out,
                        inputs=["e4", "undo", "hint", "e2e4", "Nz9", "Nf3", "resign", "new", "d4", "quit"])
    text = out.getvalue()
    assert "you play e4" in text and "🪰 the fly plays" in text and "took back 2 plies" in text
    assert "hint — the fly's own top policy moves" in text
    assert "not a legal move or command" in text
    assert "you resigned" in text and "[Result \"0-1\"]" in text
    assert "new game" in text and "game abandoned" in text
    assert res["result"] == "*" and res["plies"] == 2 and len(res["games"]) == 2
    assert res["games"][0]["result"] == "0-1" and res["games"][0]["plies"] == 4 and "1. e4" in res["games"][0]["pgn"]


def test_play_terminal_black_and_superfly(engine):
    out = io.StringIO()
    res = play_terminal(engine=engine, difficulty="superfly", color="black", out=out, inputs=["e5", "resign"])
    text = out.getvalue()
    assert "you play black" in text and "after 8 simulations" in text
    assert res["result"] == "1-0" and res["plies"] == 3
    assert "[White \"fly-superfly\"]" in res["pgn"] and "[Black \"human\"]" in res["pgn"]


def test_play_terminal_ends_with_input_exhaustion(engine):
    out = io.StringIO()
    res = play_terminal(engine=engine, out=out, inputs=iter(["Nf3"]))
    assert res["result"] == "*" and res["plies"] == 2
    with pytest.raises(ValueError):
        play_terminal(engine=engine, difficulty="boss", out=out, inputs=[])


# ---- local web ------------------------------------------------------------------------------------
def test_serve_local_exports_and_serves(checkpoint, tmp_path):
    web = tmp_path / "web"
    web.mkdir()
    (web / "index.html").write_text("<!doctype html><title>fly</title>")
    (web / "app.mjs").write_text("export const x = 1;")
    out_dir = web / "model"
    assert export_is_stale(checkpoint, out_dir)
    server, url = serve_local(checkpoint, port=0, out_dir=out_dir, open_browser=False, web_dir=web, block=False,
                              quiet=True)
    try:
        assert (out_dir / "brain.json").exists() and (out_dir / "brain.flyb.gz").exists()
        assert not export_is_stale(checkpoint, out_dir)
        r = urllib.request.urlopen(url + "model/brain.json", timeout=10)
        header = json.loads(r.read())
        assert r.headers["Content-Type"].startswith("application/json")
        assert header["run_name"] == "smoke" and header["train_steps"] == 42 and header["elo_estimates"] == {"random": 1000.0}
        r = urllib.request.urlopen(url + "model/brain.flyb.gz", timeout=10)
        assert r.headers["Content-Type"] == MIME_TYPES[".gz"] and r.headers.get("Content-Encoding") is None
        assert r.read()[:2] == b"\x1f\x8b"  # raw gzip bytes, decompressed by the JS loader
        r = urllib.request.urlopen(url + "app.mjs", timeout=10)
        assert r.headers["Content-Type"].startswith("text/javascript")
        assert urllib.request.urlopen(url, timeout=10).headers["Content-Type"].startswith("text/html")
    finally:
        server.shutdown()
        server.server_close()
    # a newer checkpoint makes the export stale again
    import os
    import time

    now = time.time() + 5
    os.utime(checkpoint, (now, now))
    assert export_is_stale(checkpoint, out_dir)
