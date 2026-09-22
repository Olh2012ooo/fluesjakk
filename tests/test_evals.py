"""Tests for flychess.data.evals: Lichess eval-database lines -> value/policy targets -> quantised shards."""
from __future__ import annotations

import json
import math
import zlib
from collections import Counter
from pathlib import Path

import chess
import numpy as np
import pytest
import zstandard
from torch.utils.data import DataLoader

from flychess.chessenv.encoding import encode_board, index_to_move, move_to_index
from flychess.data.evals import (
    VAL_EVERY,
    VALUE_SCALE,
    EvalBuildStats,
    best_eval,
    build_eval_shards,
    cp_to_value,
    decode_line,
    fen_ply,
    is_val_fen,
    iter_line_blocks,
    iter_lines,
    mate_to_value,
    open_eval_stream,
    positions_from_jsonl,
    process_chunk,
    quantize_value,
    score_to_value,
)
from flychess.data.shards import (
    ShardDataset,
    collate,
    count_positions,
    list_shards,
    list_val_shards,
    load_split,
    read_shard,
)

FIXTURE = Path(__file__).parent / "fixtures" / "evals_small.jsonl"
FIXTURE_LINES = 50

# hand-written positions: (fen, best uci, score dict, note)
_HAND = [
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -", "e2e4", {"cp": 30}, "start, white"),
    ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq -", "c7c5", {"cp": 30}, "1.e4, black to move"),
    ("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq -", "d2d3", {"cp": 300}, "+300 white"),
    ("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq -", "f8c5", {"cp": 300}, "+300 black"),
    ("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - -", "a1a8", {"mate": 1}, "mate in 1 for white (white to move)"),
    ("r5k1/5ppp/8/8/8/8/5PPP/6K1 b - -", "a8a1", {"mate": -1}, "black mates in 1 (black to move)"),
    ("6k1/5ppp/8/8/8/8/r5PP/6K1 w - -", "h2h3", {"mate": -3}, "white is getting mated"),
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "g1f3", {"cp": 20}, "6-field FEN, ply 0"),
    ("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", "g1f3", {"cp": 20}, "6-field FEN, ply 2"),
    ("rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2", "b8c6", {"cp": 20}, "6-field FEN, ply 3"),
    ("8/P7/8/8/8/8/k6K/8 w - -", "a7a8n", {"cp": 500}, "under-promotion to knight"),
    ("8/P7/8/8/8/8/k6K/8 w - -", "a7a8q", {"cp": 900}, "queen promotion"),
    ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq -", "e1g1", {"cp": 0}, "castling"),
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -", "e2e5", {"cp": 30}, "ILLEGAL pv move"),
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -", "zz99", {"cp": 30}, "UNPARSEABLE pv move"),
    ("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - -", "a1a8", {"cp": 30}, "SHALLOW (depth 12)"),
    ("7k/5Q2/6K1/8/8/8/8/8 b - -", "h8g8", {"mate": 0}, "black is CHECKMATED (no legal move)"),
]


def _line(fen: str, uci: str, score: dict, depth: int = 30, extra_evals: int = 0) -> str:
    evals = [{"pvs": [{**score, "line": f"{uci} a2a3 h7h6"}], "knodes": 1000 * depth, "depth": depth}]
    for i in range(extra_evals):  # shallower evals with a *different* first move must be ignored
        evals.insert(0, {"pvs": [{"cp": -999, "line": "h2h3"}], "knodes": 10, "depth": depth - 5 - i})
    return json.dumps({"fen": fen, "evals": evals})


def make_fixture(path: Path = FIXTURE, n: int = FIXTURE_LINES) -> Path:
    """Write the synthetic 50-line jsonl fixture (hand cases + generated positions from a few games)."""
    lines = []
    for i, (fen, uci, score, note) in enumerate(_HAND):
        depth = 12 if "SHALLOW" in note else 30 + i
        lines.append(_line(fen, uci, score, depth=depth, extra_evals=i % 3))
    # fill with positions along random games (legal first pv move = a random legal move, random scores)
    rng = np.random.default_rng(0)
    board = chess.Board()
    while len(lines) < n:
        if board.is_game_over() or board.fullmove_number > 40:
            board = chess.Board()
        moves = list(board.legal_moves)
        move = moves[rng.integers(len(moves))]
        fen = " ".join(board.fen().split()[:4])  # the real dump has 4-field FENs
        score = {"mate": int(rng.integers(1, 6)) * (1 if rng.random() < 0.5 else -1)} if rng.random() < 0.1 \
            else {"cp": int(rng.integers(-800, 800))}
        lines.append(_line(fen, move.uci(), score, depth=int(rng.integers(20, 60)), extra_evals=int(rng.integers(3))))
        board.push(move)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture(scope="module")
def fixture_path() -> Path:
    if not FIXTURE.exists():
        make_fixture()
    return FIXTURE


# ---------------------------------------------------------------------------------------------
# value convention
# ---------------------------------------------------------------------------------------------
def test_value_curve_and_sign_convention():
    assert cp_to_value(0) == 0.0
    assert abs(cp_to_value(300) - 0.5) < 0.01                  # the Lichess curve: +300 cp ~ 75 % win
    assert abs(cp_to_value(100) - 0.18) < 0.01
    assert cp_to_value(-300) == pytest.approx(-cp_to_value(300))
    assert 0.99 < cp_to_value(1500) < 1.0 and cp_to_value(10**9) <= 1.0 and cp_to_value(-(10**9)) >= -1.0
    assert mate_to_value(1) == 1.0 == -mate_to_value(-5) and mate_to_value(0) == -1.0
    # +300 cp (White's point of view) is good for the mover when White moves ...
    v_w, mate_w = score_to_value({"cp": 300}, white_to_move=True)
    assert v_w > 0 and not mate_w
    # ... and bad for the mover when Black moves (same score)
    v_b, _ = score_to_value({"cp": 300}, white_to_move=False)
    assert v_b < 0 and v_b == -v_w
    assert score_to_value({"mate": 2}, True) == (1.0, True) and score_to_value({"mate": 2}, False) == (-1.0, True)
    assert score_to_value({"mate": -2}, False) == (1.0, True)  # black to move and mates
    assert quantize_value(1.0) == 127 and quantize_value(-1.0) == -127 and quantize_value(0.0) == 0
    assert quantize_value(cp_to_value(300)) == round(cp_to_value(300) * 127) == 64
    assert quantize_value(5.0) == 127 and quantize_value(-5.0) == -127


def test_decode_hand_lines():
    lines = [_line(fen, uci, score, depth=30) for fen, uci, score, _ in _HAND]
    notes = [h[3] for h in _HAND]
    decoded = {}
    for line, note in zip(lines, notes):
        skip = Counter()
        d = decode_line(line, min_depth=20, skip=skip)
        if d is None:
            decoded[note] = dict(skip)
        else:
            decoded[note] = d
    # sign convention by hand
    assert decoded["+300 white"].value > 0 > decoded["+300 black"].value
    assert decoded["+300 white"].value == pytest.approx(-decoded["+300 black"].value)
    assert decoded["mate in 1 for white (white to move)"].value == 1.0
    assert decoded["black mates in 1 (black to move)"].value == 1.0     # mover (black) wins
    assert decoded["white is getting mated"].value == -1.0
    assert decoded["1.e4, black to move"].value == pytest.approx(-cp_to_value(30))
    assert all(decoded[n].is_mate for n in notes if "mate" in n and isinstance(decoded[n], tuple))
    # move indices decode back to the pv move in the position (incl. under-promotion / castling / black)
    for note in ("start, white", "1.e4, black to move", "under-promotion to knight", "queen promotion", "castling",
                 "black mates in 1 (black to move)"):
        fen, uci = next((h[0], h[1]) for h in _HAND if h[3] == note)
        d = decoded[note]
        assert index_to_move(d.move, chess.Board(fen)).uci() == uci and d.uci == uci
        assert d.move == move_to_index(chess.Move.from_uci(uci), chess.Board(fen))
    assert decoded["under-promotion to knight"].move >= 4096 > decoded["queen promotion"].move
    # ply from the FEN's fullmove number when present, else 0 / 1 by side to move
    assert decoded["6-field FEN, ply 0"].ply == 0 and decoded["6-field FEN, ply 2"].ply == 2
    assert decoded["6-field FEN, ply 3"].ply == 3
    assert decoded["start, white"].ply == 0 and decoded["1.e4, black to move"].ply == 1
    # skipped lines with a reason
    assert decoded["ILLEGAL pv move"] == {"illegal_move": 1}
    assert decoded["UNPARSEABLE pv move"] == {"bad_move": 1}
    assert decoded["black is CHECKMATED (no legal move)"] == {"no_legal_moves": 1}
    assert decode_line(_line(_HAND[0][0], "e2e4", {"cp": 1}, depth=12), 20, s := Counter()) is None
    assert s == {"shallow": 1}
    assert decode_line("not json", 20, s := Counter()) is None and s == {"bad_json": 1}
    assert decode_line(json.dumps({"fen": _HAND[0][0], "evals": []}), 20, s := Counter()) is None
    assert s == {"no_pv": 1}
    # the deepest eval wins, whatever its position in the list
    line = _line(_HAND[2][0], "d2d3", {"cp": 300}, depth=30, extra_evals=2)
    d = decode_line(line, 20)
    assert d.depth == 30 and d.uci == "d2d3" and d.value > 0
    assert best_eval(json.loads(line)["evals"])["depth"] == 30
    assert best_eval([{"pvs": [], "depth": 99}, {"pvs": [{"cp": 1, "line": ""}], "depth": 98}]) is None
    # the planes are the reference encoding with plane 18 (halfmove clock) = 0 for 4-field FENs
    d = decoded["+300 black"]
    from flychess.chessenv.encoding import planes_from_features
    planes = planes_from_features(d.feats[None], quantized=True)[0]
    ref = encode_board(chess.Board(_HAND[3][0]))
    assert np.array_equal(planes[:18], ref[:18].astype(np.uint8)) and planes[19].max() == 0 and planes[18].max() == 0
    assert planes[17].min() == 1
    assert fen_ply(chess.Board("8/8/8/8/8/8/8/K6k b - - 5 40"), "8/8/8/8/8/8/8/K6k b - - 5 40") == 79


# ---------------------------------------------------------------------------------------------
# fixture -> generator / worker / shards
# ---------------------------------------------------------------------------------------------
def test_fixture_and_generator(fixture_path):
    raw = fixture_path.read_text().splitlines()
    assert len(raw) == FIXTURE_LINES
    positions = list(positions_from_jsonl(fixture_path, min_depth=20))
    # 4 of the hand lines are skipped (illegal / unparseable / shallow / checkmated)
    assert len(positions) == FIXTURE_LINES - 4
    assert any(p.is_mate for p in positions) and any(not p.is_mate for p in positions)
    assert any(p.fen.split()[1] == "b" for p in positions)
    assert all(p.elo == 0 and p.value == quantize_value(p.value_float) for p in positions)
    assert all(-127 <= p.value <= 127 and p.depth >= 20 for p in positions)
    for p in positions:
        board = chess.Board(p.fen)
        assert index_to_move(p.move, board).uci() == p.uci and board.is_legal(chess.Move.from_uci(p.uci))
        assert np.array_equal(p.planes[:18], encode_board(board)[:18].astype(np.uint8))
        j = json.loads(next(l for l in raw if json.loads(l)["fen"] == p.fen and p.uci in l))
        ev = max(j["evals"], key=lambda e: e["depth"])
        pv = ev["pvs"][0]
        white = p.fen.split()[1] == "w"
        want = (1.0 if pv["mate"] > 0 else -1.0) if "mate" in pv else cp_to_value(pv["cp"])
        assert p.value_float == pytest.approx(want if white else -want) and p.depth == ev["depth"]
    # same via the generic byte-line iterator and on an in-memory list of lines
    assert [p.fen for p in positions_from_jsonl(list(iter_lines(fixture_path)))] == [p.fen for p in positions]
    # a .zst copy streams identically (chunk boundaries inside lines)
    zpath = fixture_path.with_suffix(".jsonl.zst")
    try:
        zpath.write_bytes(zstandard.ZstdCompressor().compress(fixture_path.read_bytes()))
        assert list(iter_lines(zpath, chunk_bytes=100)) == list(iter_lines(fixture_path))
        with open_eval_stream(zpath) as st:
            blocks = list(iter_line_blocks(st, chunk_bytes=333))
        assert b"".join(blocks) == fixture_path.read_bytes() and all(b.endswith(b"\n") for b in blocks)
        assert [p.fen for p in positions_from_jsonl(zpath)] == [p.fen for p in positions]
    finally:
        zpath.unlink(missing_ok=True)


def test_process_chunk_matches_generator(fixture_path):
    chunk = process_chunk(fixture_path.read_bytes(), min_depth=20, val_every=VAL_EVERY)
    positions = list(positions_from_jsonl(fixture_path))
    assert chunk.lines == FIXTURE_LINES and len(chunk) == len(positions)
    assert chunk.skip == {"illegal_move": 1, "bad_move": 1, "shallow": 1, "no_legal_moves": 1}
    assert chunk.move.tolist() == [p.move for p in positions] and chunk.value.tolist() == [p.value for p in positions]
    assert chunk.depth.tolist() == [p.depth for p in positions] and chunk.mate.tolist() == [p.is_mate for p in positions]
    assert chunk.is_val.tolist() == [is_val_fen(p.fen) for p in positions]
    assert chunk.value.dtype == np.int8 and chunk.feats.dtype == np.uint64 and chunk.feats.shape == (len(positions), 19)
    head = chunk.head(5)
    assert len(head) == 5 and head.lines == chunk.lines
    train, val = chunk.split()
    assert len(train.move) + len(val.move) == len(positions) and (train.elo == 0).all()
    # the val split is a stable hash of the FEN, 1 in VAL_EVERY, independent of everything else
    assert is_val_fen("x") == (zlib.crc32(b"x") % VAL_EVERY == 0) and not is_val_fen("x", 0)
    fens = [f"8/8/8/8/8/8/8/K6k w - - 0 {i}" for i in range(5000)]
    frac = sum(is_val_fen(f) for f in fens) / len(fens)
    assert 0.012 < frac < 0.03
    # an empty / blank block is fine
    empty = process_chunk(b"\n\n", 20)
    assert len(empty) == 0 and empty.lines == 0 and empty.split()[0].feats.shape == (0, 19)


def test_build_eval_shards_and_value_scale_roundtrip(fixture_path, tmp_path):
    # replicate the fixture so that the build has to shuffle and cut several shards
    big = tmp_path / "big.jsonl"
    text = fixture_path.read_text()
    big.write_text(text * 40)
    stats = build_eval_shards(big, tmp_path / "out", name="ev", max_positions=None, min_depth=20, workers=2,
                              val_every=4, shard_size=300, shuffle_buffer=500, chunk_bytes=5000, progress=False)
    positions = list(positions_from_jsonl(fixture_path)) * 40
    assert isinstance(stats, EvalBuildStats)
    assert stats.lines == FIXTURE_LINES * 40 and stats.kept == len(positions)
    assert stats.skipped == {"illegal_move": 40, "bad_move": 40, "shallow": 40, "no_legal_moves": 40}
    assert stats.positions_val > 0 and stats.val_shards >= 1 and stats.positions + stats.positions_val == len(positions)
    assert stats.mates == sum(p.is_mate for p in positions) and stats.value_scale == VALUE_SCALE == 127
    assert sum(stats.depth_hist.values()) == len(positions) == sum(stats.value_hist)
    assert stats.depth_hist == Counter(p.depth for p in positions)
    assert "kept" in stats.summary() and "%" in stats.depth_summary() and "[-1.0,-0.9)" in stats.value_summary()
    assert stats.to_dict()["kept"] == len(positions)
    train_files, val_files = list_shards(tmp_path / "out", "ev"), list_val_shards(tmp_path / "out", "ev")
    assert len(train_files) == stats.shards == -(-stats.positions // 300) and len(val_files) == stats.val_shards
    assert [str(f) for f in train_files] == stats.shard_files and [str(f) for f in val_files] == stats.val_shard_files
    assert count_positions(train_files) == stats.positions and count_positions(val_files) == stats.positions_val
    # every shard carries value_scale = 127 and the exact multiset of examples is preserved
    got = Counter()
    for f in train_files + val_files:
        s = read_shard(f)
        assert s["value_scale"] == 127.0 and s["value"].dtype == np.int8 and (s["elo"] == 0).all()
        with np.load(f) as z:
            assert "value_scale" in z.files and float(z["value_scale"]) == 127.0
        for i in range(len(s["move"])):
            got[(s["planes"][i].tobytes(), int(s["move"][i]), int(s["value"][i]), int(s["ply"][i]))] += 1
    want = Counter((p.planes.tobytes(), p.move, p.value, p.ply) for p in positions)
    assert got == want
    # the validation series is exactly the FEN-hash split (deterministic: the same FENs on every rebuild)
    got_val = Counter()
    for f in val_files:
        s = read_shard(f)
        for i in range(len(s["move"])):
            got_val[(s["planes"][i].tobytes(), int(s["move"][i]), int(s["value"][i]), int(s["ply"][i]))] += 1
    val_want = Counter((p.planes.tobytes(), p.move, p.value, p.ply) for p in positions if is_val_fen(p.fen, 4))
    assert got_val and got_val == val_want
    # load_split picks the val series; collate returns value = int8 / 127 as float32
    tr, va = load_split(tmp_path / "out", name="ev")
    assert (tr, va) == (train_files, val_files)
    loader = DataLoader(ShardDataset(tr, shuffle=False), batch_size=64, collate_fn=collate)
    seen = 0
    for batch in loader:
        v = batch["value"]
        assert v.dtype.is_floating_point and (v.abs() <= 1.0).all()
        assert set((v * 127).round().to(int).tolist()) <= set(range(-127, 128))
        assert (batch["elo"] == 0).all()
        seen += len(v)
    assert seen == stats.positions
    items = list(ShardDataset(tr[:1], shuffle=False))
    assert items[0]["value_scale"] == 127.0 and isinstance(items[0]["value"], int)
    b = collate(items[:3])
    assert b["value"].tolist() == pytest.approx([it["value"] / 127 for it in items[:3]])
    # value_float of the generator round-trips through the int8 storage within half a quantum
    for p in positions[:20]:
        assert abs(p.value / 127 - p.value_float) <= 0.5 / 127 + 1e-9
    # max_positions caps exactly, val_every=0 writes no val series
    stats2 = build_eval_shards(big, tmp_path / "cap", name="ev", max_positions=123, workers=2, val_every=0,
                               shard_size=100, shuffle_buffer=100, chunk_bytes=5000, progress=False)
    assert stats2.kept == stats2.positions == 123 and stats2.val_shards == 0 and stats2.positions_val == 0
    assert count_positions(tmp_path / "cap") == 123 and stats2.shards == 2
    assert sum(stats2.value_hist) == 123 == sum(stats2.depth_hist.values())
    # min_depth filter
    stats3 = build_eval_shards(fixture_path, tmp_path / "deep", name="ev", max_positions=None, min_depth=40,
                               workers=1, progress=False)
    assert stats3.kept == sum(p.depth >= 40 for p in positions[:len(positions) // 40]) > 0
    assert stats3.skipped["shallow"] > 1


def test_win_probability_matches_lichess_reference():
    # values published on lichess.org/page/accuracy: win% = 50 + 50 * (2 / (1 + exp(-0.00368208 * cp)) - 1)
    for cp, win in ((0, 50.0), (100, 59.1), (200, 67.6), (500, 86.3), (-100, 40.9)):
        assert 50 + 50 * cp_to_value(cp) == pytest.approx(win, abs=0.1)
    assert cp_to_value(400) == pytest.approx(2 / (1 + math.exp(-0.00368208 * 400)) - 1)
