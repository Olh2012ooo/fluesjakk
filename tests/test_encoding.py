"""Tests for flychess.chessenv: encoding (SPEC §3), ChessEnv and the cross-language vectors."""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import chess
import numpy as np
import pytest

from flychess.chessenv import ChessEnv
from flychess.chessenv.encoding import (
    FLAT_INPUT,
    NUM_MOVES,
    NUM_PLANES,
    board_features,
    encode_board,
    encode_boards,
    flatten_planes,
    index_to_move,
    legal_move_indices,
    legal_move_mask,
    move_to_index,
    planes_from_features,
    position_repeated,
    transposition_key,
)
from flychess.chessenv.vectors import DEFAULT_PATH, build_vectors, planes_sha256

VECTORS_PATH = Path(__file__).parent / "vectors" / "encoding.json"


def random_playout_boards(n: int, seed: int = 0, max_plies: int = 200) -> list[chess.Board]:
    """``n`` boards (with move stacks) sampled from random playouts; biased to captures/promotions."""
    rng = random.Random(seed)
    boards = []
    while len(boards) < n:
        board = chess.Board()
        for _ in range(rng.randint(1, max_plies)):
            moves = list(board.legal_moves)
            if not moves:
                break
            special = [m for m in moves if m.promotion or board.is_capture(m)]
            board.push(rng.choice(special) if special and rng.random() < 0.4 else rng.choice(moves))
        boards.append(board)
    return boards


# ---------------------------------------------------------------------------------------------
# constants & planes
# ---------------------------------------------------------------------------------------------
def test_constants():
    assert NUM_PLANES == 20 and FLAT_INPUT == 1280 and NUM_MOVES == 4168


def test_start_position_planes():
    p = encode_board(chess.Board())
    assert p.shape == (20, 8, 8) and p.dtype == np.float32
    assert p[0, 1].sum() == 8 and p[0].sum() == 8          # mover pawns on rank 2
    assert p[6, 6].sum() == 8 and p[6].sum() == 8          # opponent pawns on rank 7
    assert p[5, 0, 4] == 1 and p[11, 7, 4] == 1            # kings on e1 / e8
    assert p[12:16].sum() == 4 * 64 and p[17].sum() == 64  # castling all ones, bias plane
    assert p[16].sum() == 0 and p[18].sum() == 0 and p[19].sum() == 0
    assert flatten_planes(p).shape == (1280,)
    assert np.array_equal(flatten_planes(p), p.reshape(-1))


def test_black_to_move_is_mirrored():
    board = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
    p = encode_board(board)
    assert p[0, 1].sum() == 8                       # black pawns become the mover's, on rank "2"
    assert p[6, 4, 4] == 1 and p[6].sum() == 8      # white e4 pawn shows up on e5 from black's view
    assert p[5, 0, 4] == 1                          # mover king on e1
    assert np.array_equal(p, encode_board(board.mirror()))


def test_castling_planes_asymmetric():
    p = encode_board(chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w Kq - 0 1"))
    assert p[12].all() and not p[13].any() and not p[14].any() and p[15].all()
    p = encode_board(chess.Board("r3k2r/8/8/8/8/8/8/R3K2R b Kq - 0 1"))
    assert not p[12].any() and p[13].all() and p[14].all() and not p[15].any()


def test_en_passant_plane_only_when_legal():
    p = encode_board(chess.Board("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3"))
    assert p[16].sum() == 1 and p[16, 5, 3] == 1
    p = encode_board(chess.Board("rnbqkbnr/pppp1ppp/8/8/3Pp3/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 3"))
    assert p[16].sum() == 1 and p[16, 5, 3] == 1    # d3 mirrored to d6
    # ep square set but the capture would expose the king along the 5th rank
    p = encode_board(chess.Board("8/8/8/K1pP3r/8/8/8/6k1 w - c6 0 1"))
    assert p[16].sum() == 0


def test_halfmove_and_repetition_planes():
    assert encode_board(chess.Board("8/8/8/4k3/8/8/4K3/7R w - - 57 100"))[18, 3, 3] == np.float32(0.57)
    assert encode_board(chess.Board("8/8/8/4k3/8/8/4K3/7R b - - 120 80"))[18].min() == 1.0
    board = chess.Board()
    for uci in ["g1f3", "g8f6", "f3g1"]:
        board.push_uci(uci)
    assert encode_board(board)[19].sum() == 0 and not position_repeated(board)
    board.push_uci("f6g8")
    assert encode_board(board)[19].all() and position_repeated(board)
    # explicit override
    assert encode_board(board, repeated=False)[19].sum() == 0
    # castling rights lost -> same pieces but not a repetition
    board = chess.Board()
    for uci in ["e2e4", "e7e5", "e1e2", "e8e7", "e2e1", "e7e8"]:
        board.push_uci(uci)
    assert not position_repeated(board)


def test_transposition_key_matches_is_repetition():
    for board in random_playout_boards(20, seed=7, max_plies=300):
        replay = chess.Board()
        seen: Counter = Counter()
        for move in board.move_stack:
            seen[transposition_key(replay)] += 1
            replay.push(move)
            assert (seen[transposition_key(replay)] > 0) == replay.is_repetition(2)


def test_planes_quantized_storage():
    feats = np.stack([board_features(chess.Board("8/8/8/4k3/8/8/4K3/7R w - - 57 100")),
                      board_features(chess.Board())])
    q = planes_from_features(feats, quantized=True)
    f = planes_from_features(feats)
    assert q.dtype == np.uint8 and q.shape == (2, 20, 8, 8)
    assert q[0, 18, 0, 0] == (57 * 255 + 50) // 100 == 145 and q[1, 18, 0, 0] == 0
    assert np.array_equal(q[:, :18], f[:, :18]) and np.array_equal(q[:, 19], f[:, 19])


# ---------------------------------------------------------------------------------------------
# moves
# ---------------------------------------------------------------------------------------------
def test_specific_move_indices():
    board = chess.Board("r3k2r/pppqbppp/2npbn2/4p3/4P3/2NPBN2/PPPQBPPP/R3K2R w KQkq - 4 8")
    assert move_to_index(chess.Move.from_uci("e1g1"), board) == 4 * 64 + 6
    assert move_to_index(chess.Move.from_uci("e1c1"), board) == 4 * 64 + 2
    black = board.copy()
    black.turn = chess.BLACK
    assert move_to_index(chess.Move.from_uci("e8g8"), black) == 4 * 64 + 6  # mirrored == white castling
    assert index_to_move(4 * 64 + 6, black) == chess.Move.from_uci("e8g8")

    promo = chess.Board("1n6/P7/8/8/8/8/8/k6K w - - 0 1")
    assert move_to_index(chess.Move.from_uci("a7a8q"), promo) == 48 * 64 + 56
    assert move_to_index(chess.Move.from_uci("a7a8n"), promo) == 4096 + (0 * 3 + 1) * 3 + 0
    assert move_to_index(chess.Move.from_uci("a7a8b"), promo) == 4096 + (0 * 3 + 1) * 3 + 1
    assert move_to_index(chess.Move.from_uci("a7b8r"), promo) == 4096 + (0 * 3 + 2) * 3 + 2
    assert index_to_move(48 * 64 + 56, promo) == chess.Move.from_uci("a7a8q")
    bpromo = chess.Board("k6K/8/8/8/8/8/p7/1N6 b - - 0 1")
    assert move_to_index(chess.Move.from_uci("a2a1n"), bpromo) == 4099
    assert move_to_index(chess.Move.from_uci("a2b1q"), bpromo) == 48 * 64 + 57
    assert index_to_move(4099, bpromo) == chess.Move.from_uci("a2a1n")
    # a non-pawn arriving on the last rank is not a promotion
    rook = chess.Board("4k3/8/8/8/8/8/R7/4K3 w - - 0 1")
    assert index_to_move(move_to_index(chess.Move.from_uci("a2a8"), rook), rook) == chess.Move.from_uci("a2a8")
    with pytest.raises(ValueError):
        index_to_move(NUM_MOVES, rook)
    with pytest.raises(ValueError):
        move_to_index(chess.Move(chess.E2, chess.E4, promotion=chess.KNIGHT), rook)


def test_legal_mask_matches_indices():
    for board in random_playout_boards(30, seed=3):
        mask = legal_move_mask(board)
        idx = legal_move_indices(board)
        assert mask.dtype == bool and mask.shape == (NUM_MOVES,)
        assert np.array_equal(np.nonzero(mask)[0], idx)
        assert len(idx) == board.legal_moves.count()


def test_property_round_trip_and_mirror_500_positions():
    n_black = n_promo = 0
    for board in random_playout_boards(500, seed=42):
        seen = set()
        for move in board.legal_moves:
            idx = move_to_index(move, board)
            assert 0 <= idx < NUM_MOVES
            assert idx not in seen
            seen.add(idx)
            assert index_to_move(idx, board) == move
            n_promo += move.promotion is not None
        fen_board = chess.Board(board.fen())
        assert np.array_equal(encode_board(fen_board), encode_board(fen_board.mirror()))
        n_black += board.turn == chess.BLACK
    assert n_black > 100 and n_promo > 0


def test_encode_boards_batch():
    boards = random_playout_boards(12, seed=5)
    batch = encode_boards(boards)
    assert batch.shape == (12, 20, 8, 8) and batch.dtype == np.float32
    for b, p in zip(boards, batch):
        assert np.array_equal(p, encode_board(b))
    assert encode_boards([]).shape == (0, 20, 8, 8)


# ---------------------------------------------------------------------------------------------
# vectors file
# ---------------------------------------------------------------------------------------------
def test_vectors_file_is_up_to_date_and_self_consistent():
    assert VECTORS_PATH == DEFAULT_PATH
    with open(VECTORS_PATH) as f:
        vectors = json.load(f)
    assert vectors == build_vectors(), "tests/vectors/encoding.json is stale: python -m flychess.chessenv.vectors"
    assert len(vectors) >= 30 and any("move_list" in v for v in vectors)
    for vec in vectors:
        board = chess.Board()
        if "move_list" in vec:
            for uci in vec["move_list"]:
                board.push_uci(uci)
            assert board.fen() == vec["fen"]
        else:
            board = chess.Board(vec["fen"])
        planes = encode_board(board)
        assert planes_sha256(planes) == vec["planes_sha256"]
        rebuilt = np.zeros((20, 8, 8), np.float32)
        for p, r, f_, v in vec["planes_nonzero"]:
            rebuilt[p, r, f_] = v
        rebuilt[18] = vec["halfmove_plane"]
        rebuilt[19] = vec["repetition_plane"]
        assert np.array_equal(rebuilt, planes)
        assert legal_move_indices(board).tolist() == vec["legal_indices"]
        assert {m.uci(): move_to_index(m, board) for m in board.legal_moves} == vec["moves"]


# ---------------------------------------------------------------------------------------------
# ChessEnv
# ---------------------------------------------------------------------------------------------
def test_chess_env_scholars_mate():
    env = ChessEnv()
    obs = env.reset()
    assert obs.shape == (20, 8, 8) and env.legal_mask().sum() == 20 and not env.is_over
    for action in ["e2e4", "e7e5", chess.Move.from_uci("d1h5"), "b8c6", "f1c4", "g8f6"]:
        obs, reward, done, info = env.step(action)
        assert reward == 0.0 and not done
    idx = move_to_index(chess.Move.from_uci("h5f7"), env.board)
    obs, reward, done, info = env.step(idx)
    assert done and reward == 1.0 and info["san"] == "Qxf7#" and info["result"] == "1-0"
    assert env.result() == "1-0" and env.result_value(chess.WHITE) == 1 and env.result_value(chess.BLACK) == -1
    assert env.san_history() == ["e4", "e5", "Qh5", "Nc6", "Bc4", "Nf6", "Qxf7#"]
    assert env.uci_history()[-1] == "h5f7" and env.ply == 7
    assert "Qxf7#" in env.pgn() and env.pgn().rstrip().endswith("1-0")
    with pytest.raises(RuntimeError):
        env.step("e1e2")


def test_chess_env_fen_reset_and_illegal_move():
    env = ChessEnv("8/8/8/4k3/8/8/4K3/7R b - - 0 1")
    assert env.turn == chess.BLACK and env.result_value(chess.BLACK) == 0
    with pytest.raises(ValueError):
        env.step("e5e7")
    obs, _reward, done, _info = env.step("e5d5")
    assert obs[5, 1, 4] == 1  # white to move now: its king on e2 is the mover's king
    assert not done and env.pgn().startswith("[Event") and "FEN" in env.pgn()
    env.reset()
    assert env.board.fen() == chess.STARTING_FEN and env.ply == 0
