"""Cross-language model test vectors (docs/SPEC.md §8): ``tests/vectors/model.json``.

The vectors are computed with :func:`flychess.export.numpy_forward` **on the exported blob**
(``brain.json`` + ``brain.flyb``), so the Python reference and the JS engine
(``web/test/parity.test.mjs``) read exactly the same rounded bytes. Every vector holds::

    {fen, moves?: [uci...],           # moves = history from the start position (repetition plane)
     n_legal, argmax: idx,            # argmax of the policy over the legal moves
     top: [[idx, uci, logit], ...],   # the TOP_K legal moves with the highest logit
     value}                           # tanh value head, mover's perspective

plus file-level ``run_name`` / ``exported_at`` / ``blob_sha256`` so a consumer can tell which export
they belong to. The 12 curated positions cover black to move, castling rights on both sides, both en
passant captures, push / capture promotions for both colours, a check, a long halfmove clock and a
repeated position given as a move list.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import chess
import numpy as np

from flychess import paths
from flychess.chessenv.encoding import encode_board, index_to_move, legal_move_indices
from flychess.export.web import numpy_forward, read_flyb

# Only the type hints below and the model+graph export path need these; keeping them out of the
# module namespace lets the vectors be built from an existing export with numpy + python-chess alone.
if TYPE_CHECKING:  # pragma: no cover - typing only
    from flychess.connectome.graph import BrainGraph
    from flychess.model.flybrain import FlyBrain

DEFAULT_PATH = paths.VECTORS_DIR / "model.json"
TOP_K = 20
TOLERANCE = 1e-2  # |Δlogit|, |Δvalue| allowed between Python and JS (SPEC §8)

# (fen or None, move list or None): a move list is played from the initial position.
CURATED_POSITIONS: list[tuple[str | None, list[str] | None]] = [
    (chess.STARTING_FEN, None),
    ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", None),            # black to move (mirror)
    ("r3k2r/pppqbppp/2npbn2/4p3/4P3/2NPBN2/PPPQBPPP/R3K2R w KQkq - 4 8", None),      # castling both ways
    ("r3k2r/pppqbppp/2npbn2/4p3/4P3/2NPBN2/PPPQBPPP/R3K2R b KQkq - 4 8", None),
    ("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3", None),         # white en passant
    ("rnbqkbnr/pppp1ppp/8/8/3Pp3/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 3", None),         # black en passant
    ("1n6/P7/8/8/8/8/8/k6K w - - 0 1", None),                                       # push + capture promotion
    ("k6K/8/8/8/8/8/6p1/5N1N b - - 0 1", None),                                     # black capture promotions
    ("4k3/8/8/8/8/8/4r3/4K3 w - - 0 1", None),                                      # king in check
    ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b KQkq - 0 1", None), # kiwipete, black
    ("8/8/8/4k3/8/8/4K3/7R w - - 57 100", None),                                    # halfmove clock 57
    (None, ["g1f3", "g8f6", "f3g1", "f6g8"]),                                       # repeated start position
]


def curated_boards(positions: list[tuple[str | None, list[str] | None]] | None = None
                   ) -> list[tuple[chess.Board, list[str] | None]]:
    """``[(board, moves)]`` for the curated positions (or ``positions``); every board has legal moves."""
    out = []
    for fen, moves in positions if positions is not None else CURATED_POSITIONS:
        if moves:
            b = chess.Board()
            for u in moves:
                b.push_uci(u)
        else:
            b = chess.Board(fen)
        if not any(b.legal_moves):
            raise ValueError(f"test-vector position has no legal moves: {b.fen()}")
        out.append((b, list(moves) if moves else None))
    return out


def blob_sha256(model_dir: str | Path) -> str:
    return hashlib.sha256((Path(model_dir) / "brain.flyb").read_bytes()).hexdigest()


def vector_for(arrays: dict[str, np.ndarray], header: dict[str, Any], board: chess.Board,
               moves: list[str] | None = None, top_k: int = TOP_K) -> dict[str, Any]:
    """One vector: numpy forward of the exported brain on ``board``."""
    x = encode_board(board).reshape(-1)
    policy, value, _ = numpy_forward(arrays, header, x)
    legal = legal_move_indices(board).astype(np.int64)
    order = legal[np.argsort(-policy[legal], kind="stable")]
    top = [[int(i), index_to_move(int(i), board).uci(), float(policy[i])] for i in order[:top_k]]
    vec: dict[str, Any] = {"fen": board.fen()}
    if moves:
        vec["moves"] = list(moves)
    vec.update({"n_legal": int(legal.size), "argmax": int(order[0]), "top": top, "value": float(value)})
    return vec


def build_model_vectors(model_dir: str | Path, positions=None, top_k: int = TOP_K) -> dict[str, Any]:
    """Vectors for every curated position from the export in ``model_dir`` (``brain.json`` + ``brain.flyb``)."""
    arrays, header = read_flyb(model_dir)
    vectors = [vector_for(arrays, header, b, mv, top_k) for b, mv in curated_boards(positions)]
    return {
        "format": "flychess-model-vectors",
        "version": 1,
        "run_name": header.get("run_name", ""),
        "exported_at": header.get("exported_at", ""),
        "blob_sha256": blob_sha256(model_dir),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "n": int(header["n"]),
        "nnz": int(header["nnz"]),
        "steps": int(header["steps"]),
        "tolerance": TOLERANCE,
        "vectors": vectors,
    }


def write_model_vectors(
    model: FlyBrain | None = None,
    graph: BrainGraph | None = None,
    out: str | Path = DEFAULT_PATH,
    fens=None,
    *,
    model_dir: str | Path | None = None,
    run_name: str = "",
    quant: str = "f16",
    extra_meta: dict[str, Any] | None = None,
) -> Path:
    """Write ``out`` (default ``tests/vectors/model.json``) and return its path.

    Either pass ``model_dir`` (an existing ``fly export-web`` output: the vectors are computed from
    that blob) or ``model`` + ``graph`` (exported with ``quant`` into a temporary directory first).
    ``fens`` optionally replaces the curated positions (``[(fen, moves)]``).
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if model_dir is not None and (Path(model_dir) / "brain.json").exists():
        data = build_model_vectors(model_dir, fens)
    else:
        if model is None or graph is None:
            raise ValueError("write_model_vectors needs model + graph or an existing model_dir")
        from flychess.export.web import export_web
        with tempfile.TemporaryDirectory(prefix="flychess-vectors-") as tmp:
            export_web(model, graph, tmp, quant=quant, extra_meta={"run_name": run_name, **(extra_meta or {})})
            data = build_model_vectors(tmp, fens)
    out.write_text(json.dumps(data, indent=1))
    return out


def check_model_vectors(model_dir: str | Path, vectors_path: str | Path = DEFAULT_PATH,
                        tol: float = TOLERANCE) -> dict[str, float]:
    """Recompute the vectors from ``model_dir`` and compare (Python-side self check); returns max deltas."""
    data = json.loads(Path(vectors_path).read_text())
    arrays, header = read_flyb(model_dir)
    max_logit = max_value = 0.0
    for vec in data["vectors"]:
        board = chess.Board()
        if vec.get("moves"):
            for u in vec["moves"]:
                board.push_uci(u)
        else:
            board = chess.Board(vec["fen"])
        got = vector_for(arrays, header, board, vec.get("moves"), top_k=len(vec["top"]))
        exp_logits = {i: v for i, _, v in vec["top"]}
        got_logits = {i: v for i, _, v in got["top"]}
        for i, v in exp_logits.items():
            if i in got_logits:
                max_logit = max(max_logit, abs(got_logits[i] - v))
        max_value = max(max_value, abs(got["value"] - vec["value"]))
        if got["argmax"] != vec["argmax"]:
            raise AssertionError(f"argmax differs for {vec['fen']}: {got['argmax']} != {vec['argmax']}")
    if max_logit > tol or max_value > tol:
        raise AssertionError(f"model vectors differ: max |Δlogit|={max_logit:.4g} |Δvalue|={max_value:.4g} > {tol}")
    return {"max_logit_delta": max_logit, "max_value_delta": max_value}


__all__ = [
    "CURATED_POSITIONS", "DEFAULT_PATH", "TOLERANCE", "TOP_K", "blob_sha256", "build_model_vectors",
    "check_model_vectors", "curated_boards", "vector_for", "write_model_vectors",
]
