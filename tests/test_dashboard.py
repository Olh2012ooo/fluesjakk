"""Tests for flychess.train.metrics (MetricsLogger, readers) and flychess.dashboard.server (HTTP + websocket)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from flychess.connectome.graph import toy_graph
from flychess.dashboard.server import RunIndex, _safe_name, create_app
from flychess.train.metrics import MetricsLogger, iter_lines_reversed, list_runs, read_metrics, read_run_json


# ------------------------------------------------------------------------------------------------
# fixtures
# ------------------------------------------------------------------------------------------------
def _fill_run(run_dir: Path, n_train: int = 50, graph_path: str | None = None) -> MetricsLogger:
    log = MetricsLogger(run_dir, rate_limits={"game": 0.0, "activity": 0.0})
    cfg = {"batch_size": 8, "stage": "imitation", "steps": 1000}
    if graph_path:
        cfg["graph_path"] = graph_path
    log.write_run_json(cfg, {"n": 200, "nnz": 2000}, extra={"graph_path": graph_path} if graph_path else None)
    log.log_status(0, message="starting", stage="imitation", total_steps=1000, eta_s=None)
    for step in range(0, n_train * 20, 20):
        log.log_train(
            step, loss=3.0 / (1 + step / 100), policy_loss=2.5 / (1 + step / 100), value_loss=0.5, top1=0.1,
            top3=0.3, lr=1e-3, pos_per_sec=5000.0, gpu_mem_gb=4.2, epoch=step / 500, stage="imitation",
        )
        if step % 200 == 0:
            log.log_eval(step, val_loss=2.8, val_top1=0.12, val_top3=0.31, val_value_mse=0.45)
    log.log_elo(400, opponent="random", games=20, wins=15, draws=3, losses=2, elo_estimate=210.5)
    log.log_game(400, pgn="1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 1-0", result="1-0", moves=6, source="eval")
    log.log_activity(400, neuron_idx=np.arange(0, 16), values=np.random.default_rng(0).random(16))
    return log


@pytest.fixture
def runs_dir(tmp_path: Path) -> Path:
    root = tmp_path / "runs"
    log = _fill_run(root / "alpha")
    log.close()
    (root / "not-a-run").mkdir()  # directory without run.json / metrics.jsonl must be ignored
    return root


# ------------------------------------------------------------------------------------------------
# MetricsLogger
# ------------------------------------------------------------------------------------------------
def test_logger_writes_and_reads(tmp_path: Path) -> None:
    log = MetricsLogger(tmp_path / "r")
    rec = log.log_train(
        step=20, loss=1.5, policy_loss=1.0, value_loss=0.5, top1=np.float32(0.25), top3=0.5, lr=1e-3,
        pos_per_sec=1234.5, gpu_mem_gb=3.1, epoch=0.1, stage="imitation", extra_field=np.int64(7),
    )
    assert rec is not None and rec["kind"] == "train" and rec["step"] == 20
    log.log_eval(20, val_loss=1.4, val_top1=0.3, val_top3=0.6, val_value_mse=float("nan"))
    log.log_status(20, message="hello", stage="imitation", total_steps=100, eta_s=12.5)
    log.close()

    lines = (tmp_path / "r" / "metrics.jsonl").read_text().splitlines()
    assert len(lines) == 3
    for line in lines:
        d = json.loads(line)
        assert {"t", "step", "kind"} <= d.keys()
        assert abs(d["t"] - time.time()) < 60
    assert json.loads(lines[0])["top1"] == pytest.approx(0.25)
    assert json.loads(lines[0])["extra_field"] == 7
    assert json.loads(lines[1])["val_value_mse"] is None  # NaN -> null (browser JSON.parse safe)

    recs = read_metrics(tmp_path / "r")
    assert [r["kind"] for r in recs] == ["train", "eval", "status"]
    assert read_metrics(tmp_path / "r", kinds=["eval"])[0]["val_loss"] == 1.4
    assert [r["kind"] for r in read_metrics(tmp_path / "r", last_n=2)] == ["eval", "status"]
    assert [r["kind"] for r in read_metrics(tmp_path / "r", last_n=1, kinds=["train"])] == ["train"]
    assert read_metrics(tmp_path / "missing") == []


def test_logger_flushes_every_write(tmp_path: Path) -> None:
    log = MetricsLogger(tmp_path / "r")
    log.log("status", 1, message="x", stage="s", total_steps=1, eta_s=0)
    # visible to another reader *before* close
    assert len(read_metrics(tmp_path / "r")) == 1
    log.close()


def test_rate_limiting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    now = [1_000_000.0]
    monkeypatch.setattr("flychess.train.metrics.time.time", lambda: now[0])
    log = MetricsLogger(tmp_path / "r")
    assert log.log_game(1, pgn="1. e4 *", result="*", moves=1, source="selfplay") is True
    assert log.log_game(2, pgn="1. d4 *", result="*", moves=1, source="selfplay") is False  # < 60 s later
    now[0] += 59.0
    assert log.log_game(3, pgn="1. c4 *", result="*", moves=1, source="selfplay") is False
    now[0] += 1.5
    assert log.log_game(4, pgn="1. f4 *", result="*", moves=1, source="selfplay") is True

    assert log.log_activity(4, neuron_idx=[0, 1], values=[0.1, 0.2]) is True
    now[0] += 29.0
    assert log.log_activity(5, neuron_idx=[0, 1], values=[0.1, 0.2]) is False
    now[0] += 1.0
    assert log.log_activity(6, neuron_idx=[0, 1], values=[0.1, 0.2]) is True
    # non rate-limited kinds are never dropped
    for i in range(5):
        assert log.log_train(i, loss=1, policy_loss=1, value_loss=0, top1=0, top3=0, lr=0, pos_per_sec=0,
                             gpu_mem_gb=0, epoch=0, stage="s") is not None
    log.close()
    kinds = [r["kind"] for r in read_metrics(tmp_path / "r")]
    assert kinds.count("game") == 2 and kinds.count("activity") == 2 and kinds.count("train") == 5


def test_run_json_idempotent(tmp_path: Path) -> None:
    log = MetricsLogger(tmp_path / "r")
    first = log.write_run_json({"lr": 1e-3, "graph_path": "data/brain/tiny.npz"}, {"n": 10}, extra={"note": "a"})
    assert first["started_at"] > 0 and first["graph_path"] == "data/brain/tiny.npz" and first["note"] == "a"
    time.sleep(0.01)
    second = log.write_run_json({"lr": 2e-3}, {"n": 11})
    assert second["started_at"] == first["started_at"]
    assert second["config"] == {"lr": 2e-3} and second["graph_meta"] == {"n": 11}
    assert second["graph_path"] == "data/brain/tiny.npz"  # preserved from the first write
    assert read_run_json(tmp_path / "r")["started_at"] == first["started_at"]
    log.close()


def test_iter_lines_reversed_and_torn_line(tmp_path: Path) -> None:
    p = tmp_path / "m.jsonl"
    p.write_bytes(b"".join(json.dumps({"i": i, "pad": "x" * 50}).encode() + b"\n" for i in range(3000)))
    with open(p, "ab") as f:
        f.write(b'{"i": 3000, "kind": "tr')  # writer mid-line
    rev = [json.loads(ln)["i"] for ln in iter_lines_reversed(p, block_size=1000) if ln.endswith(b"}")]
    assert rev == list(range(2999, -1, -1))
    (tmp_path / "run").mkdir()
    p.rename(tmp_path / "run" / "metrics.jsonl")
    recs = read_metrics(tmp_path / "run", last_n=5)
    assert [r["i"] for r in recs] == [2995, 2996, 2997, 2998, 2999]


def test_list_runs(runs_dir: Path) -> None:
    runs = list_runs(runs_dir)
    assert [r["name"] for r in runs] == ["alpha"]
    r = runs[0]
    assert r["started_at"] > 0 and r["last_t"] > 0
    assert r["last_step"] == 400 and r["stage"] == "imitation"
    # run with only a metrics file and no run.json
    log = MetricsLogger(runs_dir / "beta")
    log.log_status(7, message="m", stage="selfplay", total_steps=10, eta_s=1)
    log.close()
    runs = list_runs(runs_dir)
    assert [r["name"] for r in runs] == ["beta", "alpha"]  # newest activity first
    assert runs[0]["started_at"] is None and runs[0]["stage"] == "selfplay" and runs[0]["last_step"] == 7
    assert list_runs(runs_dir / "does-not-exist") == []


# ------------------------------------------------------------------------------------------------
# server
# ------------------------------------------------------------------------------------------------
def test_http_endpoints(runs_dir: Path) -> None:
    app = create_app(runs_dir=runs_dir, default_run="alpha")
    with TestClient(app) as client:
        r = client.get("/")
        assert r.status_code == 200 and "fly-chess" in r.text and "dashboard.js" in r.text
        assert client.get("/static/dashboard.js").status_code == 200
        assert client.get("/static/dashboard.css").status_code == 200
        assert client.get("/static/chartmath.js").status_code == 200
        assert r.text.index("/static/chartmath.js") < r.text.index("/static/dashboard.js")  # helpers load first

        assert client.get("/api/default").json()["run"] == "alpha"

        runs = client.get("/api/runs").json()
        assert [x["name"] for x in runs] == ["alpha"] and runs[0]["last_step"] == 400

        data = client.get("/api/run/alpha").json()
        assert data["run"]["config"]["batch_size"] == 8 and data["run"]["graph_meta"]["n"] == 200
        kinds = {m["kind"] for m in data["metrics"]}
        assert kinds == {"status", "train", "eval", "elo", "game", "activity"}
        assert data["metrics"][0]["kind"] == "status" and data["metrics"][-1]["kind"] == "activity"

        assert client.get("/api/run/nope").status_code == 404
        assert client.get("/api/run/.hidden").status_code == 400
    for bad in ("..", ".", "a/b", "a\\b", ""):
        with pytest.raises(HTTPException):
            _safe_name(bad)

        # no graph_path in run.json -> positions unavailable but well-formed
        assert client.get("/api/run/alpha/positions").json()["available"] is False


def _fill_dense(root: Path, name: str, n_train: int, status_every: int = 1) -> None:
    """A run whose dense train/status logging would push sparse kinds out of any fixed-size tail."""
    log = MetricsLogger(root / name, rate_limits={"game": 0.0, "activity": 0.0})
    log.log_elo(0, opponent="random", games=4, wins=2, draws=1, losses=1, elo_estimate=12.0)
    log.log_game(0, pgn="1. e4 e5 *", result="*", moves=2, source="eval")
    log.log_activity(0, neuron_idx=[0, 1], values=[0.5, 0.25])
    log.log_eval(0, val_loss=9.0, val_top1=0.0, val_top3=0.0, val_value_mse=1.0)
    for i in range(n_train):
        log.log("train", i, loss=8.0 - 7.0 * i / max(1, n_train - 1), lr=1e-3, stage="imitation")
        if i % status_every == 0:
            log.log_status(i, message=f"s{i}", stage="imitation", total_steps=n_train, eta_s=1.0)
    log.close()


def test_api_run_sends_whole_series_per_kind(tmp_path: Path) -> None:
    """The init is per kind (whole train / eval / elo series, last game + activity, last status lines) rather than
    a global tail of 5000 records: a reload must show the whole run and never lose the sparse kinds."""
    root = tmp_path / "runs"
    _fill_dense(root, "big", n_train=6000)
    with TestClient(create_app(runs_dir=root)) as client:
        data = client.get("/api/run/big").json()
        m = data["metrics"]
        by_kind: dict[str, list[dict]] = {}
        for r in m:
            by_kind.setdefault(r["kind"], []).append(r)
        train = by_kind["train"]
        assert len(train) == 6000 and train[0]["step"] == 0 and train[-1]["step"] == 5999  # the *whole* series
        assert train[0]["loss"] == pytest.approx(8.0) and train[-1]["loss"] == pytest.approx(1.0)
        assert len(by_kind["status"]) == 200 and by_kind["status"][-1]["step"] == 5999  # only the log tail
        assert [r["step"] for r in by_kind["elo"]] == [0] and [r["step"] for r in by_kind["eval"]] == [0]
        assert len(by_kind["game"]) == 1 and len(by_kind["activity"]) == 1  # sparse kinds survive dense logging
        assert [r["kind"] for r in m[:4]] == ["elo", "game", "activity", "eval"]  # file order is preserved
        assert data["offset"] == (root / "big" / "metrics.jsonl").stat().st_size
        assert data["counts"] == {"train": 6000, "status": 6000, "elo": 1, "game": 1, "activity": 1, "eval": 1}


def test_run_index_is_incremental_and_handles_truncation(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    _fill_dense(root, "r", n_train=30)
    path = root / "r" / "metrics.jsonl"
    idx = RunIndex(path, status_lines=5)
    metrics, offset, counts = idx.load()
    assert offset == path.stat().st_size and counts["train"] == 30
    assert sum(r["kind"] == "train" for r in metrics) == 30 and sum(r["kind"] == "status" for r in metrics) == 5

    # append (plus a torn last line): only the new bytes are parsed, the torn line waits for the writer
    with open(path, "ab") as f:
        f.write(json.dumps({"t": 1.0, "step": 30, "kind": "train", "loss": 0.5}).encode() + b"\n")
        f.write(b'{"t": 2.0, "step": 31, "kind": "tr')
    metrics2, offset2, counts2 = idx.load()
    assert counts2["train"] == 31 and offset2 == path.stat().st_size - len(b'{"t": 2.0, "step": 31, "kind": "tr')
    assert metrics2[-1]["step"] == 30 and len(metrics2) == len(metrics) + 1
    assert idx.load()[1] == offset2  # nothing new -> no change

    # the file shrinks (run rewritten from scratch): the index rebuilds itself
    path.write_bytes(json.dumps({"t": 3.0, "step": 0, "kind": "status", "message": "reborn"}).encode() + b"\n")
    metrics3, offset3, counts3 = idx.load()
    assert counts3 == {"status": 1} and offset3 == path.stat().st_size and metrics3[0]["message"] == "reborn"


def test_run_index_caps_train_by_bucket_means(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    _fill_dense(root, "r", n_train=1000, status_every=100)
    with TestClient(create_app(runs_dir=root, init_max_train=100)) as client:
        m = client.get("/api/run/r").json()["metrics"]
    train = [r for r in m if r["kind"] == "train"]
    assert len(train) == 100
    assert train[0]["step"] == 9 and train[0]["loss"] == pytest.approx(sum(8.0 - 7.0 * i / 999 for i in range(10)) / 10)
    assert train[-1]["step"] == 999 and train[-1]["loss"] == pytest.approx(1.0)  # true last record, not a mean
    assert train[0]["stage"] == "imitation" and train[0]["lr"] == pytest.approx(1e-3)
    assert [r["step"] for r in m if r["kind"] != "train"] == sorted(r["step"] for r in m if r["kind"] != "train")
    steps = [r["step"] for r in m]
    assert steps == sorted(steps)  # buckets are merged into file order with the other kinds


def test_positions_from_graph(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    g = toy_graph(n=120, nnz=600, seed=1)
    g.position[5] = np.nan  # unknown position -> null
    gp = tmp_path / "toy.npz"
    g.save(gp)
    log = _fill_run(root / "g", n_train=2, graph_path=str(gp))
    log.close()
    with TestClient(create_app(runs_dir=root)) as client:
        j = client.get("/api/run/g/positions").json()
        assert j["available"] is True and j["n"] == 120
        assert j["neuron_idx"] == list(range(16))  # from the latest activity sample
        assert j["positions"][5] is None
        for p in j["positions"]:
            if p is not None:
                assert len(p) == 3 and all(0.0 <= v <= 1.0 for v in p)
        assert j["super_class"] == ["central"] * 16


def test_websocket_init_and_tail(runs_dir: Path) -> None:
    app = create_app(runs_dir=runs_dir)
    with TestClient(app) as client, client.websocket_connect("/ws/alpha") as ws:
        init = ws.receive_json()
        assert init["kind"] == "init"
        assert init["run"]["config"]["batch_size"] == 8
        assert len(init["metrics"]) > 10 and init["metrics"][-1]["kind"] == "activity"
        size0 = (runs_dir / "alpha" / "metrics.jsonl").stat().st_size
        assert init["offset"] == size0

        log = MetricsLogger(runs_dir / "alpha")
        log.log_status(999, message="tailed", stage="selfplay", total_steps=2000, eta_s=3.0)
        log.close()
        msg = ws.receive_json()  # pushed within ~POLL_INTERVAL
        assert msg["kind"] == "status" and msg["step"] == 999 and msg["message"] == "tailed"
        cursor = ws.receive_json()  # followed by the byte cursor the client resumes from
        assert cursor == {"kind": "cursor", "offset": (runs_dir / "alpha" / "metrics.jsonl").stat().st_size}
        assert cursor["offset"] > size0


def test_websocket_resume_keeps_history_and_falls_back_on_stale_cursor(runs_dir: Path) -> None:
    app = create_app(runs_dir=runs_dir)
    path = runs_dir / "alpha" / "metrics.jsonl"
    with TestClient(app) as client:
        with client.websocket_connect("/ws/alpha") as ws:
            offset = ws.receive_json()["offset"]
        # write while "disconnected", then reconnect with the cursor: no init, only what was missed
        log = MetricsLogger(runs_dir / "alpha")
        log.log_status(1000, message="missed", stage="selfplay", total_steps=2000, eta_s=2.0)
        log.close()
        with client.websocket_connect(f"/ws/alpha?after={offset}") as ws:
            assert ws.receive_json() == {"kind": "resume", "offset": offset}
            missed = ws.receive_json()
            assert missed["kind"] == "status" and missed["message"] == "missed"
            assert ws.receive_json() == {"kind": "cursor", "offset": path.stat().st_size}
        # a cursor beyond the file (stale / different file) -> full init
        with client.websocket_connect(f"/ws/alpha?after={path.stat().st_size + 10}") as ws:
            init = ws.receive_json()
            assert init["kind"] == "init" and init["metrics"][-1]["message"] == "missed"
        # the file shrinks while a client is connected -> a fresh init instead of records from byte 0
        with client.websocket_connect(f"/ws/alpha?after={path.stat().st_size}") as ws:
            assert ws.receive_json()["kind"] == "resume"
            path.write_bytes(json.dumps({"t": 1.0, "step": 0, "kind": "status", "message": "reborn"}).encode() + b"\n")
            init = ws.receive_json()
            assert init["kind"] == "init" and [m["message"] for m in init["metrics"]] == ["reborn"]
            assert init["offset"] == path.stat().st_size


def test_websocket_waits_for_run_to_appear(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    root.mkdir()
    app = create_app(runs_dir=root)
    with TestClient(app) as client, client.websocket_connect("/ws/later") as ws:
        time.sleep(0.7)  # run does not exist yet: the server must simply wait
        log = MetricsLogger(root / "later")
        log.write_run_json({"x": 1}, {})
        log.log_status(0, message="born", stage="imitation", total_steps=5, eta_s=None)
        init = ws.receive_json()
        assert init["kind"] == "init" and init["run"]["config"] == {"x": 1}
        assert [m["message"] for m in init["metrics"]] == ["born"]
        log.log_status(1, message="next", stage="imitation", total_steps=5, eta_s=None)
        log.close()
        assert ws.receive_json()["message"] == "next"
