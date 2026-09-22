"""Round-trip tests for the web export format (docs/SPEC.md §8)."""
from __future__ import annotations

import dataclasses
import gzip
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from flychess.connectome.graph import toy_graph
from flychess.export import SUPER_CLASS_LEGEND, export_web, numpy_forward, read_flyb
from flychess.export.web import normalise_positions
from flychess.model import BrainConfig, FlyBrain

FEATURE_ARRAYS = ["retina_idx", "retina_square", "retina_uv", "retina_eye", "w_ret", "b_ret", "mod_indptr",
                  "mod_indices", "w_mod", "central_idx", "central_w", "central_b", "retina_type"]
EXPECTED_ORDER = ["csr_indptr", "csr_indices", "w", "bias", "alpha", "input_idx", "output_idx", "w_in", "b_in",
                  "policy_w", "policy_b", "value_w", "value_b", "value_w2", "value_b2", "positions", "super_class",
                  "node_perm", *FEATURE_ARRAYS]
EXPECTED_DTYPES = {"csr_indptr": "i32", "csr_indices": "i32", "w": "f16", "bias": "f32", "alpha": "f32",
                   "input_idx": "i32", "output_idx": "i32", "w_in": "f16", "b_in": "f32", "policy_w": "f16",
                   "policy_b": "f32", "value_w": "f16", "value_b": "f32", "value_w2": "f16", "value_b2": "f32",
                   "positions": "f16", "super_class": "u8", "node_perm": "i32",
                   "retina_idx": "i32", "retina_square": "u8", "retina_uv": "f16", "retina_eye": "u8", "w_ret": "f16",
                   "b_ret": "f32", "mod_indptr": "i32", "mod_indices": "i32", "w_mod": "f16", "central_idx": "i32",
                   "central_w": "f16", "central_b": "f32", "retina_type": "u8"}
FEATURE_HEADER = ("vision", "sensory_input", "readout_steps", "neuromod", "central_dim", "n_ret", "n_central",
                  "nnz_mod", "nnz_total", "retina_type_legend")


def with_neuromod(g, seed=0):
    """Relabel a third of the excitatory (+1) neurons as DA / SER / OCT (sign stays +1: validate() holds)."""
    rng = np.random.default_rng(seed)
    nt = np.asarray(g.nt_type).astype("U4").copy()
    exc = np.flatnonzero(nt == "ACH")
    pick = rng.choice(exc, size=len(exc) // 3, replace=False)
    nt[pick] = rng.choice(["DA", "SER", "OCT"], size=len(pick))
    g2 = dataclasses.replace(g, nt_type=nt)
    g2.validate()
    return g2


def randomise_(m):
    """Non-trivial biases / leaks / input offsets so that parity tests are meaningful."""
    with torch.no_grad():
        m.bias.normal_(0, 0.3)
        m.leak_logit.normal_(0, 1.0)
        if m.sensory_input:
            m.b_in.normal_(0, 0.5)
        if m.vision:
            m.b_ret.normal_(0, 0.5)
        if isinstance(m.value_head, torch.nn.Sequential):
            m.value_head[0].bias.normal_(0, 1.0)
    return m


@pytest.fixture(scope="module")
def graph():
    g = toy_graph(n=150, nnz=1200, n_in=14, n_out=11, seed=7)
    g.super_class = np.array(["central"] * 100 + ["descending"] * 20 + ["sensory"] * 25 + ["weird"] * 5)
    g.position[3] = np.nan  # unknown position
    return g


@pytest.fixture(scope="module")
def graph_full():
    """Toy graph with a retina, DA/SER/OCT neurons and 'central' neurons (every optional feature usable)."""
    g = toy_graph(n=160, nnz=1300, n_in=12, n_out=10, n_ret=36, seed=11)
    return with_neuromod(g, seed=1)


@pytest.fixture(scope="module")
def model(graph):
    torch.manual_seed(0)
    cfg = BrainConfig(graph_path="toy", steps=5, num_moves=96, value_hidden=32, activation="gelu")
    m = FlyBrain(graph, cfg)
    with torch.no_grad():  # make biases / leaks non-trivial so the parity test is meaningful
        m.bias.normal_(0, 0.3)
        m.leak_logit.normal_(0, 1.0)
        if m.sensory_input:
            m.b_in.normal_(0, 0.5)
    return m.eval()


def test_export_roundtrip_header_and_layout(model, graph, tmp_path):
    header = export_web(model, graph, tmp_path, extra_meta={"run_name": "fly1", "train_steps": 42,
                                                             "elo_estimates": {"random": 900}, "note": "hi"})
    assert (tmp_path / "brain.json").exists() and (tmp_path / "brain.flyb").exists()
    assert (tmp_path / "brain.flyb.gz").exists()
    assert gzip.decompress((tmp_path / "brain.flyb.gz").read_bytes()) == (tmp_path / "brain.flyb").read_bytes()
    assert json.loads((tmp_path / "brain.json").read_text()) == header
    for key in ("steps", "activation", "n", "nnz", "n_in", "n_out", "num_moves", "num_planes", "run_name",
                "train_steps", "exported_at", "elo_estimates", "super_class_legend", "arrays"):
        assert key in header, key
    assert header["run_name"] == "fly1" and header["train_steps"] == 42 and header["note"] == "hi"
    assert header["elo_estimates"] == {"random": 900}
    assert header["num_planes"] == 20 and header["n"] == graph.n and header["nnz"] == graph.nnz
    assert header["steps"] == 5 and header["activation"] == "gelu" and header["value_head"] == "mlp"
    assert header["value_activation"] == "gelu"  # the value MLP's hidden non-linearity is explicit
    names = [e["name"] for e in header["arrays"]]
    assert names == EXPECTED_ORDER
    for e in header["arrays"]:
        assert e["dtype"] == EXPECTED_DTYPES[e["name"]], e["name"]
        assert e["offset"] % 8 == 0
        assert "scale" not in e
    assert header["arrays"][0]["offset"] == 0
    assert header["total_bytes"] == (tmp_path / "brain.flyb").stat().st_size
    # no retina / neuromod / central on this model: the feature arrays are present but empty
    by_name0 = {e["name"]: e for e in header["arrays"]}
    for name in FEATURE_ARRAYS:
        assert int(np.prod(by_name0[name]["shape"])) == 0 and by_name0[name]["length_bytes"] == 0, name
    assert by_name0["retina_uv"]["shape"] == [0, 2] and by_name0["w_ret"]["shape"] == [0, 20]
    for key in FEATURE_HEADER:
        assert key in header, key
    assert header["vision"] is False and header["sensory_input"] is True and header["neuromod"] is False
    assert header["readout_steps"] == [5] and header["central_dim"] == 0 and header["n_ret"] == 0
    assert header["nnz"] == graph.nnz and header["nnz_mod"] == 0 and header["nnz_total"] == graph.nnz
    # the web loader verifies the blob it downloads against these
    assert header["gzip_bytes"] == (tmp_path / "brain.flyb.gz").stat().st_size
    assert header["blob_sha256"] == hashlib.sha256((tmp_path / "brain.flyb").read_bytes()).hexdigest()
    by_name = {e["name"]: e for e in header["arrays"]}
    assert by_name["w_in"]["shape"] == [graph.n_in, 1280]
    assert by_name["policy_w"]["shape"] == [96, graph.n_out]
    assert by_name["value_w"]["shape"] == [32, graph.n_out] and by_name["value_w2"]["shape"] == [1, 32]
    assert by_name["positions"]["shape"] == [graph.n, 3] and by_name["super_class"]["shape"] == [graph.n]

    arrays, header2 = read_flyb(tmp_path)
    assert header2 == header
    # the blob is in compute (RCM) order: mapping it back through node_perm must give the canonical graph
    perm = arrays["node_perm"].astype(np.int64)
    assert header["neuron_order"] == "rcm" and sorted(perm.tolist()) == list(range(graph.n))
    inv = np.argsort(perm)
    assert np.array_equal(np.sort(inv[graph.input_idx]), np.sort(arrays["input_idx"]))
    assert np.array_equal(np.sort(inv[graph.output_idx]), np.sort(arrays["output_idx"]))
    rows = np.repeat(np.arange(graph.n), np.diff(arrays["csr_indptr"]))
    edges_blob = set(zip(perm[rows].tolist(), perm[arrays["csr_indices"]].tolist()))          # (post, pre) canonical
    rows_g = np.repeat(np.arange(graph.n), np.diff(graph.csr_indptr))
    assert edges_blob == set(zip(rows_g.tolist(), graph.csr_indices.tolist()))
    assert arrays["w"].dtype == np.float16 and arrays["w"].shape == (graph.nnz,)
    edge_perm = model.edge_perm.numpy()
    assert np.allclose(arrays["w"].astype(np.float32), model.effective_weights().detach().numpy()[edge_perm], atol=1e-2)
    assert np.allclose(arrays["alpha"], torch.sigmoid(model.leak_logit).detach().numpy()[perm])
    # sign of the exported w is the connectome sign
    assert np.array_equal(np.sign(arrays["w"].astype(np.float32)), graph.sign[edge_perm])
    # positions in [0,1], NaN -> 0.5 (graph neuron 3 has NaN positions)
    pos = arrays["positions"].astype(np.float32)
    assert pos.min() >= 0 and pos.max() <= 1 and np.allclose(pos[inv[3]], 0.5)
    # super_class legend
    legend = header["super_class_legend"]
    assert legend == list(SUPER_CLASS_LEGEND)
    sc = arrays["super_class"]
    assert legend[sc[inv[0]]] == "central" and legend[sc[inv[100]]] == "descending" and legend[sc[inv[120]]] == "sensory"
    assert legend[sc[inv[graph.n - 1]]] == "unknown"


def test_numpy_forward_matches_torch(model, graph, tmp_path):
    export_web(model, graph, tmp_path)
    arrays, header = read_flyb(tmp_path)
    m = FlyBrain.from_checkpoint(model.state_dict(), model.config, graph).eval().round_weights_to_f16_()
    rng = np.random.default_rng(0)
    for _ in range(3):
        x = (rng.random(1280) < 0.2).astype(np.float32)
        x[17 * 64:18 * 64] = 1.0  # bias plane
        policy, value, h = numpy_forward(arrays, header, x)
        with torch.no_grad():
            p_t, v_t, h_t = m(torch.from_numpy(x)[None], return_activity=True)
        assert policy.shape == (96,) and h.shape == (graph.n,)
        assert np.allclose(policy, p_t[0].numpy(), atol=1e-2), np.abs(policy - p_t[0].numpy()).max()
        assert abs(value - float(v_t[0, 0])) < 1e-2
        # the blob is in the model's compute (RCM) order; node_perm maps blob index -> canonical index
        assert np.allclose(h, h_t[0].numpy()[arrays["node_perm"]], atol=1e-3)
        assert np.abs(h).max() > 0  # the brain actually responded


def test_linear_value_head_and_i8(graph, tmp_path):
    cfg = BrainConfig(graph_path="toy", steps=2, num_moves=40, value_hidden=0)
    m = FlyBrain(graph, cfg).eval()
    with torch.no_grad():  # heavy-tailed rows: a single per-array scale would flush most of w_in to zero
        m.w_in[0].mul_(200.0)
        m.policy_head.weight[3].mul_(100.0)
    header = export_web(m, graph, tmp_path, quant="i8")
    names = [e["name"] for e in header["arrays"]]
    assert "value_w2" not in names and header["value_head"] == "linear"
    by_name = {e["name"]: e for e in header["arrays"]}
    assert by_name["w"]["dtype"] == "i8" and isinstance(by_name["w"]["scale"], float)   # 1-D: one scale
    assert by_name["value_w"]["shape"] == [1, graph.n_out] and isinstance(by_name["value_w"]["scale"], float)
    for name in ("w_in", "policy_w"):                                                   # 2-D: per-row scales
        assert by_name[name]["dtype"] == "i8"
        assert isinstance(by_name[name]["scale"], list) and len(by_name[name]["scale"]) == by_name[name]["shape"][0]
    arrays, header = read_flyb(tmp_path)
    # per-row dequantisation reproduces every row to i8 precision (the big rows did not kill the others)
    w_in = arrays["w_in"].astype(np.float32) * np.asarray(by_name["w_in"]["scale"], np.float32)[:, None]
    ref = m.w_in.detach().numpy()
    assert np.all(np.abs(w_in - ref) <= np.abs(ref).max(axis=1, keepdims=True) / 127 * 0.51)
    assert (arrays["w_in"][1:] != 0).mean() > 0.9
    x = np.zeros(1280, np.float32)
    x[17 * 64:18 * 64] = 1.0
    policy, value, _ = numpy_forward(arrays, header, x)
    with torch.no_grad():
        p_t, _ = m(torch.from_numpy(x)[None])
    assert np.allclose(policy, p_t[0].numpy(), atol=5e-2, rtol=2e-2)  # i8 is lossier
    assert -1 <= value <= 1


def test_normalise_positions():
    pos = np.array([[0, 10, np.nan], [2, 10, 5], [1, 10, 0]], np.float32)
    out = normalise_positions(pos)
    assert np.allclose(out[:, 0], [0, 1, 0.5]) and np.allclose(out[:, 1], 0.5) and np.allclose(out[:, 2], [0.5, 1, 0])


# ---- optional features: retina / readout steps / neuromod / central summary ---------------------------
FEATURE_COMBOS = [
    {},
    {"readout_steps": (2, 4)},
    {"neuromod": True},
    {"central_dim": 6},
    {"vision": False},
    {"sensory_input": False},
    {"readout_steps": (1, 3, 5), "neuromod": True, "central_dim": 5},
    {"readout_steps": (5,), "neuromod": True, "central_dim": 3, "sensory_input": False, "dale": False},
]


@pytest.mark.parametrize("activation", ["satrelu", "gelu"])
@pytest.mark.parametrize("features", FEATURE_COMBOS, ids=[",".join(f"{k}={v}" for k, v in c.items()) or "plain"
                                                         for c in FEATURE_COMBOS])
def test_numpy_forward_matches_torch_all_features(graph_full, tmp_path, features, activation):
    """numpy_forward (the reference for the JS engines) == torch for every feature combination."""
    g = graph_full
    cfg = BrainConfig(graph_path="toy", steps=5, num_moves=80, value_hidden=16, activation=activation, **features)
    torch.manual_seed(2)
    m = randomise_(FlyBrain(g, cfg)).eval()
    assert m.vision == features.get("vision", True) and m.neuromod == features.get("neuromod", False)
    assert m.nnz_mod > 0 if m.neuromod else m.nnz_mod == 0
    header = export_web(m, g, tmp_path)
    arrays, header = read_flyb(tmp_path)
    assert header["vision"] == m.vision and header["neuromod"] == m.neuromod
    assert header["readout_steps"] == list(m.readout_steps) and header["central_dim"] == m.central_dim
    assert header["n_ret"] == (g.n_ret if m.vision else 0) and header["nnz_total"] == g.nnz
    assert header["nnz"] + header["nnz_mod"] == g.nnz
    m.round_weights_to_f16_()
    rng = np.random.default_rng(5)
    for _ in range(3):
        x = (rng.random(1280) < 0.25).astype(np.float32)
        x[17 * 64:18 * 64] = 1.0
        policy, value, h = numpy_forward(arrays, header, x)
        with torch.no_grad():
            p_t, v_t, h_t = m(torch.from_numpy(x)[None], return_activity=True)
        assert np.allclose(policy, p_t[0].numpy(), atol=1e-2), np.abs(policy - p_t[0].numpy()).max()
        assert abs(value - float(v_t[0, 0])) < 1e-2
        assert np.allclose(h, h_t[0].numpy()[arrays["node_perm"]], atol=1e-3)
        assert np.abs(h).max() > 0


def test_export_feature_arrays_roundtrip(graph_full, tmp_path):
    g = graph_full
    cfg = BrainConfig(graph_path="toy", steps=4, num_moves=50, value_hidden=8, readout_steps=(2, 4), neuromod=True,
                      central_dim=7)
    m = randomise_(FlyBrain(g, cfg)).eval()
    header = export_web(m, g, tmp_path)
    names = [e["name"] for e in header["arrays"]]
    assert names == EXPECTED_ORDER
    for e in header["arrays"]:
        assert e["dtype"] == EXPECTED_DTYPES[e["name"]] and e["offset"] % 8 == 0, e["name"]
    arrays, _ = read_flyb(tmp_path)
    perm = arrays["node_perm"].astype(np.int64)
    inv = np.argsort(perm)
    # retina in compute order, other per-photoreceptor arrays in the graph's retina order
    assert np.array_equal(arrays["retina_idx"], inv[g.retina_idx])
    assert np.array_equal(arrays["retina_square"], g.retina_square.astype(np.uint8))
    assert np.array_equal(arrays["retina_eye"], g.retina_eye.astype(np.uint8))
    assert np.allclose(arrays["retina_uv"].astype(np.float32), g.retina_uv, atol=1e-3)
    legend = header["retina_type_legend"]
    assert sorted(legend) == legend and [legend[i] for i in arrays["retina_type"]] == list(g.retina_type)
    assert arrays["w_ret"].shape == (g.n_ret, 20) and np.allclose(arrays["w_ret"].astype(np.float32),
                                                                   m.w_ret.detach().numpy(), atol=1e-3)
    assert np.allclose(arrays["b_ret"], m.b_ret.detach().numpy())
    # ionotropic + modulatory CSRs partition the graph's edges; modulatory presynaptic neurons are DA/SER/OCT
    def edges(indptr, indices):
        rows = np.repeat(np.arange(g.n), np.diff(indptr.astype(np.int64)))
        return set(zip(perm[rows].tolist(), perm[indices.astype(np.int64)].tolist()))
    ion, mod = edges(arrays["csr_indptr"], arrays["csr_indices"]), edges(arrays["mod_indptr"], arrays["mod_indices"])
    rows_g = np.repeat(np.arange(g.n), np.diff(g.csr_indptr))
    assert ion.isdisjoint(mod) and ion | mod == set(zip(rows_g.tolist(), g.csr_indices.tolist()))
    assert len(mod) == header["nnz_mod"] == m.nnz_mod and len(ion) == header["nnz"]
    assert all(g.nt_type[pre] in ("DA", "SER", "OCT") for _, pre in mod)
    assert all(g.nt_type[pre] not in ("DA", "SER", "OCT") for _, pre in ion)
    assert np.all(arrays["w_mod"].astype(np.float32) > 0)
    w_can = m.effective_weights().detach().numpy()
    assert np.allclose(arrays["w_mod"].astype(np.float32), w_can[m.mod_edges.numpy()], atol=1e-2)
    assert np.allclose(arrays["w"].astype(np.float32), w_can[m.ion_edges.numpy()], atol=1e-2)
    # central summary
    central = np.flatnonzero(g.super_class == "central")
    assert np.array_equal(arrays["central_idx"], inv[central]) and header["n_central"] == len(central)
    assert arrays["central_w"].shape == (7, len(central)) and arrays["central_b"].shape == (7,)
    assert header["nnz_total"] == g.nnz and header["readout_steps"] == [2, 4]
    by_name = {e["name"]: e for e in header["arrays"]}
    assert by_name["policy_w"]["shape"] == [50, 2 * g.n_out + 7]


def test_export_writes_the_models_input_set_not_the_graphs(graph_full, tmp_path):
    """A checkpoint loaded on a graph rebuilt with different input neurons keeps its own input_idx (persistent
    buffer); the blob must pair w_in with THOSE neurons or the browser runs a different network."""
    g = graph_full
    cfg = BrainConfig(graph_path="toy", steps=4, num_moves=60, value_hidden=8, vision=False)
    torch.manual_seed(3)
    trained = randomise_(FlyBrain(g, cfg)).eval()
    spare = np.setdiff1d(np.flatnonzero(g.super_class == "central"),
                         np.concatenate([g.input_idx, g.output_idx, g.retina_idx]))[:3]
    g2 = dataclasses.replace(g, input_idx=np.sort(np.concatenate([g.input_idx[3:], spare])).astype(np.int32))
    with pytest.warns(UserWarning, match="input_idx differs"):
        m = FlyBrain.from_checkpoint(trained.state_dict(), cfg, g2).eval()
    header = export_web(m, g2, tmp_path)
    arrays, header = read_flyb(tmp_path)
    inv = np.argsort(arrays["node_perm"].astype(np.int64))
    assert np.array_equal(arrays["input_idx"], inv[m.input_idx.numpy()])
    assert np.array_equal(arrays["input_idx"], inv[g.input_idx]) and not np.array_equal(arrays["input_idx"], inv[g2.input_idx])
    assert header["n_in"] == m.n_in and header["n_out"] == m.n_out
    m.round_weights_to_f16_()
    x = (np.random.default_rng(1).random(1280) < 0.25).astype(np.float32)
    policy, value, _ = numpy_forward(arrays, header, x)
    with torch.no_grad():
        p_t, v_t = m(torch.from_numpy(x)[None])
    assert np.allclose(policy, p_t[0].numpy(), atol=1e-2) and abs(value - float(v_t[0, 0])) < 1e-2


def test_vision_only_export_has_empty_w_in(graph_full, tmp_path):
    cfg = BrainConfig(graph_path="toy", steps=4, num_moves=60, value_hidden=8, sensory_input=False)
    m = randomise_(FlyBrain(graph_full, cfg)).eval()
    with pytest.warns(UserWarning, match="sensory_input=false"):
        header = export_web(m, graph_full, tmp_path)
    by_name = {e["name"]: e for e in header["arrays"]}
    assert header["sensory_input"] is False and header["vision"] is True
    assert by_name["w_in"]["shape"] == [0, 1280] and by_name["b_in"]["shape"] == [0]
    assert by_name["input_idx"]["shape"] == [graph_full.n_in] and header["n_in"] == graph_full.n_in
    assert [e["name"] for e in header["arrays"]] == EXPECTED_ORDER
    arrays, header = read_flyb(tmp_path)
    assert arrays["w_in"].shape == (0, 1280)
    m.round_weights_to_f16_()
    x = (np.random.default_rng(2).random(1280) < 0.25).astype(np.float32)
    policy, value, _ = numpy_forward(arrays, header, x)
    with torch.no_grad():
        p_t, v_t = m(torch.from_numpy(x)[None])
    assert np.allclose(policy, p_t[0].numpy(), atol=1e-2) and abs(value - float(v_t[0, 0])) < 1e-2


def test_old_blob_without_feature_arrays_still_runs(model, graph, tmp_path):
    """A blob written before the optional features existed (no retina/mod/central arrays, no header flags)."""
    export_web(model, graph, tmp_path)
    arrays, header = read_flyb(tmp_path)
    for name in FEATURE_ARRAYS:
        arrays.pop(name)
    header["arrays"] = [e for e in header["arrays"] if e["name"] not in FEATURE_ARRAYS]
    for key in FEATURE_HEADER:
        header.pop(key)
    x = np.zeros(1280, np.float32)
    x[17 * 64:18 * 64] = 1.0
    policy, value, _ = numpy_forward(arrays, header, x)
    m = FlyBrain.from_checkpoint(model.state_dict(), model.config, graph).eval().round_weights_to_f16_()
    with torch.no_grad():
        p_t, v_t = m(torch.from_numpy(x)[None])
    assert np.allclose(policy, p_t[0].numpy(), atol=1e-2) and abs(value - float(v_t[0, 0])) < 1e-2


def test_i8_quantisation_with_features(graph_full, tmp_path):
    cfg = BrainConfig(graph_path="toy", steps=3, num_moves=40, value_hidden=0, neuromod=True, central_dim=4)
    m = randomise_(FlyBrain(graph_full, cfg)).eval()
    header = export_web(m, graph_full, tmp_path, quant="i8")
    by_name = {e["name"]: e for e in header["arrays"]}
    assert by_name["w_mod"]["dtype"] == "i8" and isinstance(by_name["w_mod"]["scale"], float)
    assert by_name["w_ret"]["dtype"] == "i8" and len(by_name["w_ret"]["scale"]) == graph_full.n_ret
    assert by_name["central_w"]["dtype"] == "i8" and len(by_name["central_w"]["scale"]) == 4
    arrays, header = read_flyb(tmp_path)
    x = (np.random.default_rng(1).random(1280) < 0.2).astype(np.float32)
    policy, value, _ = numpy_forward(arrays, header, x)
    with torch.no_grad():
        p_t, v_t = m(torch.from_numpy(x)[None])
    assert np.allclose(policy, p_t[0].numpy(), atol=8e-2, rtol=3e-2) and abs(value - float(v_t[0, 0])) < 5e-2


# ---- JS engine parity (needs node; skipped otherwise) -------------------------------------------
REPO = Path(__file__).resolve().parents[1]
JS_ENGINE = REPO / "web" / "engine" / "flybrain.js"
NODE = shutil.which("node")
_JS_RUNNER = """
import {{ readFileSync }} from 'node:fs';
import {{ parseArrays }} from '{loader}';
import {{ FlyBrain }} from '{engine}';
const dir = process.argv[2];
const header = JSON.parse(readFileSync(dir + '/brain.json', 'utf8'));
const buf = readFileSync(dir + '/brain.flyb');
const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
const brain = new FlyBrain({{ header, arrays: parseArrays(header, ab) }});
const xs = JSON.parse(readFileSync(dir + '/x.json', 'utf8'));
const out = xs.map((x) => {{ const r = brain.forward(Float32Array.from(x)); return {{ policy: Array.from(r.policy), value: r.value }}; }});
process.stdout.write(JSON.stringify(out));
"""


@pytest.mark.skipif(NODE is None or not JS_ENGINE.exists(), reason="needs node and web/engine/flybrain.js")
@pytest.mark.parametrize("activation", ["relu", "gelu", "tanh", "satrelu"])
def test_js_engine_matches_numpy_forward(graph, tmp_path, activation):
    """web/engine/flybrain.js == numpy_forward on the exported blob (policy and value), every activation."""
    torch.manual_seed(1)
    cfg = BrainConfig(graph_path="toy", steps=4, num_moves=96, value_hidden=32, activation=activation)
    m = FlyBrain(graph, cfg)
    with torch.no_grad():
        m.bias.normal_(0, 0.3)
        m.leak_logit.normal_(0, 1.0)
        if m.sensory_input:
            m.b_in.normal_(0, 0.5)
        m.value_head[0].bias.normal_(0, 1.0)  # make the value MLP's hidden non-linearity matter
    export_web(m.eval(), graph, tmp_path)
    arrays, header = read_flyb(tmp_path)
    rng = np.random.default_rng(3)
    xs = (rng.random((3, 1280)) < 0.2).astype(np.float32)
    xs[:, 17 * 64:18 * 64] = 1.0
    (tmp_path / "x.json").write_text(json.dumps(xs.tolist()))
    runner = tmp_path / "run.mjs"
    runner.write_text(_JS_RUNNER.format(loader=(REPO / "web/engine/loader.js").as_uri(), engine=JS_ENGINE.as_uri()))
    res = subprocess.run([NODE, str(runner), str(tmp_path)], capture_output=True, text=True, timeout=60, check=False)
    assert res.returncode == 0, res.stderr
    out = json.loads(res.stdout)
    for x, r in zip(xs, out, strict=True):
        policy, value, _ = numpy_forward(arrays, header, x)
        assert np.allclose(np.asarray(r["policy"], np.float32), policy, atol=1e-3), \
            np.abs(np.asarray(r["policy"]) - policy).max()
        assert abs(r["value"] - value) < 1e-3, (r["value"], value)
