"""Tests for flychess.connectome (download / load / graph) on small synthetic data.

The synthetic `Connectome` objects here stand in for the FlyWire tables ONLY in tests: the real
player is always built from the downloaded connectome via `load_connectome` / `build_brain_graph`.
"""
from __future__ import annotations

import gzip
import io
import json
from pathlib import Path

import numpy as np
import pytest

from flychess import paths
from flychess.connectome import (
    NT_SIGN,
    BrainGraph,
    Connectome,
    GraphConfig,
    build_brain_graph,
    edge_signs,
    load_or_build,
    neuron_nt_type,
    toy_graph,
)
from flychess.connectome import download as dl
from flychess.connectome.load import aggregate_connections, load_connectome, parse_connectome

RID0 = 720_575_940_000_000_000


# ---- helpers ---------------------------------------------------------------------------------------
def make_conn(
    n: int,
    edges: list[tuple[int, int, int]],
    super_class: list[str] | None = None,
    nt_type: list[str] | None = None,
    edge_nt: list[str] | None = None,
) -> Connectome:
    """Synthetic connectome: neurons 0..n-1 (root_id = RID0 + i), edges (pre, post, syn_count)."""
    pre = np.array([e[0] for e in edges], dtype=np.int32)
    post = np.array([e[1] for e in edges], dtype=np.int32)
    syn = np.array([e[2] for e in edges], dtype=np.int32)
    order = np.lexsort((post, pre))
    return Connectome(
        root_ids=RID0 + np.arange(n, dtype=np.int64),
        super_class=np.array(super_class or ["central"] * n, dtype=str),
        cell_class=np.array([""] * n, dtype=str),
        cell_type=np.array([f"T{i}" for i in range(n)], dtype=str),
        side=np.array(["left"] * n, dtype=str),
        nt_type=np.array(nt_type or ["ACH"] * n, dtype=str),
        position=np.arange(3 * n, dtype=np.float32).reshape(n, 3),
        pre=pre[order],
        post=post[order],
        syn_count=syn[order],
        neuropil=np.array(["AL_R"] * len(edges), dtype=str),
        edge_nt_type=np.array(edge_nt or ["ACH"] * len(edges), dtype=str)[order],
        sources={"connections.csv.gz": {"size": 1, "mtime": 2.0}},
    )


def dense(g: BrainGraph) -> np.ndarray:
    """(post, pre) dense matrix of signed synapse counts."""
    m = np.zeros((g.n, g.n), dtype=np.float32)
    for post in range(g.n):
        for k in range(g.csr_indptr[post], g.csr_indptr[post + 1]):
            m[post, g.csr_indices[k]] = g.sign[k] * g.syn_count[k]
    return m


def edges_of(g: BrainGraph) -> set[tuple[int, int]]:
    rows = np.repeat(np.arange(g.n), np.diff(g.csr_indptr))
    return {(int(r), int(c)) for r, c in zip(g.root_ids[rows] - RID0, g.root_ids[g.csr_indices] - RID0)}


# ---- load.py: aggregation --------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _no_real_column_table(monkeypatch, tmp_path: Path):
    """Hermetic builds: a synthetic connectome must never pick up the real `column_assignment.csv.gz`
    from data/connectome (the retina would then depend on what the machine has downloaded). The
    builder's "building WITHOUT a retina" warning is expected here and silenced; the retina itself is
    tested with its own column table in tests/test_retina.py."""
    import warnings

    monkeypatch.setattr(paths, "CONNECTOME_DIR", tmp_path / "no-connectome-tables")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*column_assignment.csv.gz not found.*")
        yield


def test_aggregate_sums_and_picks_dominant_row():
    n = 5
    pre = np.array([0, 0, 0, 1, 1, 2, 3, 3], dtype=np.int32)
    post = np.array([1, 1, 1, 0, 0, 4, 2, 2], dtype=np.int32)
    syn = np.array([3, 10, 2, 4, 4, 7, 1, 1], dtype=np.int32)
    nt = np.array(["GABA", "ACH", "GLUT", "GLUT", "GABA", "DA", "SER", "OCT"])
    npil = np.array(["a", "b", "c", "d", "e", "f", "g", "h"])
    p, q, s, e_nt, e_np = aggregate_connections(pre, post, syn, nt, npil, n)
    assert p.tolist() == [0, 1, 2, 3] and q.tolist() == [1, 0, 4, 2]
    assert s.tolist() == [15, 8, 7, 2]
    assert e_nt.tolist() == ["ACH", "GLUT", "DA", "SER"]  # max row; ties -> first in file order
    assert e_np.tolist() == ["b", "d", "f", "g"]
    assert p.dtype == np.int32 and s.dtype == np.int32


def test_aggregate_output_sorted_by_pre_post():
    rng = np.random.default_rng(1)
    n = 30
    pre = rng.integers(0, n, 500).astype(np.int32)
    post = rng.integers(0, n, 500).astype(np.int32)
    syn = rng.integers(1, 20, 500).astype(np.int32)
    nt = rng.choice(["ACH", "GABA"], 500)
    p, q, s, *_ = aggregate_connections(pre, post, syn, nt, nt, n)
    key = p.astype(np.int64) * n + q
    assert np.all(np.diff(key) > 0)
    ref = {}
    for a, b, c in zip(pre, post, syn):
        ref[(a, b)] = ref.get((a, b), 0) + int(c)
    assert {(int(a), int(b)): int(c) for a, b, c in zip(p, q, s)} == ref


# ---- load.py: parsing real-format CSVs written to tmp ----------------------------------------------
def _write_gz(path: Path, text: str) -> None:
    with gzip.open(path, "wt") as f:
        f.write(text)


def test_parse_connectome_from_tiny_csvs(tmp_path: Path):
    a, b, c = RID0 + 1, RID0 + 2, RID0 + 3
    _write_gz(tmp_path / "connections.csv.gz",
              "pre_root_id,post_root_id,neuropil,syn_count,nt_type\n"
              f"{a},{b},AL_R,3,GABA\n{a},{b},AL_L,9,ACH\n{b},{c},LO_R,6,GLUT\n")
    _write_gz(tmp_path / "neurons.csv.gz",
              "root_id,group,nt_type,nt_type_score\n"
              f"{a},X,ACH,0.9\n{b},Y,,0.0\n{RID0 + 9},Z,GABA,0.8\n")  # c missing; +9 isolated
    _write_gz(tmp_path / "classification.csv.gz",
              "root_id,flow,super_class,class,sub_class,hemilineage,side,nerve\n"
              f"{a},afferent,sensory,olfactory,,,left,\n{b},intrinsic,central,,,,right,\n")
    _write_gz(tmp_path / "consolidated_cell_types.csv.gz",
              f"root_id,primary_type,additional_type(s)\n{a},ORN_DA1,\n")
    _write_gz(tmp_path / "coordinates.csv.gz",
              "root_id,position,supervoxel_id\n"
              f"{b},[10 20  30],1\n{b},[99 99 99],2\n{a},[1 2 3],3\n")
    conn = parse_connectome(tmp_path, verbose=False)
    assert conn.root_ids.tolist() == [a, b, c, RID0 + 9]  # union of neurons.csv and endpoints, sorted
    assert conn.super_class.tolist() == ["sensory", "central", "", ""]
    assert conn.cell_type.tolist() == ["ORN_DA1", "", "", ""]
    assert conn.side.tolist() == ["left", "right", "", ""]
    assert conn.nt_type.tolist() == ["ACH", "", "", "GABA"]
    assert conn.position[0].tolist() == [1, 2, 3] and conn.position[1].tolist() == [10, 20, 30]
    assert np.isnan(conn.position[2]).all()
    assert conn.pre.tolist() == [0, 1] and conn.post.tolist() == [1, 2]
    assert conn.syn_count.tolist() == [12, 6]
    assert conn.edge_nt_type.tolist() == ["ACH", "GLUT"] and conn.neuropil.tolist() == ["AL_L", "LO_R"]
    assert set(conn.sources) == {"connections.csv.gz", "neurons.csv.gz", "classification.csv.gz",
                                 "consolidated_cell_types.csv.gz", "coordinates.csv.gz"}
    for arr in (conn.super_class, conn.cell_type, conn.nt_type, conn.neuropil):
        assert arr.dtype.kind == "U"

    # cache round trip through load_connectome
    c1 = load_connectome(tmp_path, cache=True, verbose=False)
    assert (tmp_path / "connectome.npz").exists()
    c2 = load_connectome(tmp_path, cache=True, verbose=False)
    for f in ("root_ids", "super_class", "nt_type", "pre", "post", "syn_count", "edge_nt_type", "neuropil"):
        assert np.array_equal(getattr(c1, f), getattr(c2, f)), f
    assert np.array_equal(c1.position, c2.position, equal_nan=True)
    assert c2.sources == conn.sources


def test_connectome_save_load_roundtrip(tmp_path: Path):
    conn = make_conn(4, [(0, 1, 7), (1, 2, 5)], nt_type=["ACH", "", "GABA", "GLUT"])
    conn.position[3] = np.nan
    conn.save(tmp_path / "c.npz")
    with np.load(tmp_path / "c.npz", allow_pickle=False) as z:  # no pickle needed
        assert z["super_class"].dtype.kind == "U"
    c2 = Connectome.load(tmp_path / "c.npz")
    assert np.array_equal(c2.root_ids, conn.root_ids) and np.array_equal(c2.nt_type, conn.nt_type)
    assert np.array_equal(c2.position, conn.position, equal_nan=True)
    assert np.array_equal(c2.syn_count, conn.syn_count) and c2.sources == conn.sources
    assert c2.index_of([RID0 + 2, RID0]).tolist() == [2, 0]
    with pytest.raises(KeyError):
        c2.index_of([RID0 + 99])


# ---- graph.py: sign rule ---------------------------------------------------------------------------
def test_nt_sign_table():
    assert NT_SIGN == {"ACH": 1, "GABA": -1, "GLUT": -1, "DA": 1, "SER": 1, "OCT": 1, "": 1}


def test_edge_signs_is_per_neuron_only():
    pre_nt = np.array(["ACH", "GABA", "GLUT", "DA", "SER", "OCT", "", "WEIRD"])
    s = edge_signs(pre_nt)
    assert s.tolist() == [1, -1, -1, 1, 1, 1, 1, 1] and s.dtype == np.int8
    assert edge_signs(np.array([], dtype=str)).shape == (0,)
    with pytest.raises(TypeError):  # the per-synapse fallback is gone for good (Dale's law)
        edge_signs(pre_nt, np.array(["GABA"] * 8))


def test_neuron_nt_type_majority_rule():
    #        0 ACH (annotated, edge predictions ignored), 1..4 unannotated, 5 unannotated & no edges
    nt = np.array(["ACH", "", "", "", "", ""])
    pre = np.array([0, 0, 1, 1, 1, 2, 2, 3, 3, 4], dtype=np.int32)
    edge_nt = np.array(["GABA", "GABA", "ACH", "GABA", "GABA", "ACH", "GABA", "", "", "GLUT"])
    syn = np.array([50, 50, 10, 3, 3, 7, 7, 9, 9, 1], dtype=np.int32)
    out = neuron_nt_type(nt, pre, edge_nt, syn)
    assert out.tolist() == ["ACH", "ACH", "ACH", "", "GLUT", ""]
    #  1: ACH 10 > GABA 6 (synapse-weighted, not vote-counted); 2: 7 vs 7 -> alphabetical tie-break;
    #  3: only empty predictions -> stays '' (+1); 4: single GLUT edge; 5: no outgoing edge at all
    assert out.dtype.kind == "U" and nt.tolist() == ["ACH", "", "", "", "", ""]  # input untouched
    assert neuron_nt_type(nt, pre, None, syn).tolist() == nt.tolist()             # no predictions
    assert neuron_nt_type(np.array([], dtype=str), pre[:0], edge_nt[:0], syn[:0]).shape == (0,)


def test_build_sign_uses_presynaptic_neuron():
    # neuron 0 GABA, 1 ACH, 2 unknown (edge says GLUT), 3 unknown (edge says ACH)
    conn = make_conn(
        4, [(0, 1, 5), (1, 0, 5), (2, 1, 5), (3, 1, 5)],
        nt_type=["GABA", "ACH", "", ""], edge_nt=["ACH", "GABA", "GLUT", "ACH"],
    )
    g = build_brain_graph(conn, GraphConfig(max_inputs=0, max_outputs=0))
    m = dense(g)
    assert m[1, 0] == -5 and m[0, 1] == 5 and m[1, 2] == -5 and m[1, 3] == 5
    assert g.meta["nt_fallback"]["unannotated_neurons"] == 2
    assert g.meta["nt_fallback"]["resolved_by_majority"] == 2


def test_build_unannotated_neuron_has_one_sign():
    """Regression: an unannotated neuron with conflicting edge predictions must not get mixed signs."""
    # neuron 0 unannotated: ACH (10 synapses) to 1, GABA (3 synapses) to 2 -> majority ACH -> both +1
    conn = make_conn(
        3, [(0, 1, 10), (0, 2, 3), (1, 2, 5)],
        nt_type=["", "GABA", "ACH"], edge_nt=["ACH", "GABA", "GABA"],
    )
    cfg = GraphConfig(min_syn=1, max_inputs=0, max_outputs=0)
    g = build_brain_graph(conn, cfg)
    m = dense(g)
    assert m[1, 0] == 10 and m[2, 0] == 3 and m[2, 1] == -5
    # the mirror image: GABA carries more synapses -> both outgoing synapses inhibitory
    conn = make_conn(
        3, [(0, 1, 2), (0, 2, 8), (1, 2, 5)],
        nt_type=["", "GABA", "ACH"], edge_nt=["ACH", "GABA", "GABA"],
    )
    m = dense(build_brain_graph(conn, cfg))
    assert m[1, 0] == -2 and m[2, 0] == -8
    # every pre-synaptic neuron of the tiny random graph ends up with a single sign
    rng = np.random.default_rng(7)
    n = 60
    edges = {(int(a), int(b)): int(s) for a, b, s in
             zip(rng.integers(0, n, 600), rng.integers(0, n, 600), rng.integers(5, 40, 600))}
    edges = [(a, b, s) for (a, b), s in edges.items()]
    conn = make_conn(n, edges, nt_type=rng.choice(["ACH", "GABA", ""], n).tolist(),
                     edge_nt=rng.choice(["ACH", "GABA", "GLUT", ""], len(edges)).tolist())
    g = build_brain_graph(conn, GraphConfig(max_inputs=0, max_outputs=0))
    pos = np.bincount(g.csr_indices, weights=g.sign > 0, minlength=g.n)
    neg = np.bincount(g.csr_indices, weights=g.sign < 0, minlength=g.n)
    assert not np.any((pos > 0) & (neg > 0))
    # ... and the sign is the one the per-neuron rule predicts
    nt_neuron = neuron_nt_type(conn.nt_type, conn.pre, conn.edge_nt_type, conn.syn_count)
    kept = (g.root_ids - RID0).astype(int)
    assert np.array_equal(g.sign, edge_signs(nt_neuron[kept][g.csr_indices]))


def test_nt_majority_is_independent_of_subgraph_selection():
    """The label is voted on the FULL edge set, so pruning (min_syn / max_neurons) cannot flip it."""
    # neuron 0 unannotated: ACH 2+2 synapses to 1 and 2 (below min_syn=5), GABA 6 synapses to 3
    conn = make_conn(
        4, [(0, 1, 2), (0, 2, 2), (0, 3, 6), (1, 3, 5), (2, 3, 5)],
        nt_type=["", "ACH", "ACH", "ACH"], edge_nt=["ACH", "ACH", "GABA", "ACH", "ACH"],
    )
    cfg = GraphConfig(min_syn=1, max_inputs=0, max_outputs=0)
    assert dense(build_brain_graph(conn, cfg))[3, 0] == -6  # GABA 6 > ACH 4 on the full set
    cfg = GraphConfig(min_syn=5, max_inputs=0, max_outputs=0)
    assert dense(build_brain_graph(conn, cfg))[3, 0] == -6  # same label after pruning the ACH edges
    # flip the weights: ACH wins on the full set even though only the GABA edge survives min_syn=5
    conn = make_conn(
        4, [(0, 1, 4), (0, 2, 4), (0, 3, 6), (1, 3, 5), (2, 3, 5)],
        nt_type=["", "ACH", "ACH", "ACH"], edge_nt=["ACH", "ACH", "GABA", "ACH", "ACH"],
    )
    g = build_brain_graph(conn, cfg)
    assert dense(g)[3, 0] == 6 and g.meta["nt_fallback"]["resolved_by_majority"] == 1


def test_validate_rejects_mixed_outgoing_signs():
    g = build_brain_graph(make_conn(3, [(0, 1, 5), (0, 2, 5)], nt_type=["ACH", "ACH", "ACH"]),
                          GraphConfig(max_inputs=0, max_outputs=0))
    g.validate()
    g.sign = g.sign.copy()
    g.sign[1] = -1  # neuron 0 now excites 1 and inhibits 2
    with pytest.raises(AssertionError, match="Dale"):
        g.validate()


# ---- graph.py: selection ---------------------------------------------------------------------------
def _role_conn() -> Connectome:
    # 0..3 sensory (inputs), 4..5 central, 6..8 descending (outputs), 9 optic
    sc = ["sensory"] * 4 + ["central"] * 2 + ["descending"] * 3 + ["optic"]
    edges = [
        (0, 4, 5), (0, 5, 5), (0, 6, 5),          # sensory 0: out-degree 3
        (1, 4, 5), (1, 5, 5),                      # sensory 1: out-degree 2
        (2, 4, 5), (2, 5, 5),                      # sensory 2: out-degree 2 (tie with 1 -> root_id)
        (3, 4, 5),                                 # sensory 3: out-degree 1
        (4, 6, 9), (5, 6, 9), (4, 7, 9),           # descending 6: in-degree 3 (incl. 0->6), 7: 1
        (4, 4, 50),                                # self loop, must go
        (9, 4, 5), (4, 9, 5),                      # optic <-> central
        (5, 8, 2),                                 # below min_syn -> descending 8 has degree 0
    ]
    return make_conn(10, edges, super_class=sc)


def test_input_output_selection_deterministic():
    conn = _role_conn()
    cfg = GraphConfig(input_super_classes=("sensory",), max_inputs=2,
                      output_super_classes=("descending",), max_outputs=2)
    g = build_brain_graph(conn, cfg)
    assert (g.root_ids[g.input_idx] - RID0).tolist() == [0, 1]   # 0 (deg 3), then 1 beats 2 by root_id
    assert (g.root_ids[g.output_idx] - RID0).tolist() == [6, 7]  # in-degree 3, 1 (8 has 0)
    # same result on a permuted-edge copy of the connectome
    g2 = build_brain_graph(conn, cfg)
    assert np.array_equal(g.input_idx, g2.input_idx) and np.array_equal(g.csr_indices, g2.csr_indices)
    assert g.meta["cfg"]["max_inputs"] == 2 and g.meta["n_in"] == 2 and g.meta["n_out"] == 2


def test_self_loops_min_syn_and_zero_degree_removal():
    conn = _role_conn()
    g = build_brain_graph(conn, GraphConfig(input_super_classes=("sensory",), max_inputs=4,
                                            output_super_classes=("descending",), max_outputs=4))
    e = edges_of(g)
    assert (4, 4) not in e                      # self loop removed
    assert (8, 5) not in e                      # below min_syn removed
    kept = set((g.root_ids - RID0).tolist())
    assert 8 in kept                            # zero-degree but in the output set -> kept
    assert g.n == 10 and g.meta["nnz"] == 13
    # a zero-degree neuron that has no role is dropped
    conn2 = make_conn(3, [(0, 1, 5), (2, 2, 5)])
    g2 = build_brain_graph(conn2, GraphConfig(max_inputs=0, max_outputs=0))
    assert g2.n == 2 and (g2.root_ids - RID0).tolist() == [0, 1]


def test_region_central_drops_optic():
    conn = _role_conn()
    g = build_brain_graph(conn, GraphConfig(region="central", input_super_classes=("sensory",),
                                            output_super_classes=("descending",)))
    assert "optic" not in set(g.super_class.tolist())
    assert (9, 4) not in edges_of(g) and (4, 9) not in edges_of(g)
    with pytest.raises(ValueError):
        build_brain_graph(conn, GraphConfig(region="nope"))


def test_max_neurons_keeps_roles_and_top_k():
    conn = _role_conn()
    cfg = GraphConfig(max_neurons=5, input_super_classes=("sensory",), max_inputs=1,
                      output_super_classes=("descending",), max_outputs=1)
    g = build_brain_graph(conn, cfg)
    assert g.n <= 5
    ids = set((g.root_ids - RID0).tolist())
    assert {0, 6} <= ids                        # input 0 and output 6 always kept
    assert 4 in ids and 5 in ids                # highest total synapse count among the rest
    g.validate()


def test_csr_is_canonical_and_matches_dense_reference():
    rng = np.random.default_rng(3)
    n = 40
    edges = [(int(a), int(b), int(s)) for a, b, s in
             zip(rng.integers(0, n, 300), rng.integers(0, n, 300), rng.integers(1, 30, 300))]
    # de-duplicate (pre, post) the way load.py would
    uniq = {}
    for a, b, s in edges:
        uniq[(a, b)] = uniq.get((a, b), 0) + s
    edges = [(a, b, s) for (a, b), s in uniq.items()]
    nt = rng.choice(["ACH", "GABA", "GLUT", ""], n).tolist()
    conn = make_conn(n, edges, nt_type=nt)
    g = build_brain_graph(conn, GraphConfig(min_syn=5, max_inputs=0, max_outputs=0))
    g.validate()
    # entries sorted by (post, pre)
    rows = np.repeat(np.arange(g.n), np.diff(g.csr_indptr))
    key = rows.astype(np.int64) * g.n + g.csr_indices
    assert np.all(np.diff(key) > 0)
    # dense reference
    ref = np.zeros((n, n), dtype=np.float32)
    for a, b, s in edges:
        if a != b and s >= 5:
            ref[b, a] = NT_SIGN[nt[a]] * s
    kept = (g.root_ids - RID0).astype(int)
    assert np.array_equal(dense(g), ref[np.ix_(kept, kept)])
    assert g.csr_indptr.dtype == np.int32 and g.csr_indices.dtype == np.int32
    assert g.syn_count.dtype == np.float32 and g.sign.dtype == np.int8


def test_meta_contents_and_graph_roundtrip(tmp_path: Path):
    conn = _role_conn()
    cfg = GraphConfig(input_super_classes=("sensory",), output_super_classes=("descending",), name="t")
    g = build_brain_graph(conn, cfg)
    for k in ("cfg", "super_class_counts", "n", "nnz", "n_in", "n_out", "sources", "build_time_s",
              "sign_counts", "built_at"):
        assert k in g.meta, k
    assert g.meta["super_class_counts"]["sensory"] == 4
    assert g.meta["sign_counts"]["excitatory"] + g.meta["sign_counts"]["inhibitory"] == g.nnz
    assert g.meta["sources"] == conn.sources
    p = g.save(tmp_path / "g.npz")
    g2 = BrainGraph.load(p)
    g2.validate()
    for f in ("root_ids", "csr_indptr", "csr_indices", "syn_count", "sign", "input_idx", "output_idx",
              "super_class", "position"):
        assert np.array_equal(getattr(g, f), getattr(g2, f)), f
    assert g2.meta == json.loads(json.dumps(g.meta))


def test_load_or_build_caches(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(paths, "BRAIN_DIR", tmp_path)
    conn = _role_conn()
    cfg = GraphConfig(input_super_classes=("sensory",), output_super_classes=("descending",), name="unit")
    g = load_or_build(cfg, conn=conn, verbose=False)
    assert (tmp_path / "unit.npz").exists()
    calls = []

    def spy(*a, **k):
        calls.append(1)
        return build_brain_graph(*a, **k)

    monkeypatch.setattr("flychess.connectome.graph.build_brain_graph", spy)
    g2 = load_or_build(cfg, verbose=False)  # served from disk: neither connectome nor build touched
    assert not calls and np.array_equal(g.csr_indices, g2.csr_indices)
    g3 = load_or_build(cfg, out_path=tmp_path / "explicit.npz", conn=conn, verbose=False)
    assert calls == [1] and (tmp_path / "explicit.npz").exists()  # explicit path missing -> built
    assert np.array_equal(g3.csr_indices, g.csr_indices)


# ---- download.py -----------------------------------------------------------------------------------
class _FakeResponse:
    def __init__(self, payload: bytes, status: int = 200):
        self.payload, self.status_code = payload, status
        self.headers = {"Content-Length": str(len(payload))}

    def iter_content(self, chunk_size: int):
        buf = io.BytesIO(self.payload)
        while chunk := buf.read(chunk_size):
            yield chunk

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_verify_gzip(tmp_path: Path):
    good = tmp_path / "ok.csv.gz"
    _write_gz(good, "root_id\n1\n" * 1000)
    assert dl.verify_gzip(good)
    bad = tmp_path / "bad.csv.gz"
    bad.write_bytes(good.read_bytes()[:-20])
    assert not dl.verify_gzip(bad)
    assert not dl.verify_gzip(tmp_path / "missing.gz")


def test_download_connectome_skips_valid_and_replaces_corrupt(tmp_path: Path, monkeypatch, capsys):
    payload = gzip.compress(b"pre_root_id,post_root_id\n1,2\n")
    urls = []

    def fake_get(url, stream, timeout):
        urls.append(url)
        return _FakeResponse(payload)

    monkeypatch.setattr(dl.requests, "get", fake_get)
    (tmp_path / "neurons.csv.gz").write_bytes(payload)          # valid -> skipped
    (tmp_path / "connections.csv.gz").write_bytes(b"garbage")   # corrupt -> re-downloaded
    out = dl.download_connectome(tmp_path, files=("connections.csv.gz", "neurons.csv.gz", "coordinates.csv.gz"))
    assert [p.name for p in out] == ["connections.csv.gz", "neurons.csv.gz", "coordinates.csv.gz"]
    assert urls == [dl.connectome_url("connections.csv.gz"), dl.connectome_url("coordinates.csv.gz")]
    assert urls[0].startswith("https://storage.googleapis.com/flywire-data/codex/data/fafb/783/")
    assert all(dl.verify_gzip(p) for p in out)
    assert not list(tmp_path.glob("*.partial"))
    assert "skipping" in capsys.readouterr().out


def test_download_failure_prints_manual_instructions(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setattr(dl.requests, "get", lambda url, stream, timeout: _FakeResponse(b"", status=404))
    with pytest.raises(RuntimeError):
        dl.download_games(["2014-01"], dest=tmp_path)
    out = capsys.readouterr().out
    assert "manually" in out and "lichess_db_standard_rated_2014-01.pgn.zst" in out
    assert dl.games_url("2014-01") == "https://database.lichess.org/standard/lichess_db_standard_rated_2014-01.pgn.zst"
    assert not (tmp_path / "lichess_db_standard_rated_2014-01.pgn.zst").exists()


def test_download_rejects_corrupt_stream(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(dl.requests, "get", lambda url, stream, timeout: _FakeResponse(b"not gzip"))
    with pytest.raises(RuntimeError):
        dl.download_file(dl.connectome_url("x.csv.gz"), tmp_path / "x.csv.gz")
    assert not (tmp_path / "x.csv.gz").exists() and not (tmp_path / "x.csv.gz.partial").exists()


# ---- real data (slow) ------------------------------------------------------------------------------
@pytest.mark.slow
def test_real_full_graph_validates():
    p = paths.BRAIN_DIR / "full.npz"
    if not p.exists():
        pytest.skip("data/brain/full.npz not built (run `fly build-brain`)")
    g = BrainGraph.load(p)
    if "nt_fallback" not in g.meta:
        pytest.skip("data/brain/full.npz was built before the per-neuron nt rule (mixed outgoing signs); "
                    "rebuild with `fly build-brain --force` once no run depends on it")
    g.validate()
    assert 130_000 <= g.n <= 140_000 and 2_600_000 <= g.nnz <= 2_800_000
    assert g.n_in == 2048 and g.n_out >= 1000
    assert g.meta["cfg"]["region"] == "full" and g.meta["cfg"]["max_neurons"] is None
    assert set(g.meta["super_class_counts"]) >= {"optic", "central", "sensory", "descending", "motor"}
    assert 0.3 < g.meta["sign_counts"]["inhibitory"] / g.nnz < 0.5
    assert not np.isnan(g.position).any()


def test_summary_distinguishes_connections_from_synapses():
    """README/SPEC vocabulary: a CSR non-zero is a pre->post *connection*; synapses are sum(syn_count)."""
    g = toy_graph(n=50, nnz=300, n_in=4, n_out=4, seed=1)
    text = g.summary()
    assert f"connections={g.nnz:,}" in text
    assert f"synapses={int(g.syn_count.sum()):,}" in text
    assert int(g.syn_count.sum()) > g.nnz  # every connection carries >= 5 synapses, so the two must differ


# ---- v3 optional fields (nt_type / retina, see tests/test_retina.py for the retina itself) ----------
def test_build_populates_nt_type_and_empty_retina_without_column_table(tmp_path: Path, monkeypatch):
    """A synthetic connectome has no column assignments: the graph gets `nt_type` (per-neuron label,
    origin of the signs) but an empty retina, and both survive a save/load round trip. The test is
    hermetic: `paths.CONNECTOME_DIR` points at an empty directory (autouse fixture) so the real
    `column_assignment.csv.gz` (if downloaded) is never read, and the builder must warn about it."""
    conn = make_conn(4, [(0, 1, 5), (1, 0, 5), (2, 1, 5), (3, 1, 5)],
                     nt_type=["GABA", "ACH", "", ""], edge_nt=["ACH", "GABA", "GLUT", "ACH"])
    with pytest.warns(UserWarning, match="column_assignment.csv.gz not found"):
        g = build_brain_graph(conn, GraphConfig(max_inputs=0, max_outputs=0))
    assert g.has_nt_type and g.nt_type.tolist() == ["GABA", "ACH", "GLUT", "ACH"]
    assert np.array_equal(g.sign, edge_signs(g.nt_type[g.csr_indices]))
    assert not g.has_retina and g.n_ret == 0 and g.meta["n_ret"] == 0
    assert g.meta["retina"] == {"enabled": False, "n_ret": 0, "reason": "column_assignment.csv.gz missing"}
    g2 = BrainGraph.load(g.save(tmp_path / "g.npz"))
    g2.validate()
    assert g2.nt_type.tolist() == g.nt_type.tolist() and not g2.has_retina
    assert "retina" not in g2.summary()
    # retina=False skips the column table entirely (no warning)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        g3 = build_brain_graph(conn, GraphConfig(max_inputs=0, max_outputs=0, retina=False))
    assert g3.meta["retina"] == {"enabled": False, "n_ret": 0, "reason": "disabled by GraphConfig.retina=False"}
    # GraphConfig round-trips through to_dict with the new options
    d = GraphConfig(retina_max=256).to_dict()
    assert d["retina"] is True and d["retina_max"] == 256 and d["retina_field"] == "split"
