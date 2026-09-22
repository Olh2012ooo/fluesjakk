"""Tests for the fly's retina (flychess/connectome/retina.py, docs/RETINA.md).

A synthetic connectome + column table stand in for the FlyWire tables ONLY in tests; the real
player is always built from the downloaded connectome and `column_assignment.csv.gz`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from flychess import paths
from flychess.connectome import (
    BrainGraph,
    ColumnTable,
    Connectome,
    GraphConfig,
    build_brain_graph,
    build_eye_map,
    build_retina,
    edge_signs,
    hex_to_xy,
    retina_candidates,
    toy_graph,
)
from flychess.connectome import retina as ret

RID0 = 720_575_940_000_000_000
RANGE = 3  # hex rhombus p, q in [-RANGE, RANGE] -> 49 columns per eye


# ---- fixture: two small hexagonal eyes wired to a lamina ---------------------------------------------
def make_columns(n_range: int = RANGE, eyes=("left", "right")) -> tuple[ColumnTable, dict]:
    """Column table with (2*n_range+1)^2 columns per eye; per column an R7, an R8 and an L1 neuron.

    Returns the table and a dict `cells[(eye, column_id, type)] -> neuron index` (indices are
    allocated left eye first, then right, column by column, R7/R8/L1) so tests can build edges.
    """
    rows = []
    cells = {}
    nxt = 0
    for eye in eyes:
        cid = 0
        for p in range(-n_range, n_range + 1):
            for q in range(-n_range, n_range + 1):
                cid += 1
                for t in ("R7", "R8", "L1"):
                    rows.append((RID0 + nxt, eye, t, cid, p, q))
                    cells[(eye, cid, t)] = nxt
                    nxt += 1
    r = np.array(rows, dtype=object)
    table = ColumnTable.from_arrays(r[:, 0].astype(np.int64), r[:, 1], r[:, 2], r[:, 3].astype(np.int32),
                                    r[:, 4].astype(np.int16), r[:, 5].astype(np.int16), source={"path": "synthetic"})
    return table, cells


def make_conn(n: int, edges: list[tuple[int, int, int]], cell_type: list[str], super_class: list[str],
              side: list[str] | None = None, nt_type: list[str] | None = None) -> Connectome:
    pre = np.array([e[0] for e in edges], dtype=np.int32)
    post = np.array([e[1] for e in edges], dtype=np.int32)
    syn = np.array([e[2] for e in edges], dtype=np.int32)
    order = np.lexsort((post, pre))
    return Connectome(
        root_ids=RID0 + np.arange(n, dtype=np.int64),
        super_class=np.array(super_class, dtype=str),
        cell_class=np.array([""] * n, dtype=str),
        cell_type=np.array(cell_type, dtype=str),
        side=np.array(side or [""] * n, dtype=str),
        nt_type=np.array(nt_type or ["ACH"] * n, dtype=str),
        position=np.arange(3 * n, dtype=np.float32).reshape(n, 3),
        pre=pre[order], post=post[order], syn_count=syn[order],
        neuropil=np.array(["ME_R"] * len(edges), dtype=str),
        edge_nt_type=np.array(["ACH"] * len(edges), dtype=str)[order],
        sources={"connections.csv.gz": {"size": 1, "mtime": 2.0}},
    )


def eye_fixture(n_r16_per_column: int = 2, isolated_r7: int = 1, extra_sensory: int = 6,
                sparse_left: bool = False, filler: int = 0):
    """A connectome whose every column has R7, R8, L1 (+ `n_r16_per_column` R1-6 wired to the L1),
    a central layer, descending outputs, some generic sensory neurons and `isolated_r7` R7s with a
    column but no connection at all. `sparse_left`: only the first third of the left eye's columns
    get R1-6 cells (uneven photoreceptor density). `filler`: that many central neurons with huge
    synapse counts among themselves (and onto the outputs), which outrank every lamina cell in a
    `max_neurons` selection — the real brain's situation, where no L1 makes it into a 2000-neuron graph."""
    table, cells = make_columns()
    n_col = len(table.root_ids) // 3 // 2
    n = len(table.root_ids)
    cell_type = list(table.cell_type)
    super_class = ["sensory" if t in ("R7", "R8") else "optic" for t in cell_type]
    side = list(table.hemisphere)
    edges = []
    r16 = {}
    for eye in ("left", "right"):
        for cid in range(1, n_col + 1):
            l1 = cells[(eye, cid, "L1")]
            for t in ("R7", "R8"):
                edges.append((cells[(eye, cid, t)], l1, 6))  # R7/R8 -> L1 (keeps them connected)
            for k in range(n_r16_per_column if not (sparse_left and eye == "left" and cid > n_col // 3) else 0):
                r = n
                n += 1
                cell_type.append("R1-6")
                super_class.append("sensory")
                side.append(eye)
                r16[(eye, cid, k)] = r
                edges.append((r, l1, 20))
                # a weaker connection to a neighbouring column's L1 must not win the majority
                other = cells[(eye, cid % n_col + 1, "L1")]
                edges.append((r, other, 5))
    # central + descending + generic sensory
    central = list(range(n, n + 4))
    n += 4
    cell_type += [f"C{i}" for i in range(4)]
    super_class += ["central"] * 4
    side += ["center"] * 4
    desc = list(range(n, n + 3))
    n += 3
    cell_type += [f"DN{i}" for i in range(3)]
    super_class += ["descending"] * 3
    side += ["center"] * 3
    sens = list(range(n, n + extra_sensory))
    n += extra_sensory
    cell_type += [f"ORN{i}" for i in range(extra_sensory)]
    super_class += ["sensory"] * extra_sensory
    side += ["left"] * extra_sensory
    for eye in ("left", "right"):
        for cid in range(1, n_col + 1):
            edges.append((cells[(eye, cid, "L1")], central[cid % 4], 8))
    for c in central:
        for d in desc:
            edges.append((c, d, 10))
    for i, s in enumerate(sens):
        edges.append((s, central[i % 4], 7 + i))
    fill = list(range(n, n + filler))
    n += filler
    cell_type += [f"BIG{i}" for i in range(filler)]
    super_class += ["central"] * filler
    side += ["center"] * filler
    for i, f in enumerate(fill):
        edges.append((f, fill[(i + 1) % filler], 500))
        edges.append((f, desc[i % 3], 100))
    # R7s with a column assignment but no synapse >= min_syn (cannot drive anything -> not retina)
    iso = []
    for k in range(isolated_r7):
        eye, cid = "left", k + 1
        # replace that column's R7 edge by a sub-threshold one
        r7 = cells[(eye, cid, "R7")]
        edges = [e for e in edges if e[0] != r7] + [(r7, cells[(eye, cid, "L1")], 2)]
        iso.append(r7)
    conn = make_conn(n, edges, cell_type, super_class, side)
    return conn, table, cells, r16, {"central": central, "desc": desc, "sens": sens, "isolated": iso, "n_col": n_col,
                                     "filler": fill}


def cfg_for(**kw) -> GraphConfig:
    base = {"min_syn": 5, "max_inputs": 4, "max_outputs": 8, "input_super_classes": ("sensory",),
            "output_super_classes": ("descending",)}
    base.update(kw)
    return GraphConfig(**base)


# ---- geometry --------------------------------------------------------------------------------------
def test_hex_to_xy_is_a_regular_lattice():
    X0, Y0 = hex_to_xy(np.array([0]), np.array([0]))
    for dp, dq in [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, -1)]:
        X, Y = hex_to_xy(np.array([dp]), np.array([dq]))
        assert np.hypot(X - X0, Y - Y0)[0] == pytest.approx(1.0), (dp, dq)
    # (1, -1) is NOT a neighbour on this lattice (it is sqrt(3) away)
    X, Y = hex_to_xy(np.array([1]), np.array([-1]))
    assert np.hypot(X - X0, Y - Y0)[0] == pytest.approx(np.sqrt(3))
    # +Y is the p+q direction, +X the q-p direction
    assert hex_to_xy(2, 2) == (pytest.approx(0.0), pytest.approx(2.0))
    assert hex_to_xy(-1, 1)[0] == pytest.approx(np.sqrt(3))


def test_quantile_bins_balanced_and_deterministic():
    rng = np.random.default_rng(0)
    u = np.repeat(np.linspace(0, 1, 10), 5)  # many ties
    v = rng.random(50)
    ids = np.arange(50)
    b = ret.quantile_bins(u, v, ids, 4)
    assert sorted(np.bincount(b, minlength=4).tolist()) == [12, 12, 13, 13]
    assert np.array_equal(b, ret.quantile_bins(u, v, ids, 4))
    perm = rng.permutation(50)  # same items in another order -> same bin per id
    assert np.array_equal(ret.quantile_bins(u[perm], v[perm], ids[perm], 4)[np.argsort(perm)], b)
    assert np.all(u[b == 0] <= u[b == 3].min())  # monotone in the primary key
    assert ret.quantile_bins(np.zeros(0), np.zeros(0), np.zeros(0), 4).shape == (0,)
    # weighted: bins balanced in total weight, not in count
    w = np.where(np.arange(50) < 10, 10.0, 1.0)
    bw = ret.quantile_bins(np.arange(50.0), np.zeros(50), ids, 4, w)
    tot = np.bincount(bw, weights=w, minlength=4)
    assert tot.max() - tot.min() <= 10 and np.bincount(bw, minlength=4)[0] < 13
    assert np.array_equal(ret.quantile_bins(np.arange(50.0), np.zeros(50), ids, 4, np.zeros(50)),
                          ret.quantile_bins(np.arange(50.0), np.zeros(50), ids, 4))  # zero weight -> plain
    assert ret.uniform_bins(np.array([0.0, 0.49, 0.5, 1.0]), 2).tolist() == [0, 0, 1, 1]


def test_assign_squares_fields_and_binning():
    rng = np.random.default_rng(1)
    u, v = rng.random(200), rng.random(200)
    ids = np.arange(200)
    left = ret.assign_squares(u, v, ids, "left", "split", "quantile")
    right = ret.assign_squares(u, v, ids, "right", "split", "quantile")
    assert np.all(left % 8 < 4) and np.all(right % 8 >= 4)
    assert np.array_equal(right, left + 4)
    counts = np.bincount(left, minlength=64)[ret.field_squares("left", "split")]
    assert counts.min() >= 6 and counts.max() <= 7 and counts.sum() == 200  # 200 / 32 = 6.25
    full = ret.assign_squares(u, v, ids, "right", "full", "quantile")
    assert set(full % 8) == set(range(8)) and np.bincount(full, minlength=64).min() >= 3
    uni = ret.assign_squares(u, v, ids, "left", "split", "uniform")
    assert np.array_equal(uni, (np.minimum((v * 8).astype(int), 7) * 8 + np.minimum((u * 4).astype(int), 3)))
    # geometry: a column low & lateral in the left eye ends up on a1-ish, high & frontal near d8
    assert ret.assign_squares([0.0, 1.0], [0.0, 1.0], [0, 1], "left", "split", "uniform").tolist() == [0, 59]
    assert ret.assign_squares([0.0, 1.0], [0.0, 1.0], [0, 1], "right", "split", "uniform").tolist() == [4, 63]
    with pytest.raises(ValueError):
        ret.assign_squares(u, v, ids, "left", "nope", "quantile")
    with pytest.raises(ValueError):
        ret.assign_squares(u, v, ids, "left", "split", "nope")
    with pytest.raises(ValueError):
        ret.assign_squares(u, v, ids, "middle", "split", "quantile")


def test_eye_map_orientation_and_normalisation():
    table, _ = make_columns()
    left = build_eye_map(table, "left", "split", "quantile")
    right = build_eye_map(table, "right", "split", "quantile")
    assert left.n == right.n == (2 * RANGE + 1) ** 2
    X, Y = hex_to_xy(left.p, left.q)
    # v follows +Y (dorsal); u_board follows +X for the left eye and -X for the right eye
    assert np.corrcoef(left.v, Y)[0, 1] > 0.999 and np.corrcoef(right.v, Y)[0, 1] > 0.999
    assert np.corrcoef(left.u_board, X)[0, 1] > 0.999 and np.corrcoef(right.u_board, X)[0, 1] < -0.999
    for em in (left, right):
        assert em.u_board.min() == 0 and em.u_board.max() == 1 and em.v.min() == 0 and em.v.max() == 1
        assert em.uv.dtype == np.float32 and np.all((em.uv >= 0) & (em.uv <= 1))
    assert left.uv[:, 0].max() < 0.5 and right.uv[:, 0].min() >= 0.5
    assert np.array_equal(np.sort(left.column_id), left.column_id)  # unique, sorted columns
    # the frontal-most columns of the two eyes meet at the middle of the eye map
    assert abs(left.uv[np.argmax(left.u_board), 0] - 0.5) < 0.002
    assert right.uv[np.argmin(right.u_board), 0] == pytest.approx(0.5)
    assert build_eye_map(table, "right", "split").square.min() % 8 >= 4
    empty = build_eye_map(table, "center")
    assert empty.n == 0 and empty.uv.shape == (0, 2)


# ---- photoreceptors -> columns --------------------------------------------------------------------
def test_retina_candidates_direct_and_partner_placement():
    conn, table, cells, r16, extra = eye_fixture()
    cand = retina_candidates(conn, None, table)
    n_col = extra["n_col"]
    st = cand.stats
    # every R7 / R8 is placed directly (also the isolated one: candidates ignore degree), R1-6 via L1
    assert st["per_type"]["R7"] == {"connectome": 2 * n_col, "direct": 2 * n_col, "via_partner": 0, "unplaced": 0,
                                    "inactive": 0}
    assert st["per_type"]["R8"]["direct"] == 2 * n_col
    assert st["per_type"]["R1-6"] == {"connectome": 4 * n_col, "direct": 0, "via_partner": 4 * n_col, "unplaced": 0,
                                      "inactive": 0}
    assert cand.n == 8 * n_col and np.all(np.diff(cand.idx) > 0) and st["placed"] == cand.n
    # `active` restricts the retina (and the weighting) without changing any placement
    active = np.ones(conn.n, dtype=bool)
    active[cand.idx[:10]] = False
    sub = retina_candidates(conn, None, table, active=active)
    assert sub.n == cand.n - 10 and sub.stats["placed"] == cand.n and np.array_equal(sub.idx, cand.idx[10:])
    assert np.array_equal(sub.column_id, cand.column_id[10:]) and np.array_equal(sub.uv, cand.uv[10:])
    assert sum(v["inactive"] for v in sub.stats["per_type"].values()) == 10
    assert st["columns_per_eye"] == {"left": n_col, "right": n_col} and st["side_mismatch"] == 0
    # an R1-6 sits on the column of its STRONGEST L1 partner (20 synapses beat 5 to the neighbour)
    lookup = {int(i): k for k, i in enumerate(cand.idx)}
    for (eye, cid, _k), r in r16.items():
        k = lookup[r]
        assert cand.column_id[k] == cid and cand.eye[k] == (eye == "right") and cand.via_partner[k]
        assert cand.square[k] == cand.square[lookup[cells[(eye, cid, "R7")]]]
    # R7 placed directly: same square as the R8 of its column, not via partner
    for eye in ("left", "right"):
        for cid in (1, n_col):
            a, b = lookup[cells[(eye, cid, "R7")]], lookup[cells[(eye, cid, "R8")]]
            assert cand.square[a] == cand.square[b] and not cand.via_partner[a]
            assert np.array_equal(cand.uv[a], cand.uv[b])
    # split field: left eye on files a-d, right eye on e-h; every square of both blocks used
    assert np.all(cand.square[cand.eye == 0] % 8 < 4) and np.all(cand.square[cand.eye == 1] % 8 >= 4)
    assert st["photoreceptors_per_square"]["empty_squares"] == 0 if "photoreceptors_per_square" in st else True
    assert len(np.unique(cand.square)) == 64
    assert cand.cell_type.dtype.kind == "U" and cand.square.dtype == np.int8 and cand.eye.dtype == np.int8


def test_partner_majority_is_synapse_weighted_with_deterministic_ties():
    conn, table, cells, _r16, extra = eye_fixture(n_r16_per_column=0)
    n = conn.n
    # new R1-6 cells: a) two partners 9 vs 10 synapses (10 wins even if it is the 'later' column),
    # b) exact tie -> the smaller column id, c) only a non-partner target -> unplaced
    edges = list(zip(conn.pre.tolist(), conn.post.tolist(), conn.syn_count.tolist()))
    ct, sc, sd = conn.cell_type.tolist(), conn.super_class.tolist(), conn.side.tolist()
    a, b, c = n, n + 1, n + 2
    ct += ["R1-6"] * 3
    sc += ["sensory"] * 3
    sd += ["right"] * 3
    edges += [(a, cells[("right", 3, "L1")], 9), (a, cells[("right", 7, "L1")], 10),
              (b, cells[("right", 9, "L1")], 6), (b, cells[("right", 4, "L1")], 6),
              (c, extra["central"][0], 30)]
    conn2 = make_conn(n + 3, edges, ct, sc, sd)
    cand = retina_candidates(conn2, None, table)
    lookup = {int(i): k for k, i in enumerate(cand.idx)}
    assert cand.column_id[lookup[a]] == 7 and cand.column_id[lookup[b]] == 4
    assert c not in lookup and cand.stats["per_type"]["R1-6"]["unplaced"] == 1
    # side mismatch is only counted, never used: an R1-6 labelled 'left' wired to the right eye
    sd[a] = "left"
    cand = retina_candidates(make_conn(n + 3, edges, ct, sc, sd), None, table)
    assert cand.stats["side_mismatch"] == 1 and cand.eye[lookup[a]] == 1


def test_retina_options_eyes_field_types_and_partners():
    conn, table, _cells, _r16, extra = eye_fixture()
    n_col = extra["n_col"]
    # 49 columns per eye cannot cover the 64 squares of a 'full' field: the builder warns about it
    with pytest.warns(UserWarning, match="15 square.s. of the left eye's field receive no column"):
        only_left = retina_candidates(conn, GraphConfig(retina_eyes="left"), table)
    assert np.all(only_left.eye == 0) and only_left.n == 4 * n_col
    assert only_left.stats["options"]["effective_field"] == "full"  # a single eye sees the whole board
    assert set(only_left.square % 8) == set(range(8)) and set(only_left.square // 8) == set(range(8))
    assert only_left.stats["columns_per_square"]["left"]["empty_squares"] == 15
    with pytest.warns(UserWarning, match="right eye"):
        full = retina_candidates(conn, GraphConfig(retina_field="full"), table)
    assert set(full.square[full.eye == 0] % 8) == set(range(8)) and set(full.square[full.eye == 1] % 8) == set(range(8))
    no_r16 = retina_candidates(conn, GraphConfig(retina_types=("R7", "R8")), table)
    assert no_r16.n == 4 * n_col and "R1-6" not in no_r16.stats["per_type"]
    # R1-6 cannot be placed without a lamina partner type in the table
    none = retina_candidates(conn, GraphConfig(retina_partner_types=("L2",)), table)
    assert none.stats["per_type"]["R1-6"]["unplaced"] == 4 * n_col and none.n == 4 * n_col
    for bad in ({"retina_eyes": "middle"}, {"retina_field": "half"}, {"retina_binning": "random"}):
        with pytest.raises(ValueError):
            retina_candidates(conn, GraphConfig(**bad), table)


def test_binning_modes_on_the_fixture():
    conn, table, *_ = eye_fixture(n_r16_per_column=3)
    q = retina_candidates(conn, GraphConfig(retina_binning="quantile"), table)
    cps = q.stats["columns_per_square"]
    for eye in ("left", "right"):
        assert cps[eye]["max"] - cps[eye]["min"] <= 1 and cps[eye]["empty_squares"] == 0
    w = retina_candidates(conn, GraphConfig(retina_binning="weighted"), table)
    # uniform photoreceptor density here -> weighted == quantile up to rounding, both balanced
    assert w.stats["photoreceptors_per_square"]["empty_squares"] == 0
    assert w.stats["photoreceptors_per_square"]["max"] - w.stats["photoreceptors_per_square"]["min"] <= 5
    with pytest.warns(UserWarning, match="receive no column"):  # a round eye leaves corner squares empty
        u = retina_candidates(conn, GraphConfig(retina_binning="uniform"), table)
    assert u.stats["options"]["binning"] == "uniform"
    assert u.stats["columns_per_square"]["left"]["empty_squares"] == 7
    # weighted binning actually balances photoreceptors when the density is uneven (R1-6 only in a
    # third of the left eye's columns): fewer photoreceptors spread, more columns per square
    conn2, table2, *_ = eye_fixture(n_r16_per_column=3, sparse_left=True)
    qs = retina_candidates(conn2, GraphConfig(retina_binning="quantile"), table2).stats
    ws = retina_candidates(conn2, GraphConfig(retina_binning="weighted"), table2).stats

    def spread(st):
        left = st["photoreceptors_per_square_per_eye"]["left"]
        return left["max"] - left["min"]

    assert spread(ws) < spread(qs) and spread(qs) >= 3
    assert ws["columns_per_square"]["left"]["max"] > qs["columns_per_square"]["left"]["max"]
    assert ws["columns_per_square"]["left"]["empty_squares"] == 0


# ---- graph integration ---------------------------------------------------------------------------
def test_build_brain_graph_with_retina():
    conn, table, _cells, _r16, extra = eye_fixture()
    n_col = extra["n_col"]
    g = build_brain_graph(conn, cfg_for(), columns=table)
    g.validate()
    assert g.has_retina and g.n_ret == 8 * n_col - 1  # the isolated R7 has no connection -> not retina
    assert g.meta["n_ret"] == g.n_ret and g.meta["retina"]["enabled"]
    kept = (g.root_ids - RID0).astype(int)
    assert extra["isolated"][0] not in set(kept.tolist())  # ... and, having no role, is dropped
    # retina neurons are the photoreceptors, excluded from the generic inputs, which are the ORNs
    assert set(conn.cell_type[kept[g.retina_idx]]) == {"R1-6", "R7", "R8"}
    assert not np.isin(g.retina_idx, g.input_idx).any()
    assert set(conn.cell_type[kept[g.input_idx]]) <= {f"ORN{i}" for i in range(6)} and g.n_in == 4
    assert np.all(np.diff(g.retina_idx) > 0)
    # arrays are consistent with the candidates (restricted to the connected photoreceptors, which is
    # what the builder does: the weighted binning is computed on the neurons that end up in the graph)
    active = np.zeros(conn.n, dtype=bool)
    active[kept] = True
    cand = retina_candidates(conn, cfg_for(), table, active=active)
    lookup = {int(i): k for k, i in enumerate(cand.idx)}
    for j, gi in enumerate(g.retina_idx):
        k = lookup[int(kept[gi])]
        assert g.retina_square[j] == cand.square[k] and g.retina_eye[j] == cand.eye[k]
        assert g.retina_type[j] == cand.cell_type[k] and np.array_equal(g.retina_uv[j], cand.uv[k])
    assert g.retina_square.dtype == np.int8 and g.retina_uv.dtype == np.float32 and g.retina_eye.dtype == np.int8
    assert g.retina_idx.dtype == np.int32 and g.retina_type.dtype.kind == "U"
    # nt_type is populated (one label per neuron) and is the origin of the signs
    assert g.has_nt_type and set(g.nt_type) == {"ACH"} and np.array_equal(g.sign, edge_signs(g.nt_type[g.csr_indices]))
    m = g.meta["retina"]
    assert m["n_ret"] == g.n_ret and m["types"] == {"R1-6": 4 * n_col, "R7": 2 * n_col - 1, "R8": 2 * n_col}
    assert m["eyes"] == {"left": 4 * n_col - 1, "right": 4 * n_col}
    assert m["columns_per_eye"] == {"left": n_col, "right": n_col} and m["source"] == {"path": "synthetic"}
    assert m["photoreceptors_per_square"]["empty_squares"] == 0
    assert m["options"]["binning"] == "weighted" and m["options"]["field"] == "split"
    assert "retina" in g.summary() and "R7=" in g.summary()
    # deterministic
    g2 = build_brain_graph(conn, cfg_for(), columns=table)
    for f in ("root_ids", "csr_indices", "input_idx", "retina_idx", "retina_square", "retina_uv", "nt_type"):
        assert np.array_equal(getattr(g, f), getattr(g2, f)), f


def test_build_retina_post_hoc_matches_and_caps():
    conn, table, *_ = eye_fixture()
    g = build_brain_graph(conn, cfg_for(), columns=table)
    r = build_retina(conn, g, cfg_for(), columns=table)
    for k in ret.RETINA_KEYS:
        assert np.array_equal(r[k], getattr(g, k)), k
    assert r["meta"]["n_ret"] == g.n_ret and "build_time_s" in r["meta"]
    # cap: balanced over squares, deterministic, highest synapse count first
    capped = build_retina(conn, g, cfg_for(retina_max=128), columns=table)
    assert capped["retina_idx"].shape == (128,) and capped["meta"]["capped_from"] == g.n_ret
    assert np.bincount(capped["retina_square"], minlength=64).tolist() == [2] * 64
    assert np.all(np.diff(capped["retina_idx"]) > 0)
    g_cap = build_brain_graph(conn, cfg_for(retina_max=128), columns=table)
    assert np.array_equal(g_cap.retina_idx, capped["retina_idx"]) and np.array_equal(g_cap.retina_square, capped["retina_square"])
    # a cap of 64 keeps exactly one photoreceptor per square
    g64 = build_brain_graph(conn, cfg_for(retina_max=64), columns=table)
    assert g64.n_ret == 64 and len(set(g64.retina_square.tolist())) == 64
    # a huge cap changes nothing
    assert build_brain_graph(conn, cfg_for(retina_max=10_000), columns=table).n_ret == g.n_ret
    # max_neurons without an explicit cap -> automatic cap (2000 -> 256, never below 64)
    from flychess.connectome.graph import default_retina_max

    assert default_retina_max(2000) == 256 and default_retina_max(100) == 64 and default_retina_max(10_000) == 1280
    assert GraphConfig(max_neurons=2000).effective_retina_max() == 256
    assert GraphConfig().effective_retina_max() is None and GraphConfig(retina_max=7).effective_retina_max() == 7
    g_auto = build_brain_graph(conn, cfg_for(max_neurons=200), columns=table)
    assert g_auto.n_ret == 64 and g_auto.n <= 200 and g_auto.meta["retina"]["cap"] == 64


def test_retina_neurons_survive_max_neurons_and_input_exclusion():
    conn, table, *_ = eye_fixture()
    cfg = cfg_for(max_neurons=90, retina_max=64, max_inputs=2)
    g = build_brain_graph(conn, cfg, columns=table)
    g.validate()
    assert g.n <= 90 and g.n_ret == 64 and g.n_in == 2 and g.n_out == 3
    kept = (g.root_ids - RID0).astype(int)
    # one photoreceptor per square, the best-connected one first (R1-6 carry 25 synapses, R7/R8 6)
    assert set(conn.cell_type[kept[g.retina_idx]]) == {"R1-6"} and len(set(g.retina_square.tolist())) == 64
    # the photoreceptors that are NOT in the retina are ordinary sensory neurons: eligible as inputs
    # when the ORNs are taken away
    conn_no_orn = eye_fixture(extra_sensory=0)[0]
    g2 = build_brain_graph(conn_no_orn, cfg_for(max_inputs=3, retina_max=8), columns=table)
    kept2 = (g2.root_ids - RID0).astype(int)
    assert set(conn_no_orn.cell_type[kept2[g2.input_idx]]) <= {"R1-6", "R7", "R8"}
    assert not np.isin(g2.input_idx, g2.retina_idx).any()
    # without a retina every photoreceptor competes for the input slots (previous behaviour)
    g3 = build_brain_graph(conn, cfg_for(retina=False), columns=table)
    assert not g3.has_retina and g3.meta["retina"] == {"enabled": False, "n_ret": 0, "reason": "disabled by GraphConfig.retina=False"}
    assert g3.n_in == 4 and g3.has_nt_type


def test_missing_column_table_builds_without_retina(monkeypatch, tmp_path, capsys):
    conn, *_ = eye_fixture()
    monkeypatch.setattr(paths, "CONNECTOME_DIR", tmp_path)  # no column_assignment.csv.gz here
    with pytest.warns(UserWarning, match="column_assignment.csv.gz not found .* WITHOUT a retina"):
        g = build_brain_graph(conn, cfg_for(), verbose=True)
    assert not g.has_retina and g.meta["retina"]["enabled"] is False
    assert g.meta["retina"]["reason"] == "column_assignment.csv.gz missing"
    assert "WITHOUT a retina" in capsys.readouterr().out
    g.validate()


def test_graph_roundtrip_and_old_format(tmp_path: Path):
    conn, table, *_ = eye_fixture()
    g = build_brain_graph(conn, cfg_for(), columns=table)
    p = g.save(tmp_path / "g.npz")
    with np.load(p, allow_pickle=False) as z:
        assert {"nt_type", "retina_idx", "retina_square", "retina_uv", "retina_type", "retina_eye"} <= set(z.files)
        assert z["retina_type"].dtype.kind == "U"
    g2 = BrainGraph.load(p)
    g2.validate()
    for f in ("root_ids", "csr_indptr", "csr_indices", "syn_count", "sign", "input_idx", "output_idx",
              "super_class", "position", "nt_type") + ret.RETINA_KEYS:
        assert np.array_equal(getattr(g, f), getattr(g2, f)), f
    assert g2.meta == json.loads(json.dumps(g.meta))
    # old format (v2 npz without the optional keys) -> empty retina / nt_type, still validates
    old = {k: v for k, v in np.load(p).items() if k not in ("nt_type",) + ret.RETINA_KEYS}
    np.savez(tmp_path / "old.npz", **old)
    g3 = BrainGraph.load(tmp_path / "old.npz")
    g3.validate()
    assert not g3.has_retina and not g3.has_nt_type and g3.n_ret == 0
    assert g3.retina_uv.shape == (0, 2) and g3.retina_idx.dtype == np.int32 and g3.nt_type.shape == (0,)
    assert np.array_equal(g3.csr_indices, g.csr_indices)
    # an empty-retina graph saves and loads again
    g4 = BrainGraph.load(g3.save(tmp_path / "old2.npz"))
    assert not g4.has_retina and g4.retina_uv.shape == (0, 2)
    # dataclass construction without the optional fields (what older callers / compute_ordering do)
    g5 = BrainGraph(n=g.n, root_ids=g.root_ids, csr_indptr=g.csr_indptr, csr_indices=g.csr_indices,
                    syn_count=g.syn_count, sign=g.sign, input_idx=g.input_idx, output_idx=g.output_idx,
                    super_class=g.super_class, position=g.position, meta={})
    g5.validate()
    assert not g5.has_retina


def test_validate_checks_retina_and_nt_type():
    g = toy_graph(n=120, nnz=800, n_in=8, n_out=8, n_ret=32, seed=2)
    assert g.has_retina and g.n_ret == 32 and g.has_nt_type
    assert set(g.retina_square.tolist()) == set(range(64)) or len(set(g.retina_square.tolist())) == 32
    assert np.all(g.retina_uv[g.retina_eye == 0, 0] < 0.5) and np.all(g.retina_uv[g.retina_eye == 1, 0] >= 0.5)
    assert not np.isin(g.retina_idx, g.input_idx).any() and not np.isin(g.retina_idx, g.output_idx).any()
    assert set(g.super_class[g.retina_idx]) == {"sensory"}

    def broken(**kw):
        h = toy_graph(n=120, nnz=800, n_in=8, n_out=8, n_ret=32, seed=2)
        for k, v in kw.items():
            setattr(h, k, v)
        return h

    with pytest.raises(AssertionError, match="generic inputs"):
        broken(input_idx=np.sort(np.concatenate([g.input_idx, g.retina_idx[:1]])).astype(np.int32)).validate()
    with pytest.raises(AssertionError, match="0..63"):
        broken(retina_square=np.where(np.arange(32) == 0, 64, g.retina_square).astype(np.int8)).validate()
    with pytest.raises(AssertionError, match="shapes"):
        broken(retina_eye=g.retina_eye[:-1]).validate()
    with pytest.raises(AssertionError, match="unique"):
        broken(retina_idx=np.where(np.arange(32) == 1, g.retina_idx[0], g.retina_idx).astype(np.int32)).validate()
    with pytest.raises(AssertionError, match="u < 0.5"):
        uv = g.retina_uv.copy()
        uv[g.retina_eye == 0, 0] = 0.75
        broken(retina_uv=uv).validate()
    with pytest.raises(AssertionError, match="0 .left. / 1"):
        broken(retina_eye=(g.retina_eye + 1).astype(np.int8)).validate()
    with pytest.raises(AssertionError, match="nt_type"):
        broken(nt_type=g.nt_type[:-1]).validate()
    with pytest.raises(AssertionError, match="NT_SIGN"):
        nt = g.nt_type.copy()
        nt[g.csr_indices[0]] = "GABA" if nt[g.csr_indices[0]] != "GABA" else "ACH"
        broken(nt_type=nt).validate()
    with pytest.raises(AssertionError):
        toy_graph(n=40, n_in=16, n_out=16, n_ret=16)  # does not fit


def test_toy_graph_retina_roundtrip(tmp_path: Path):
    g = toy_graph(n=100, nnz=600, n_in=8, n_out=8, n_ret=24, seed=5)
    g2 = BrainGraph.load(g.save(tmp_path / "toy.npz"))
    g2.validate()
    for k in ret.RETINA_KEYS + ("nt_type",):
        assert np.array_equal(getattr(g, k), getattr(g2, k)), k
    assert toy_graph(n=100, nnz=600, seed=5).n_ret == 0  # default unchanged


# ---- connectivity of a pruned graph's retina (reviewer finding: the tiny retina was inert) ----------
def _graph_hops(g: BrainGraph) -> np.ndarray:
    rows = np.repeat(np.arange(g.n), np.diff(g.csr_indptr))
    return ret.hops_to_targets(g.n, g.csr_indices, rows, g.output_idx)


def test_strongest_partner_cap_priority_and_round_robin():
    n = 8
    pre = np.array([0, 0, 1, 1, 2, 3, 4])
    post = np.array([5, 6, 6, 7, 5, 3, 1])
    syn = np.array([5, 9, 4, 4, 7, 6, 2])
    # 0 -> 6 (9 beats 5); 1: tie 6 / 7 -> smaller; 2 -> 5; 3 -> 3 is a self loop? no: 3 -> 3 not present
    part = ret.strongest_partner(np.array([0, 1, 2, 4, 7]), pre, post, syn, n)
    assert part.tolist() == [6, 6, 5, 1, -1]
    excl = np.zeros(n, dtype=bool)
    excl[6] = True  # partner 6 excluded (e.g. a photoreceptor) -> next best
    assert ret.strongest_partner(np.array([0, 1]), pre, post, syn, n, exclude=excl).tolist() == [5, 7]
    assert ret.strongest_partner(np.zeros(0, dtype=np.int64), pre, post, syn, n).shape == (0,)
    # cap priority: within a square the group sharing a partner with the largest summed weight first,
    # its members by weight; partnerless candidates last
    cand = ret.RetinaCandidates(idx=np.arange(6), square=np.array([0, 0, 0, 0, 1, 1], dtype=np.int8),
                                uv=np.zeros((6, 2), np.float32), cell_type=np.array(["R7"] * 6),
                                eye=np.zeros(6, np.int8), column_id=np.zeros(6, np.int32),
                                via_partner=np.zeros(6, bool))
    partner = np.array([10, 11, 11, -1, 12, 12])
    weight = np.array([30.0, 20.0, 20.0, 50.0, 1.0, 2.0])
    keys = ret.cap_priority(cand, partner, weight)
    order = ret.round_robin_order(cand.square, keys, cand.idx)
    # square 0: group 11 (40) beats group 10 (30); partnerless 3 last -> 1, 2, 0, 3; square 1: 5 then 4
    assert order.tolist() == [1, 5, 2, 4, 0, 3]
    capped = ret.balanced_cap(cand, keys, 4)
    assert capped.idx.tolist() == [1, 2, 4, 5] and capped.stats == {"capped_from": 6, "cap": 4}
    assert ret.round_robin_order(np.zeros(0), np.zeros(0), np.zeros(0)).shape == (0,)
    # a single priority array still works (previous API)
    assert np.array_equal(ret.balanced_cap(cand, weight, 2).idx, [3, 5])


def test_hops_and_output_paths_on_a_small_chain():
    #  0 -> 1 -> 2 -> 6(target)     0 -> 3 -> 6      4 -> 5 (dead end)     7 isolated
    n = 8
    pre = np.array([0, 1, 2, 0, 3, 4])
    post = np.array([1, 2, 6, 3, 6, 5])
    syn = np.array([5, 9, 9, 5, 5, 5])
    dist = ret.hops_to_targets(n, pre, post, np.array([6]))
    assert dist.tolist() == [2, 2, 1, 1, -1, -1, 0, -1]
    assert ret.hops_to_targets(n, pre, post, np.array([6]), max_hops=1).tolist() == [-1, -1, 1, 1, -1, -1, 0, -1]
    assert ret.hops_to_targets(n, pre, post, np.zeros(0, dtype=np.int64)).tolist() == [-1] * 8
    assert ret.hops_stats(np.array([2, 3, -1, 3])) == {"min": 2, "median": 3.0, "max": 3, "unreachable": 1, "n": 4}
    assert ret.hops_stats(np.array([-1]))["min"] is None
    ids = np.arange(n)
    kept = np.zeros(n, dtype=bool)
    kept[[0, 6]] = True
    # the shortest path 0 -> 3 -> 6 is preferred over the longer 0 -> 1 -> 2 -> 6
    nodes, ok = ret.output_paths(np.array([0, 4, 7]), n, pre, post, syn, dist, kept, ids)
    assert nodes.tolist() == [3, 6] and ok.tolist() == [True, False, False]
    # at equal distance (make 1 -> 6 exist): a kept neighbour is preferred ...
    pre2, post2, syn2 = np.r_[pre, 1], np.r_[post, 6], np.r_[syn, 1]
    dist2 = ret.hops_to_targets(n, pre2, post2, np.array([6]))
    assert dist2[1] == 1
    kept2 = kept.copy()
    kept2[3] = True
    nodes, ok = ret.output_paths(np.array([0]), n, pre2, post2, syn2, dist2, kept2, ids)
    assert nodes.tolist() == [3, 6]  # 3 (kept) wins over 1 (unkept, same synapses)
    # ... otherwise the most synapses on the edge out of the current neuron, then the smaller tiebreak
    syn3 = syn2.copy()
    syn3[0] = 7  # 0 -> 1 now carries 7 > 5
    nodes, _ = ret.output_paths(np.array([0]), n, pre2, post2, syn3, dist2, kept, ids)
    assert nodes.tolist() == [1, 6]
    nodes, _ = ret.output_paths(np.array([0]), n, pre2, post2, syn2, dist2, kept, ids)
    assert nodes.tolist() == [1, 6]  # equal synapses: smaller id
    nodes, _ = ret.output_paths(np.array([0]), n, pre2, post2, syn2, dist2, kept, -ids)
    assert nodes.tolist() == [3, 6]  # ... or whatever the tiebreak says
    # every accepted path's nodes are returned, merged over the starts
    nodes, ok = ret.output_paths(np.array([0, 1]), n, pre2, post2, syn2, dist2, kept, ids)
    assert nodes.tolist() == [1, 6] and ok.tolist() == [True, True]
    # budget: counts neurons outside `free` (default: kept); a path that does not fit is skipped whole
    nodes, ok = ret.output_paths(np.array([0]), n, pre, post, syn, dist, kept, ids, budget=0)
    assert nodes.shape == (0,) and not ok[0]
    nodes, ok = ret.output_paths(np.array([0]), n, pre, post, syn, dist, kept, ids, budget=1)
    assert nodes.tolist() == [3, 6] and ok[0]
    free = np.zeros(n, dtype=bool)
    free[6] = True
    nodes, ok = ret.output_paths(np.array([0]), n, pre, post, syn, dist, np.ones(n, bool), ids, budget=0, free=free)
    assert not ok[0]  # 3 is kept but not free -> costs a slot


def test_pruned_graph_retina_is_wired_to_the_outputs():
    conn, table, _cells, _r16, extra = eye_fixture(filler=120)
    cfg = cfg_for(max_neurons=150)
    g = build_brain_graph(conn, cfg, columns=table)
    g.validate()
    assert g.n == 150 and g.n_ret == 64 and g.meta["retina"]["cap"] == 64
    m = g.meta["retina"]
    assert m["connect"]["enabled"] and m["connect"]["connected"] == 64 and m["connect"]["unconnected"] == 0
    assert m["connect"]["path_neurons"] >= 5 and m["connect"]["path_neurons"] <= m["connect"]["budget"]
    hops = _graph_hops(g)[g.retina_idx]
    assert np.all(hops == 3) and m["hops_to_output"] == {"min": 3, "median": 3.0, "max": 3, "unreachable": 0, "n": 64}
    kept = (g.root_ids - RID0).astype(int)
    # the paths are R -> L1 -> central -> descending: every kept L1 is the partner of a kept photoreceptor
    l1 = [i for i in kept if conn.cell_type[i] == "L1"]
    assert len(l1) >= 1 and all(conn.cell_type[i] in ("R1-6", "R7", "R8") for i in conn.pre[np.isin(conn.post, l1) & np.isin(conn.pre, kept)][:5])
    assert set(extra["central"]) <= set(kept.tolist())
    # deterministic
    g2 = build_brain_graph(conn, cfg, columns=table)
    assert np.array_equal(g.root_ids, g2.root_ids) and np.array_equal(g.csr_indices, g2.csr_indices)
    # the path neurons displaced the lowest-ranked fillers, nothing else
    assert len(set(extra["filler"]) & set(kept.tolist())) == 150 - 64 - 4 - 3 - m["connect"]["path_neurons"]
    # retina_connect=False reproduces the old selection: the lamina is outranked by the fillers and the
    # whole retina is inert (no photoreceptor can reach an output)
    g0 = build_brain_graph(conn, cfg_for(max_neurons=150, retina_connect=False), columns=table)
    assert g0.n == 150 and g0.meta["retina"]["connect"]["enabled"] is False
    assert g0.meta["retina"]["hops_to_output"] == {"min": None, "median": None, "max": None, "unreachable": 64, "n": 64}
    assert not any(conn.cell_type[i] == "L1" for i in (g0.root_ids - RID0).astype(int))
    assert g0.meta["cfg"]["retina_connect"] is False and "retina_connect" in GraphConfig().to_dict()
    # the full (unpruned) fixture graph reports the hops too and needs no paths
    gf = build_brain_graph(conn, cfg_for(), columns=table)
    assert gf.meta["retina"]["hops_to_output"]["unreachable"] == 0 and gf.meta["retina"]["connect"]["enabled"]
    assert "note" in gf.meta["retina"]["connect"]


def test_output_path_budget_never_exceeds_max_neurons():
    conn, table, *_ = eye_fixture(filler=120)
    # roles: 64 retina + 2 inputs + 3 outputs = 69 -> 11 slots for paths (each new column costs an L1)
    g = build_brain_graph(conn, cfg_for(max_neurons=80, max_inputs=2), columns=table)
    g.validate()
    c = g.meta["retina"]["connect"]
    assert g.n == 80 and g.n_ret == 64 and c["budget"] == 11 and c["path_neurons"] <= 11
    assert 1 <= c["connected"] < 64 and c["connected"] + c["unconnected"] == 64
    hops = _graph_hops(g)[g.retina_idx]
    # (photoreceptors sharing a kept L1 with a connected one get connected for free)
    assert int((hops >= 0).sum()) >= c["connected"]
    # the connected photoreceptors are spread over the squares (round-robin), not clustered
    assert len(set(g.retina_square[hops >= 0].tolist())) >= c["connected"]
    # the budget never pushes n above max_neurons, even when the roles alone nearly fill it
    g2 = build_brain_graph(conn, cfg_for(max_neurons=70, max_inputs=2), columns=table)
    assert g2.n == 70 and g2.meta["retina"]["connect"]["budget"] == 1 and g2.meta["retina"]["connect"]["connected"] == 0


def test_receive_only_photoreceptors_are_not_retina():
    conn, table, cells, _r16, extra = eye_fixture(isolated_r7=0)
    # an R8 whose only connection is INCOMING (from its column's R7) cannot drive anything
    r7, r8 = cells[("right", 5, "R7")], cells[("right", 5, "R8")]
    edges = [(int(a), int(b), int(c)) for a, b, c in zip(conn.pre, conn.post, conn.syn_count) if a != r8]
    edges.append((r7, r8, 6))
    conn2 = make_conn(conn.n, edges, list(conn.cell_type), list(conn.super_class), list(conn.side))
    g = build_brain_graph(conn2, cfg_for(), columns=table)
    kept = (g.root_ids - RID0).astype(int)
    assert r8 in set(kept.tolist()) and r8 not in set(kept[g.retina_idx].tolist())
    assert g.meta["retina"]["per_type"]["R8"]["inactive"] == 1 and g.n_ret == 8 * extra["n_col"] - 1
    r = build_retina(conn2, g, cfg_for(), columns=table)
    assert np.array_equal(r["retina_idx"], g.retina_idx) and r["meta"]["hops_to_output"]["unreachable"] == 0


def test_vision_reaches_the_heads_on_a_pruned_fixture_graph():
    """The model check the reviewer asked for: with the retina wired through, toggling vision changes
    the activity of output neurons (and hence policy / value); on an inert retina it changes nothing."""
    torch = pytest.importorskip("torch")
    from flychess.model.config import BrainConfig
    from flychess.model.flybrain import FlyBrain

    def output_change(g: BrainGraph) -> tuple[int, float]:
        torch.manual_seed(0)
        m_on = FlyBrain(g, BrainConfig(vision=True, sensory_input=True, steps=4))
        m_off = FlyBrain(g, BrainConfig(vision=False, sensory_input=True, steps=4))
        m_off.load_state_dict({k: v for k, v in m_on.state_dict().items() if not k.startswith(("w_ret", "b_ret"))},
                              strict=False)
        x = torch.rand(2, 1280).round()
        with torch.no_grad():
            p1, _v1, h1 = m_on(x, return_activity=True)
            p2, _v2, h2 = m_off(x, return_activity=True)
        d = (h1 - h2).abs().amax(0).numpy()
        return int((d[g.output_idx] > 0).sum()), float((p1 - p2).abs().max())

    conn, table, *_ = eye_fixture(filler=120)
    g = build_brain_graph(conn, cfg_for(max_neurons=150), columns=table)
    n_changed, policy_diff = output_change(g)
    assert n_changed >= 1 and policy_diff > 0
    g0 = build_brain_graph(conn, cfg_for(max_neurons=150, retina_connect=False), columns=table)
    assert _graph_hops(g0)[g0.retina_idx].max() < 0  # inert: no photoreceptor reaches an output ...
    assert output_change(g0) == (0, 0.0)              # ... and vision changes nothing


# ---- real data (slow) ------------------------------------------------------------------------------
@pytest.mark.slow
def test_real_tiny_graph_retina_drives_the_outputs():
    p = paths.BRAIN_DIR / "tiny.npz"
    if not p.exists():
        pytest.skip("data/brain/tiny.npz not built (run `fly build-brain --tiny`)")
    g = BrainGraph.load(p)
    if not g.has_retina or "connect" not in g.meta["retina"]:
        pytest.skip("data/brain/tiny.npz predates retina_connect (rebuild with `fly build-brain --tiny --force`)")
    g.validate()
    assert g.n == 2000 and g.n_ret == 256 and g.meta["retina"]["connect"]["unconnected"] == 0
    hops = _graph_hops(g)[g.retina_idx]
    assert hops.min() >= 1 and hops.max() <= 8  # within BrainConfig.steps (default 8)
    torch = pytest.importorskip("torch")
    from flychess.model.config import BrainConfig
    from flychess.model.flybrain import FlyBrain

    torch.manual_seed(0)
    m_on = FlyBrain(g, BrainConfig(vision=True, sensory_input=True))
    m_off = FlyBrain(g, BrainConfig(vision=False, sensory_input=True))
    m_off.load_state_dict({k: v for k, v in m_on.state_dict().items() if not k.startswith(("w_ret", "b_ret"))}, strict=False)
    x = torch.rand(2, 1280).round()
    with torch.no_grad():
        p1, _v1, h1 = m_on(x, return_activity=True)
        p2, _v2, h2 = m_off(x, return_activity=True)
    d = (h1 - h2).abs().amax(0).numpy()
    is_ret = np.zeros(g.n, dtype=bool)
    is_ret[g.retina_idx] = True
    assert int((d[g.output_idx] > 0).sum()) >= 32 and int(((d > 0) & ~is_ret).sum()) >= 200
    assert float((p1 - p2).abs().max()) > 0


@pytest.mark.slow
def test_real_full_graph_retina():
    p = paths.BRAIN_DIR / "full.npz"
    if not p.exists():
        pytest.skip("data/brain/full.npz not built (run `fly build-brain`)")
    g = BrainGraph.load(p)
    if not g.has_retina:
        pytest.skip("data/brain/full.npz was built without a retina (rebuild with the column table)")
    g.validate()
    assert 5000 <= g.n_ret <= 7000 and g.has_nt_type
    m = g.meta["retina"]
    assert m["enabled"] and m["n_ret"] == g.n_ret
    assert 780 <= m["columns_per_eye"]["left"] <= 800 and 780 <= m["columns_per_eye"]["right"] <= 800
    assert m["photoreceptors_per_square"]["empty_squares"] == 0 and m["photoreceptors_per_square"]["min"] >= 10
    assert m["per_type"]["R1-6"]["via_partner"] >= 4000 and m["per_type"]["R7"]["direct"] >= 1200
    assert set(g.retina_type) == {"R1-6", "R7", "R8"} and set(g.retina_eye.tolist()) == {0, 1}
    assert set(g.super_class[g.retina_idx]) == {"sensory"}
    assert np.all(g.retina_square[g.retina_eye == 0] % 8 < 4) and np.all(g.retina_square[g.retina_eye == 1] % 8 >= 4)
    assert len(set(g.retina_square.tolist())) == 64
    assert not np.isin(g.retina_idx, g.input_idx).any() and g.n_in == 2048
    assert set(g.nt_type) <= {"", "ACH", "GABA", "GLUT", "DA", "SER", "OCT"}
    assert np.all(np.bincount(g.csr_indices, minlength=g.n)[g.retina_idx] > 0)  # every photoreceptor drives something
    if "hops_to_output" in m:
        assert m["hops_to_output"]["max"] <= 8 and m["hops_to_output"]["unreachable"] < 0.02 * g.n_ret
    # the v3 file keeps the v2 neurons, order and CSR (checkpoints of runs/fly1, runs/fly2 stay loadable)
    v2 = paths.BRAIN_DIR / "full-v2.npz"
    if v2.exists():
        old = BrainGraph.load(v2)
        for f in ("root_ids", "csr_indptr", "csr_indices", "syn_count", "sign", "output_idx"):
            assert np.array_equal(getattr(g, f), getattr(old, f)), f
        assert g.n_in == old.n_in and np.isin(old.input_idx, g.input_idx).sum() >= old.n_in - 8


@pytest.mark.slow
def test_real_column_table_lattice():
    from flychess.connectome import load_column_assignment

    try:
        table = load_column_assignment()
    except FileNotFoundError:
        pytest.skip("column_assignment.csv.gz not downloaded")
    assert 45_000 <= table.n <= 46_000 and {"R7", "R8", "L1", "L2", "L3", "Mi1"} <= set(table.types())
    assert np.array_equal(table.y.astype(int), table.p.astype(int) + table.q.astype(int))
    assert np.array_equal(table.x.astype(int), np.floor_divide(table.q.astype(int) - table.p.astype(int), 2))
    for eye in ("left", "right"):
        em = build_eye_map(table, eye)
        assert 780 <= em.n <= 800
        X, Y = hex_to_xy(em.p, em.q)
        pts = np.stack([X, Y], 1)
        d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        assert np.allclose(d.min(1), 1.0)  # every column's nearest neighbour is at unit distance
