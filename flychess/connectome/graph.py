"""BrainGraph: the connectome subgraph consumed by the model (see docs/SPEC.md §2.3).

`build_brain_graph` (selection from the full FlyWire connectome) is implemented further down;
the dataclass, its (de)serialisation and validation live here so that every module shares them.
"""
from __future__ import annotations

import json
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from flychess import paths

if TYPE_CHECKING:  # load.py imports pandas; keep graph.py importable without it
    from flychess.connectome.load import Connectome

# Dale's law in the fly CNS: acetylcholine excitatory, GABA and glutamate (GluCl) inhibitory,
# monoamines treated as excitatory/modulatory. Unknown transmitter -> excitatory.
# The sign is a property of the PRE-synaptic NEURON (one transmitter per neuron), never of a synapse:
# see `neuron_nt_type` for how neurons without an annotation in neurons.csv get their single label.
NT_SIGN = {"ACH": 1, "GABA": -1, "GLUT": -1, "DA": 1, "SER": 1, "OCT": 1, "": 1}


@dataclass
class GraphConfig:
    region: str = "full"                 # 'full' | 'central'
    max_neurons: int | None = None       # keep top-k neurons by total synapse count (inputs/outputs always kept)
    min_syn: int = 5                     # per (pre, post) pair, summed over neuropils
    input_super_classes: tuple[str, ...] = ("sensory", "ascending", "sensory_ascending")
    max_inputs: int = 2048
    output_super_classes: tuple[str, ...] = ("descending", "motor")
    max_outputs: int = 2048
    name: str = "full"
    # retina (flychess/connectome/retina.py, docs/RETINA.md): photoreceptors placed on the board
    retina: bool = True                                   # build the retina when the column table exists
    retina_types: tuple[str, ...] = ("R1-6", "R7", "R8")  # photoreceptor cell types
    retina_eyes: str = "both"                             # 'both' | 'left' | 'right'
    retina_field: str = "split"                           # 'split': left eye files a-d, right e-h | 'full'
    retina_binning: str = "weighted"                      # 'weighted' (photoreceptors per square balanced) | 'quantile' (columns) | 'uniform'
    retina_partner_types: tuple[str, ...] = ("L1", "L2", "L3")  # lamina partners that place R1-6
    retina_max: int | None = None                         # cap n_ret (balanced over squares); None = no cap for
    #                                                       the full brain, `default_retina_max` with max_neurons
    retina_connect: bool = True                           # with max_neurons: force-keep a shortest path from every
    #                                                       retina photoreceptor to an output neuron (never inert)

    def effective_retina_max(self) -> int | None:
        """`retina_max`, or the automatic cap when `max_neurons` is set (a 2000-neuron smoke graph must
        not be swamped by the ~6,200 photoreceptors): max_neurons / 8 rounded up to a multiple of 64,
        at least 64 (2000 -> 256)."""
        if self.retina_max is not None or self.max_neurons is None:
            return self.retina_max
        return default_retina_max(self.max_neurons)

    def to_dict(self) -> dict:
        return asdict(self)


def default_retina_max(max_neurons: int) -> int:
    return max(64, -(-(max_neurons // 8) // 64) * 64)


@dataclass
class BrainGraph:
    n: int
    root_ids: np.ndarray          # (n,) int64
    csr_indptr: np.ndarray        # (n+1,) int32, rows = POST-synaptic neuron
    csr_indices: np.ndarray       # (nnz,) int32, PRE-synaptic neuron of each synapse
    syn_count: np.ndarray         # (nnz,) float32
    sign: np.ndarray              # (nnz,) int8, +1 / -1
    input_idx: np.ndarray         # (n_in,) int32
    output_idx: np.ndarray        # (n_out,) int32
    super_class: np.ndarray       # (n,) str
    position: np.ndarray          # (n, 3) float32, NaN when unknown
    meta: dict = field(default_factory=dict)
    # ---- optional (v3, docs/RETINA.md); empty arrays when absent from an older npz ------------------
    nt_type: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=str))  # (n,) per-neuron transmitter
    retina_idx: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))      # (n_ret,) graph idx
    retina_square: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int8))    # (n_ret,) 0..63
    retina_uv: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float32))  # (n_ret, 2)
    retina_type: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=str))          # (n_ret,) 'R7'...
    retina_eye: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int8))       # (n_ret,) 0 L / 1 R

    # ---- derived -------------------------------------------------------------------------------
    @property
    def nnz(self) -> int:
        return int(self.csr_indices.shape[0])

    @property
    def n_in(self) -> int:
        return int(self.input_idx.shape[0])

    @property
    def n_out(self) -> int:
        return int(self.output_idx.shape[0])

    @property
    def n_ret(self) -> int:
        return int(self.retina_idx.shape[0])

    @property
    def has_retina(self) -> bool:
        return self.n_ret > 0

    @property
    def has_nt_type(self) -> bool:
        return self.nt_type.shape[0] == self.n and self.n > 0

    def validate(self) -> None:
        n, nnz = self.n, self.nnz
        assert self.root_ids.shape == (n,)
        assert self.csr_indptr.shape == (n + 1,) and self.csr_indptr[0] == 0 and self.csr_indptr[-1] == nnz
        assert np.all(np.diff(self.csr_indptr) >= 0)
        assert self.syn_count.shape == (nnz,) and self.sign.shape == (nnz,)
        assert np.all(np.isin(self.sign, (-1, 1)))
        assert self.csr_indices.min(initial=0) >= 0 and self.csr_indices.max(initial=-1) < n
        assert self.input_idx.min(initial=0) >= 0 and self.input_idx.max(initial=-1) < n
        assert self.output_idx.min(initial=0) >= 0 and self.output_idx.max(initial=-1) < n
        assert len(np.unique(self.input_idx)) == self.n_in and len(np.unique(self.output_idx)) == self.n_out
        assert self.super_class.shape == (n,) and self.position.shape == (n, 3)
        # canonical CSR: column indices strictly increasing within every row (no duplicate synapses)
        row_starts = self.csr_indptr[:-1]
        row_ends = self.csr_indptr[1:]
        inner = np.ones(nnz, dtype=bool)
        inner[row_starts[row_starts < nnz]] = False  # first entry of every row is exempt
        assert np.all(np.diff(self.csr_indices)[inner[1:]] > 0), "CSR columns must be sorted & unique per row"
        # no self loops
        rows = np.repeat(np.arange(n, dtype=np.int32), row_ends - row_starts)
        assert not np.any(rows == self.csr_indices), "self-loops are not allowed"
        # Dale's law: every PRE-synaptic neuron has a single sign on all of its outgoing synapses
        pos = np.bincount(self.csr_indices, weights=self.sign > 0, minlength=n)
        neg = np.bincount(self.csr_indices, weights=self.sign < 0, minlength=n)
        assert not np.any((pos > 0) & (neg > 0)), "Dale's law: a neuron has mixed outgoing signs"
        # optional per-neuron transmitter: when present it must be the origin of the signs
        assert self.nt_type.shape in ((0,), (n,)), "nt_type must be empty or (n,)"
        if self.has_nt_type:
            assert np.array_equal(self.sign, edge_signs(self.nt_type[self.csr_indices])), \
                "sign must equal NT_SIGN[nt_type[pre]]"
        # optional retina (docs/RETINA.md)
        k = self.n_ret
        assert self.retina_square.shape == (k,) and self.retina_uv.shape == (k, 2), "retina array shapes"
        assert self.retina_type.shape == (k,) and self.retina_eye.shape == (k,), "retina array shapes"
        if k:
            assert self.retina_idx.min() >= 0 and self.retina_idx.max() < n
            assert len(np.unique(self.retina_idx)) == k, "retina_idx must be unique"
            assert not np.isin(self.retina_idx, self.input_idx).any(), "retina neurons are not generic inputs"
            assert self.retina_square.min() >= 0 and self.retina_square.max() < 64, "square must be 0..63"
            assert np.all(np.isin(self.retina_eye, (0, 1))), "retina_eye must be 0 (left) / 1 (right)"
            assert np.all((self.retina_uv >= 0) & (self.retina_uv <= 1)), "retina_uv must be in [0, 1]"
            assert np.all(self.retina_uv[self.retina_eye == 0, 0] < 0.5), "left eye must have u < 0.5"
            assert np.all(self.retina_uv[self.retina_eye == 1, 0] >= 0.5), "right eye must have u >= 0.5"

    # ---- io ------------------------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            root_ids=self.root_ids.astype(np.int64),
            csr_indptr=self.csr_indptr.astype(np.int32),
            csr_indices=self.csr_indices.astype(np.int32),
            syn_count=self.syn_count.astype(np.float32),
            sign=self.sign.astype(np.int8),
            input_idx=self.input_idx.astype(np.int32),
            output_idx=self.output_idx.astype(np.int32),
            super_class=self.super_class.astype(str),
            position=self.position.astype(np.float32),
            meta=np.array(json.dumps(self.meta)),
            # v3 optional fields (always written; older readers ignore unknown keys)
            nt_type=np.asarray(self.nt_type).astype(str),
            retina_idx=self.retina_idx.astype(np.int32),
            retina_square=self.retina_square.astype(np.int8),
            retina_uv=self.retina_uv.astype(np.float32).reshape(-1, 2),
            retina_type=np.asarray(self.retina_type).astype(str),
            retina_eye=self.retina_eye.astype(np.int8),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> BrainGraph:
        with np.load(Path(path), allow_pickle=False) as z:
            g = cls(
                n=int(z["root_ids"].shape[0]),
                root_ids=z["root_ids"],
                csr_indptr=z["csr_indptr"],
                csr_indices=z["csr_indices"],
                syn_count=z["syn_count"],
                sign=z["sign"],
                input_idx=z["input_idx"],
                output_idx=z["output_idx"],
                super_class=z["super_class"].astype(str),
                position=z["position"],
                meta=json.loads(str(z["meta"])),
            )
            files = set(z.files)
            if "nt_type" in files:
                g.nt_type = z["nt_type"].astype(str)
            if "retina_idx" in files:  # v3: retina (absent from older graphs -> empty, see defaults)
                g.retina_idx = z["retina_idx"].astype(np.int32)
                g.retina_square = z["retina_square"].astype(np.int8)
                g.retina_uv = z["retina_uv"].astype(np.float32).reshape(-1, 2)
                g.retina_type = z["retina_type"].astype(str)
                g.retina_eye = z["retina_eye"].astype(np.int8)
        return g

    def summary(self) -> str:
        classes, counts = np.unique(self.super_class, return_counts=True)
        cls_str = ", ".join(f"{c}={k}" for c, k in zip(classes, counts))
        exc = int((self.sign > 0).sum())
        total_syn = int(self.syn_count.sum())
        retina = ""
        if self.has_retina:
            types, counts = np.unique(self.retina_type, return_counts=True)
            retina = f", retina={self.n_ret} [" + " ".join(f"{t}={k}" for t, k in zip(types, counts)) + "]"
        return (
            f"BrainGraph(n={self.n:,}, connections={self.nnz:,} [{exc:,} excitatory / {self.nnz - exc:,} inhibitory], "
            f"synapses={total_syn:,}, inputs={self.n_in}, outputs={self.n_out}{retina}; {cls_str})"
        )


def toy_graph(n: int = 200, nnz: int = 2000, n_in: int = 16, n_out: int = 16, seed: int = 0,
              n_ret: int = 0) -> BrainGraph:
    """Random graph with the BrainGraph invariants. FOR TESTS ONLY — never a substitute for the fly brain.

    ``n_ret > 0`` adds a random retina (``n_ret`` neurons disjoint from inputs/outputs, squares spread
    round-robin over the board, left eye first) so that the model's vision path can be unit-tested.
    """
    rng = np.random.default_rng(seed)
    pre = rng.integers(0, n, nnz * 2)
    post = rng.integers(0, n, nnz * 2)
    keep = pre != post
    pairs = np.unique(np.stack([post[keep], pre[keep]], 1), axis=0)[:nnz]
    post, pre = pairs[:, 0], pairs[:, 1]
    counts = np.bincount(post, minlength=n)
    indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)
    nt = rng.choice(["ACH", "GABA", "GLUT"], size=n, p=[0.6, 0.2, 0.2])
    sign = np.array([NT_SIGN[nt[p]] for p in pre], dtype=np.int8)
    perm = rng.permutation(n)
    assert n_in + n_out + n_ret <= n, "toy_graph: n_in + n_out + n_ret must not exceed n"
    ret = np.sort(perm[n_in + n_out:n_in + n_out + n_ret]).astype(np.int32)
    super_class = np.array(["central"] * n)
    super_class[ret] = "sensory"
    eye = (np.arange(n_ret) % 2).astype(np.int8)  # alternate left / right
    sq = np.zeros(n_ret, dtype=np.int64)
    for e in (0, 1):  # split field: left eye files 0-3, right eye files 4-7, round-robin over the block
        k = np.flatnonzero(eye == e)
        j = np.arange(len(k))
        sq[k] = (j // 4 % 8) * 8 + 4 * e + j % 4
    uv = np.stack([0.5 * eye + 0.5 * 0.998 * ((sq % 8 % 4) + rng.random(n_ret)) / 4,
                   (sq // 8 + rng.random(n_ret)) / 8], 1).astype(np.float32)
    g = BrainGraph(
        n=n,
        root_ids=np.arange(n, dtype=np.int64) + 720_575_940_000_000_000,
        csr_indptr=indptr,
        csr_indices=pre.astype(np.int32),
        syn_count=rng.integers(5, 60, len(pre)).astype(np.float32),
        sign=sign,
        input_idx=np.sort(perm[:n_in]).astype(np.int32),
        output_idx=np.sort(perm[n_in:n_in + n_out]).astype(np.int32),
        super_class=super_class,
        position=rng.random((n, 3)).astype(np.float32),
        meta={"toy": True, "seed": seed},
        nt_type=nt.astype(str),
        retina_idx=ret,
        retina_square=sq.astype(np.int8),
        retina_uv=uv,
        retina_type=np.array(["R1-6", "R7", "R8"], dtype=str)[np.arange(n_ret) % 3],
        retina_eye=eye,
    )
    g.validate()
    return g


# =================================================================================================
# Building a BrainGraph from the real connectome (docs/SPEC.md §2.3)
# =================================================================================================
CENTRAL_EXCLUDED = ("optic", "visual_projection", "visual_centrifugal")


def _pick_by_degree(candidates: np.ndarray, degree: np.ndarray, root_ids: np.ndarray, k: int) -> np.ndarray:
    """Deterministic role selection: sort `candidates` by (degree desc, root_id asc), take the first k."""
    if len(candidates) == 0:
        return candidates.astype(np.int32)
    order = np.lexsort((root_ids[candidates], -degree[candidates].astype(np.int64)))
    return np.sort(candidates[order[:k]]).astype(np.int32)


def neuron_nt_type(nt_type: np.ndarray, pre: np.ndarray, edge_nt: np.ndarray | None,
                   syn_count: np.ndarray) -> np.ndarray:
    """One transmitter label per NEURON (Dale's law), `''` when nothing is known.

    Neurons annotated in neurons.csv (`nt_type != ''`) keep their label. An unannotated neuron gets
    the `syn_count`-weighted majority of the per-connection predictions (`edge_nt`, one per
    (pre, post) pair) over ALL of its outgoing connections, ties broken alphabetically; edges whose
    own prediction is empty do not vote. Because the vote runs over the full edge set, a neuron's
    label does not depend on which subgraph is later selected. A neuron stays `''` (-> +1) when it
    has no labelled outgoing connection at all.
    """
    nt = np.asarray(nt_type).astype(str).copy()
    if edge_nt is None or len(nt) == 0:
        return nt
    n = len(nt)
    pre = np.asarray(pre)
    edge_nt = np.asarray(edge_nt).astype(str)
    vote = (nt[pre] == "") & (edge_nt != "")
    if not vote.any():
        return nt
    labels, code = np.unique(edge_nt[vote], return_inverse=True)  # sorted -> argmax ties = alphabetical
    k = len(labels)
    nt = nt.astype(f"U{max(nt.dtype.itemsize, labels.dtype.itemsize) // 4}")  # room for the new labels
    weight = np.asarray(syn_count)[vote].astype(np.float64)
    tally = np.bincount(pre[vote].astype(np.int64) * k + code, weights=weight, minlength=n * k).reshape(n, k)
    voted = tally.sum(1) > 0
    nt[voted] = labels[tally[voted].argmax(1)]
    return nt


def edge_signs(pre_nt: np.ndarray) -> np.ndarray:
    """Dale's-law sign per synapse from the PRE-synaptic NEURON's transmitter (`NT_SIGN`).

    `pre_nt[k]` is the per-neuron label (see `neuron_nt_type`) of the pre-synaptic neuron of synapse
    `k`; `''` and unknown labels map to +1. There is deliberately no per-synapse fallback: a neuron
    must carry one sign on every outgoing synapse.
    """
    nt = np.asarray(pre_nt).astype(str)
    uniq, inv = np.unique(nt, return_inverse=True)
    table = np.array([NT_SIGN.get(u, 1) for u in uniq], dtype=np.int8)
    return table[inv]


def _load_columns_or_none(cfg: GraphConfig, columns, verbose: bool):
    """The column table for the retina: the given one, else `data/connectome/column_assignment.csv.gz`
    (None -> retina disabled, with a note in meta) when it is missing."""
    if not cfg.retina:
        return None, "disabled by GraphConfig.retina=False"
    if columns is not None:
        return columns, None
    from flychess.connectome.load import COLUMN_FILE, load_column_assignment

    try:
        return load_column_assignment(paths.CONNECTOME_DIR), None
    except FileNotFoundError:
        # loud on purpose: a graph without a retina makes BrainConfig.vision=True a silent no-op
        warnings.warn(f"{COLUMN_FILE} not found in {paths.CONNECTOME_DIR}: building the graph WITHOUT a retina "
                      f"(download it from Codex 783 next to the other tables, or pass GraphConfig(retina=False))",
                      stacklevel=3)
        if verbose:
            print(f"[graph] {COLUMN_FILE} not found in {paths.CONNECTOME_DIR}: building WITHOUT a retina")
        return None, f"{COLUMN_FILE} missing"


def build_brain_graph(conn: Connectome, cfg: GraphConfig, columns=None, verbose: bool = False) -> BrainGraph:
    """Select the trained subgraph of the connectome and build it in canonical CSR form.

    Pipeline (all steps deterministic):
      1. region filter on neurons (`central` drops the optic-lobe super classes);
      2. keep synapses with `syn_count >= min_syn` between kept neurons, drop self-loops;
      2b. retina (`cfg.retina`, docs/RETINA.md): photoreceptors placed on a board square by
         `retina.retina_candidates` (full connectome), restricted to neurons with at least one
         OUTGOING connection on the filtered edge set (a photoreceptor that only receives could not
         drive anything; it stays an ordinary sensory neuron) and capped at `cfg.effective_retina_max()` (balanced over squares; within a square
         photoreceptors sharing their strongest post-synaptic partner come first, then highest total
         synapse count; automatic cap when `max_neurons` is set, e.g. 256 for the 2000-neuron tiny graph);
      3. inputs  = top `max_inputs`  neurons of `input_super_classes`  by (out-degree desc, root_id),
         EXCLUDING the retina neurons (they get the board through the retina path instead);
         outputs = top `max_outputs` neurons of `output_super_classes` by (in-degree desc, root_id);
         (degree = number of distinct partners on the filtered edge set; a neuron in both lists is an
         input first and is removed from the output candidates);
      4. if `max_neurons`: keep inputs ∪ outputs ∪ retina plus the highest-ranked other neurons by
         (total synapse count in+out desc, root_id) so that n <= max_neurons; with `retina_connect`
         a shortest path (on the filtered edge set) from every retina photoreceptor to an output
         neuron is force-kept as well (`retina.output_paths`: preferring neurons that are kept
         anyway, paths merge; it never pushes n above max_neurons — photoreceptors whose path does
         not fit stay unconnected and are counted in meta['retina']['connect']), so that the retina
         of a pruned graph can actually drive the outputs;
      5. drop neurons that end up with zero degree, except inputs/outputs/retina;
      6. rows = POST-synaptic neuron, columns sorted, i.e. entries ordered by (post, pre).
    Sign comes from the PRE-synaptic neuron's transmitter (`edge_signs`), never learned; neurons
    without an annotation get one label from the synapse-weighted majority of their connection-level
    predictions over the FULL connectome (`neuron_nt_type`), so every neuron has exactly one sign.
    The per-neuron label is stored as `nt_type`.

    `columns` is the retina's `ColumnTable` (tests); by default it is read from `data/connectome/`
    and the retina is silently empty when the table is absent (meta['retina']['enabled'] = False).
    """
    from flychess.connectome import retina as ret
    from flychess.connectome.load import Connectome  # local import: load.py depends on pandas

    assert isinstance(conn, Connectome)
    t0 = time.time()
    if cfg.region not in ("full", "central"):
        raise ValueError(f"unknown region {cfg.region!r} (expected 'full' or 'central')")
    N = conn.n
    keep_neuron = np.ones(N, dtype=bool)
    if cfg.region == "central":
        keep_neuron &= ~np.isin(conn.super_class, CENTRAL_EXCLUDED)

    # one transmitter per neuron, decided on the full edge set (independent of the selection below)
    nt_neuron = neuron_nt_type(conn.nt_type, conn.pre, conn.edge_nt_type, conn.syn_count)

    # 2. edge filter
    pre, post, syn = conn.pre, conn.post, conn.syn_count
    e_keep = (syn >= cfg.min_syn) & keep_neuron[pre] & keep_neuron[post] & (pre != post)
    pre, post, syn = pre[e_keep], post[e_keep], syn[e_keep]
    out_deg = np.bincount(pre, minlength=N)
    in_deg = np.bincount(post, minlength=N)
    total_syn = np.bincount(pre, weights=syn, minlength=N) + np.bincount(post, weights=syn, minlength=N)

    # 2b. retina candidates (photoreceptors with a column), on the full connectome
    columns, retina_note = _load_columns_or_none(cfg, columns, verbose)
    retina_glob = np.zeros(0, dtype=np.int64)
    cand = None
    priority = None
    if columns is not None:
        cand = ret.retina_candidates(conn, cfg, columns, active=keep_neuron & (out_deg > 0))
        is_cand = np.zeros(N, dtype=bool)
        is_cand[cand.idx] = True

        def _priority(c):  # photoreceptors sharing their strongest (non-photoreceptor) partner first
            return ret.cap_priority(c, ret.strongest_partner(c.idx, pre, post, syn, N, exclude=is_cand),
                                    total_syn[c.idx])

        cand = ret.balanced_cap(cand, _priority(cand), cfg.effective_retina_max())
        retina_glob = cand.idx
        priority = _priority(cand)
    is_retina = np.zeros(N, dtype=bool)
    is_retina[retina_glob] = True

    # 3. roles
    in_cand = np.flatnonzero(keep_neuron & ~is_retina & np.isin(conn.super_class, cfg.input_super_classes))
    input_glob = _pick_by_degree(in_cand, out_deg, conn.root_ids, cfg.max_inputs)
    out_cand = np.flatnonzero(keep_neuron & ~is_retina & np.isin(conn.super_class, cfg.output_super_classes))
    out_cand = out_cand[~np.isin(out_cand, input_glob)]
    output_glob = _pick_by_degree(out_cand, in_deg, conn.root_ids, cfg.max_outputs)
    role = np.zeros(N, dtype=bool)
    role[input_glob] = True
    role[output_glob] = True
    role |= is_retina

    # 4. top-k by total synapse count (+ retina output paths)
    connect_meta = None
    if cfg.max_neurons is not None:
        others = np.flatnonzero(keep_neuron & ~role)
        ranked = others[np.lexsort((conn.root_ids[others], -total_syn[others]))]

        def _select(role: np.ndarray) -> np.ndarray:
            budget = max(cfg.max_neurons - int(role.sum()), 0)
            k = role.copy()
            k[ranked[~role[ranked]][:budget]] = True
            return k

        keep_neuron = _select(role)
        if cfg.retina_connect and len(retina_glob):
            # force-keep a shortest path photoreceptor -> ... -> output for every retina neuron,
            # processed in the balanced (round-robin over squares) order so that a tight budget
            # still connects every square first
            starts = retina_glob[ret.round_robin_order(cand.square, priority, cand.idx)]
            dist = ret.hops_to_targets(N, pre, post, output_glob)
            path_budget = max(cfg.max_neurons - int(role.sum()), 0)
            nodes, connected = ret.output_paths(starts, N, pre, post, syn, dist, keep_neuron, conn.root_ids,
                                                path_budget, free=role)
            new_roles = nodes[~role[nodes]]
            role[nodes] = True
            keep_neuron = _select(role)
            connect_meta = {"enabled": True, "connected": int(connected.sum()),
                            "unconnected": int((~connected).sum()), "path_neurons": len(new_roles),
                            "budget": int(path_budget)}
        e_keep = keep_neuron[pre] & keep_neuron[post]
        pre, post, syn = pre[e_keep], post[e_keep], syn[e_keep]

    # 5. zero-degree removal (roles exempt)
    deg = np.bincount(pre, minlength=N) + np.bincount(post, minlength=N)
    keep_neuron &= (deg > 0) | role
    kept = np.flatnonzero(keep_neuron)
    n = len(kept)
    new_index = np.full(N, -1, dtype=np.int64)
    new_index[kept] = np.arange(n)

    # 6. canonical CSR (rows = post, columns = pre, sorted by (post, pre))
    pre_n, post_n = new_index[pre], new_index[post]
    order = np.lexsort((pre_n, post_n))
    pre_n, post_n, syn = pre_n[order], post_n[order], syn[order]
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(post_n, minlength=n), out=indptr[1:])
    sign = edge_signs(nt_neuron[kept][pre_n])

    # retina arrays in graph indices (every retina candidate is a role, hence kept)
    if cand is not None:
        retina = ret.candidates_to_arrays(cand, new_index)
        assert retina["retina_idx"].shape[0] == len(retina_glob)
        retina_meta = ret.retina_meta(cand, retina, columns)
        # how far the outputs are from the photoreceptors IN THE FINAL GRAPH (synaptic hops; the model
        # needs hops <= BrainConfig.steps for the board to reach the heads)
        out_n = new_index[output_glob]
        hops = ret.hops_to_targets(n, pre_n, post_n, out_n)[retina["retina_idx"]]
        retina_meta["hops_to_output"] = ret.hops_stats(hops)
        retina_meta["connect"] = connect_meta if connect_meta is not None else {
            "enabled": bool(cfg.retina_connect), "note": "no pruning (max_neurons=None): every path of the "
            "filtered connectome is kept"}
    else:
        retina = ret.empty_retina()
        retina_meta = {"enabled": False, "n_ret": 0, "reason": retina_note}

    classes, counts = np.unique(conn.super_class[kept], return_counts=True)
    meta = {
        "cfg": cfg.to_dict(),
        "n": int(n),
        "nnz": len(pre_n),
        "n_in": len(input_glob),
        "n_out": len(output_glob),
        "n_ret": int(retina["retina_idx"].shape[0]),
        "retina": retina_meta,
        "super_class_counts": {str(c or ""): int(k) for c, k in zip(classes, counts)},
        "sign_counts": {"excitatory": int((sign > 0).sum()), "inhibitory": int((sign < 0).sum())},
        "nt_fallback": {
            "rule": "unannotated neuron -> syn_count-weighted majority of its edge nt predictions (one sign per neuron)",
            "unannotated_neurons": int((conn.nt_type[kept] == "").sum()),
            "resolved_by_majority": int(((conn.nt_type[kept] == "") & (nt_neuron[kept] != "")).sum()),
        },
        "total_syn_count": float(syn.sum()),
        "connectome": {"neurons": int(N), "edges": int(conn.n_edges)},
        "sources": dict(conn.sources),
        "build_time_s": round(time.time() - t0, 3),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    g = BrainGraph(
        n=n,
        root_ids=conn.root_ids[kept].astype(np.int64),
        csr_indptr=indptr.astype(np.int32),
        csr_indices=pre_n.astype(np.int32),
        syn_count=syn.astype(np.float32),
        sign=sign.astype(np.int8),
        input_idx=new_index[input_glob].astype(np.int32),
        output_idx=new_index[output_glob].astype(np.int32),
        super_class=conn.super_class[kept].astype(str),
        position=conn.position[kept].astype(np.float32),
        meta=meta,
        nt_type=nt_neuron[kept].astype(str),
        **retina,
    )
    g.validate()
    return g


def graph_path(cfg: GraphConfig) -> Path:
    return paths.BRAIN_DIR / f"{cfg.name}.npz"


def load_or_build(cfg: GraphConfig, out_path: str | Path | None = None, conn: Connectome | None = None,
                  verbose: bool = True) -> BrainGraph:
    """Return the BrainGraph for `cfg`, loading `data/brain/<cfg.name>.npz` when it exists.

    Otherwise the connectome is loaded (`load_connectome`, or the given `conn`), the graph is built,
    saved to `out_path` (default `data/brain/<cfg.name>.npz`) and returned.
    """
    path = Path(out_path) if out_path is not None else graph_path(cfg)
    if path.exists():
        return BrainGraph.load(path)
    if conn is None:
        from flychess.connectome.load import load_connectome

        conn = load_connectome(verbose=verbose)
    g = build_brain_graph(conn, cfg, verbose=verbose)
    g.save(path)
    if verbose:
        print(f"built {g.summary()} in {g.meta['build_time_s']}s -> {path}")
    return g
