"""The fly's retina: which photoreceptor looks at which square of the chess board (docs/RETINA.md).

FlyWire's optic-lobe column assignment (Matsliah et al. 2024, doi:10.1038/s41586-024-07981-1) places
the neurons of 31 columnar cell types — among them the R7 / R8 photoreceptors and the lamina
monopolar cells L1–L5 — on the hexagonal lattice of ~796 ommatidial columns per eye. This module
turns that lattice into a retinotopic input for the chess network:

* every column gets a position ``(u_board, v)`` in its eye's visual field (regular cartesian embedding
  of the hex axial coordinates, normalised per eye);
* the columns of each eye are binned onto the board — by default the left eye sees files a–d and the
  right eye files e–h (``field='split'``), each eye's field being an 8-rank x 4-file block; with
  ``field='full'`` both eyes see all 64 squares. Binning is by quantiles (``binning='quantile'``:
  rank-based, nested first in u then in v) so that every square receives the same number of columns
  (+-1), or uniform (``binning='uniform'``) for a geometrically faithful but unbalanced map;
* photoreceptors inherit their column's square: R7 / R8 have column assignments directly; the outer
  photoreceptors R1-6 (not in the table) take the column of their strongest postsynaptic lamina
  partner (``partner_types``, default L1 / L2 / L3; the column receiving the most synapses — a
  plurality, argmax — over all their outgoing connections in the full connectome);
* in a pruned graph (``GraphConfig.max_neurons``) the retina is capped (``balanced_cap``: round-robin
  over squares, photoreceptors that share a post-synaptic partner first) and every kept
  photoreceptor is wired through: a shortest path to an output neuron is force-kept
  (``output_paths``), so the retina is never an inert appendix of the graph.

The model (``BrainConfig.vision``) injects ``planes[:, :, retina_square[k]]`` into neuron
``retina_idx[k]`` through a learned per-photoreceptor projection ``w_ret[k]`` (20 planes -> 1), so a
photoreceptor only ever sees the one square it looks at. The network is still the only thing that
chooses moves; the retina merely decides *where* each photoreceptor looks.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from flychess import paths

if TYPE_CHECKING:  # avoid the graph.py <-> retina.py import cycle at runtime (duck-typed cfg)
    from flychess.connectome.graph import BrainGraph
    from flychess.connectome.load import ColumnTable, Connectome

RETINA_TYPES: tuple[str, ...] = ("R1-6", "R7", "R8")
PARTNER_TYPES: tuple[str, ...] = ("L1", "L2", "L3")
EYES = ("both", "left", "right")
FIELDS = ("split", "full")
BINNINGS = ("quantile", "weighted", "uniform")
DEFAULT_BINNING = "weighted"  # must match GraphConfig.retina_binning
EYE_CODE = {"left": 0, "right": 1}
N_RANKS = 8
N_FILES = 8

RETINA_KEYS = ("retina_idx", "retina_square", "retina_uv", "retina_type", "retina_eye")


# =================================================================================================
# geometry
# =================================================================================================
def hex_to_xy(p: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Regular cartesian embedding of the column lattice's hex axial coordinates.

    Codex's ``(p, q)`` axes are at 120 degrees (the six neighbours of a column are ``(±1, 0)``,
    ``(0, ±1)`` and ``±(1, 1)``): ``X = (q - p) * sqrt(3) / 2``, ``Y = (p + q) / 2`` gives unit
    spacing between all neighbouring columns (checked on the real table: every column's nearest
    neighbour is at distance 1.000). The file's own ``(x, y)`` offset coordinates are the same thing
    squashed onto integers (``y = p + q``, ``x = floor((q - p) / 2)``). Orientation (from the
    correlation with the neurons' FAFB positions, see docs/RETINA.md): ``+Y`` = dorsal in both eyes;
    ``+X`` = posterior medulla = frontal visual field (after the outer-chiasm inversion) in both eyes.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    return (q - p) * (np.sqrt(3.0) / 2.0), (p + q) / 2.0


def quantile_bins(primary: np.ndarray, secondary: np.ndarray, tiebreak: np.ndarray, nbins: int,
                  weight: np.ndarray | None = None) -> np.ndarray:
    """Rank-based binning: sort by (primary, secondary, tiebreak), give bin ``rank * nbins // n``.

    Every bin receives ``n // nbins`` or ``n // nbins + 1`` items; fully deterministic for a given
    ``tiebreak`` (unique ids). With ``weight`` (>= 0 per item) the bins are cut at equal cumulative
    weight instead (an item goes to the bin holding the midpoint of its weight interval), so bins
    are balanced in total weight rather than in item count. Every bin is non-empty iff ``n >= nbins``
    (with fewer items than bins some bins are necessarily empty).
    """
    n = len(primary)
    out = np.zeros(n, dtype=np.int64)
    if n == 0:
        return out
    order = np.lexsort((tiebreak, secondary, primary))
    if weight is None:
        ranks = np.empty(n, dtype=np.int64)
        ranks[order] = np.arange(n)
        out[:] = (ranks * nbins) // n
        return out
    w = np.asarray(weight, dtype=np.float64)[order]
    total = w.sum()
    if total <= 0:
        return quantile_bins(primary, secondary, tiebreak, nbins)
    mid = np.cumsum(w) - 0.5 * w
    # cut positions in sorted order: bin b starts at the first item whose weight midpoint reaches
    # b/nbins of the total; then every bin is forced to hold >= 1 item (a heavy item could otherwise
    # skip a bin entirely) as long as there are enough items
    cuts = np.searchsorted(mid, np.arange(nbins + 1) * (total / nbins), side="left")
    cuts[0], cuts[-1] = 0, n
    if n >= nbins:
        for b in range(1, nbins):
            cuts[b] = max(cuts[b], cuts[b - 1] + 1)
        for b in range(nbins - 1, 0, -1):
            cuts[b] = min(cuts[b], cuts[b + 1] - 1)
    out[order] = np.repeat(np.arange(nbins), np.diff(cuts))
    return out


def uniform_bins(value: np.ndarray, nbins: int) -> np.ndarray:
    """``floor(value * nbins)`` for value in [0, 1], clipped into ``[0, nbins)``."""
    return np.clip(np.floor(np.asarray(value, dtype=np.float64) * nbins), 0, nbins - 1).astype(np.int64)


def assign_squares(u_board: np.ndarray, v: np.ndarray, ids: np.ndarray, eye: str, field: str,
                   binning: str, weight: np.ndarray | None = None) -> np.ndarray:
    """Square ``rank * 8 + file`` (mover's perspective) for the columns of ONE eye.

    ``u_board`` in [0, 1] runs from file a to file h across the eye's field, ``v`` from rank 1 to
    rank 8; ``ids`` are the (unique) column ids used as deterministic tie-breaker. With
    ``field='split'`` the left eye covers files 0-3 and the right eye files 4-7 (4 x 8 squares),
    with ``field='full'`` each eye covers all 8 files. ``binning='quantile'`` balances the number
    of columns per square, ``'weighted'`` the total ``weight`` per square (the number of
    photoreceptors a column carries), ``'uniform'`` cuts the eye map into equal rectangles.

    The rank binning is nested inside every file bin, so the guarantee "every square of the field
    receives >= 1 column" needs at least ``N_RANKS`` columns per file bin, i.e. >= 32 (``split``) /
    64 (``full``) columns per eye for the quantile binnings (the real eyes have ~790; a synthetic
    table with fewer columns can leave squares empty — `square_stats` reports ``empty_squares`` and
    `retina_candidates` warns).
    """
    if field not in FIELDS:
        raise ValueError(f"field must be one of {FIELDS}, got {field!r}")
    if binning not in BINNINGS:
        raise ValueError(f"binning must be one of {BINNINGS}, got {binning!r}")
    if eye not in EYE_CODE:
        raise ValueError(f"eye must be 'left' or 'right', got {eye!r}")
    u_board = np.asarray(u_board, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    ids = np.asarray(ids)
    nfiles = 4 if field == "split" else N_FILES
    offset = (4 if eye == "right" else 0) if field == "split" else 0
    if binning == "uniform":
        file_bin = uniform_bins(u_board, nfiles)
        rank_bin = uniform_bins(v, N_RANKS)
    else:
        w = None if binning == "quantile" or weight is None else np.asarray(weight, dtype=np.float64)
        file_bin = quantile_bins(u_board, v, ids, nfiles, w)
        rank_bin = np.zeros(len(u_board), dtype=np.int64)
        for b in range(nfiles):  # nested: ranks are binned within every file bin -> balanced squares
            m = file_bin == b
            rank_bin[m] = quantile_bins(v[m], u_board[m], ids[m], N_RANKS, None if w is None else w[m])
    return (rank_bin * N_FILES + offset + file_bin).astype(np.int64)


# =================================================================================================
# columns -> eye map
# =================================================================================================
@dataclass
class EyeMap:
    """The unique columns of one eye with their normalised coordinates and board squares."""

    eye: str
    column_id: np.ndarray   # (C,) int32 unique column ids
    p: np.ndarray           # (C,)
    q: np.ndarray           # (C,)
    u_board: np.ndarray     # (C,) in [0, 1], file a -> h direction within this eye's field
    v: np.ndarray           # (C,) in [0, 1], rank 1 -> 8 (ventral -> dorsal)
    uv: np.ndarray          # (C, 2) float32 eye-map coordinates for visualisation (left u<0.5, right u>0.5)
    square: np.ndarray      # (C,) int64 board square 0..63

    @property
    def n(self) -> int:
        return int(self.column_id.shape[0])

    def lookup(self) -> dict[int, int]:
        return {int(c): i for i, c in enumerate(self.column_id)}


def _robust_minmax(a: np.ndarray) -> tuple[float, float]:
    """Min/max over the *columns* (each column counted once, so a crowded column cannot bias it);
    the hex lattice has no outliers, so plain extremes are the robust choice. Degenerate -> (a, a+1)."""
    lo, hi = float(np.min(a)), float(np.max(a))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def build_eye_map(columns: ColumnTable, eye: str, field: str = "split", binning: str = "quantile",
                  column_weight: dict[int, float] | None = None) -> EyeMap:
    """Embed one eye's columns, normalise per eye and bin them onto the board (see module doc).

    ``column_weight`` (column id -> weight, e.g. photoreceptors placed on it) is used by the
    ``'weighted'`` binning only.
    """
    m = columns.hemisphere == eye
    if not m.any():
        empty = np.zeros(0)
        return EyeMap(eye, np.zeros(0, np.int32), empty, empty, empty, empty, np.zeros((0, 2), np.float32),
                      np.zeros(0, np.int64))
    cid, first = np.unique(columns.column_id[m], return_index=True)
    idx = np.flatnonzero(m)[first]
    p, q = columns.p[idx].astype(np.int32), columns.q[idx].astype(np.int32)
    X, Y = hex_to_xy(p, q)
    x0, x1 = _robust_minmax(X)
    y0, y1 = _robust_minmax(Y)
    xn = np.clip((X - x0) / (x1 - x0), 0.0, 1.0)   # 0 = lateral, 1 = frontal (both eyes)
    v = np.clip((Y - y0) / (y1 - y0), 0.0, 1.0)    # 0 = ventral (rank 1), 1 = dorsal (rank 8)
    # frontal field meets the midline of the board: left eye a -> d is lateral -> frontal,
    # right eye e -> h is frontal -> lateral
    u_board = xn if eye == "left" else 1.0 - xn
    # eye map for visualisation: left eye u in [0, 0.499], right eye u in [0.5, 1] (the 0.998 keeps
    # the frontmost left column below 0.5 even after f16 rounding in the web export)
    u_map = 0.5 * u_board * 0.998 if eye == "left" else 0.5 + 0.5 * u_board
    uv = np.stack([u_map, v], 1).astype(np.float32)
    weight = None
    if column_weight is not None:
        weight = np.array([float(column_weight.get(int(c), 0.0)) for c in cid])
    square = assign_squares(u_board, v, cid, eye, field, binning, weight)
    return EyeMap(eye, cid.astype(np.int32), p, q, u_board, v, uv, square)


def field_squares(eye: str, field: str) -> np.ndarray:
    """The board squares an eye's field covers (32 for 'split', 64 for 'full')."""
    files = np.arange(8) if field == "full" else np.arange(4) + (4 if eye == "right" else 0)
    return (np.arange(8)[:, None] * 8 + files[None, :]).ravel()


# =================================================================================================
# photoreceptors -> columns
# =================================================================================================
@dataclass
class RetinaCandidates:
    """Every photoreceptor of the connectome that could be placed on a column (connectome indices).

    Computed on the FULL connectome, independent of the subgraph selection (like the per-neuron
    transmitter rule), so the square a photoreceptor looks at never depends on the graph config.
    """

    idx: np.ndarray        # (K,) int64 connectome index, sorted
    square: np.ndarray     # (K,) int8
    uv: np.ndarray         # (K, 2) float32
    cell_type: np.ndarray  # (K,) str
    eye: np.ndarray        # (K,) int8 0 = left, 1 = right
    column_id: np.ndarray  # (K,) int32
    via_partner: np.ndarray  # (K,) bool, True when placed through a lamina partner
    stats: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.idx.shape[0])

    def take(self, mask: np.ndarray) -> RetinaCandidates:
        return RetinaCandidates(self.idx[mask], self.square[mask], self.uv[mask], self.cell_type[mask],
                                self.eye[mask], self.column_id[mask], self.via_partner[mask], dict(self.stats))


def _cfg_get(cfg, name: str, default):
    return getattr(cfg, name, default) if cfg is not None else default


def retina_options(cfg) -> dict:
    """Normalised retina options from a `GraphConfig` (or anything with the same attributes)."""
    types = tuple(str(t) for t in _cfg_get(cfg, "retina_types", RETINA_TYPES))
    eyes = str(_cfg_get(cfg, "retina_eyes", "both"))
    fld = str(_cfg_get(cfg, "retina_field", "split"))
    binning = str(_cfg_get(cfg, "retina_binning", DEFAULT_BINNING))
    partners = tuple(str(t) for t in _cfg_get(cfg, "retina_partner_types", PARTNER_TYPES))
    max_neurons = _cfg_get(cfg, "retina_max", None)
    if eyes not in EYES:
        raise ValueError(f"retina_eyes must be one of {EYES}, got {eyes!r}")
    if fld not in FIELDS:
        raise ValueError(f"retina_field must be one of {FIELDS}, got {fld!r}")
    if binning not in BINNINGS:
        raise ValueError(f"retina_binning must be one of {BINNINGS}, got {binning!r}")
    effective_field = fld if eyes == "both" else "full"  # a single eye always sees the whole board
    return {
        "types": types, "eyes": eyes, "field": fld, "effective_field": effective_field, "binning": binning,
        "partner_types": partners, "max": None if max_neurons is None else int(max_neurons),
    }


def _partner_columns(conn: Connectome, need: np.ndarray, partner_ok: np.ndarray, col_key: np.ndarray) -> np.ndarray:
    """Column key (hemisphere-coded column id) of the strongest postsynaptic partner per neuron.

    ``need`` marks the pre-synaptic neurons to place, ``partner_ok`` the admissible post-synaptic
    partners (which must have ``col_key >= 0``). Votes are synapse-count-weighted over all outgoing
    connections of the full connectome; ties are broken by the smaller column key. Returns -1 where
    nothing votes.
    """
    n = conn.n
    out = np.full(n, -1, dtype=np.int64)
    e = need[conn.pre] & partner_ok[conn.post]
    if not e.any():
        return out
    pre = conn.pre[e].astype(np.int64)
    key = col_key[conn.post[e]].astype(np.int64)
    syn = conn.syn_count[e].astype(np.float64)
    # aggregate synapses per (pre, column key), then pick the max per pre (ties -> smaller key)
    pair = pre * (key.max() + 1) + key
    uniq, inv = np.unique(pair, return_inverse=True)
    tot = np.bincount(inv, weights=syn)
    u_pre = uniq // (key.max() + 1)
    u_key = uniq % (key.max() + 1)
    order = np.lexsort((u_key, -tot, u_pre))  # per pre: strongest first, then smaller key
    u_pre_o = u_pre[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = u_pre_o[1:] != u_pre_o[:-1]
    out[u_pre_o[first]] = u_key[order][first]
    return out


def retina_candidates(conn: Connectome, cfg=None, columns: ColumnTable | None = None,
                      active: np.ndarray | None = None) -> RetinaCandidates:
    """Place every photoreceptor of the requested types on a column / square (full connectome).

    Direct assignments come from the column table (R7, R8); the rest (R1-6, and any R7/R8 missing
    from the table) go through their strongest postsynaptic ``partner_types`` neuron that has a
    column. Photoreceptors that end up without a column are simply not part of the retina (they stay
    ordinary sensory neurons and remain candidates for the generic `input_idx`).

    ``active`` (bool mask over connectome neurons) restricts the retina to those neurons — the graph
    builder passes the neurons that have at least one outgoing connection, so the ``'weighted'``
    binning balances the photoreceptors that really drive the graph. The column placement itself
    never depends on it.
    """
    from flychess.connectome.load import load_column_assignment

    opt = retina_options(cfg)
    if columns is None:
        columns = load_column_assignment(paths.CONNECTOME_DIR)
    eyes = ("left", "right") if opt["eyes"] == "both" else (opt["eyes"],)

    n = conn.n
    # column key per connectome neuron: eye_code * 2^20 + column_id (-1 = none); table rows of the
    # other eye (when a single eye is requested) or of unknown neurons are ignored
    col_key = np.full(n, -1, dtype=np.int64)
    rid_idx = np.searchsorted(conn.root_ids, columns.root_ids)
    rid_idx[rid_idx >= n] = 0
    known = (conn.root_ids[rid_idx] == columns.root_ids) & np.isin(columns.hemisphere, eyes)
    for eye in eyes:
        m = known & (columns.hemisphere == eye)
        col_key[rid_idx[m]] = EYE_CODE[eye] * (1 << 20) + columns.column_id[m].astype(np.int64)

    is_type = np.isin(conn.cell_type, opt["types"])
    direct = is_type & (col_key >= 0)
    need = is_type & (col_key < 0)
    partner_ok = np.isin(conn.cell_type, opt["partner_types"]) & (col_key >= 0)
    via = _partner_columns(conn, need, partner_ok, col_key)
    placed_key = col_key.copy()
    placed_key[need] = via[need]
    ok = is_type & (placed_key >= 0)
    n_placed = int(ok.sum())
    if active is not None:
        ok &= np.asarray(active, dtype=bool)
    idx = np.flatnonzero(ok).astype(np.int64)
    key = placed_key[idx]
    eye_code = (key >> 20).astype(np.int8)
    column_id = (key & ((1 << 20) - 1)).astype(np.int32)

    # eye maps (the 'weighted' binning balances photoreceptors per square instead of columns)
    maps = {}
    for eye in eyes:
        m = eye_code == EYE_CODE[eye]
        cids, cnt = np.unique(column_id[m], return_counts=True)
        maps[eye] = build_eye_map(columns, eye, opt["effective_field"], opt["binning"],
                                  column_weight=dict(zip(cids.tolist(), cnt.tolist())))

    square = np.zeros(len(idx), dtype=np.int8)
    uv = np.zeros((len(idx), 2), dtype=np.float32)
    for eye in eyes:
        em = maps[eye]
        m = eye_code == EYE_CODE[eye]
        if not m.any():
            continue
        pos = np.searchsorted(em.column_id, column_id[m])
        pos[pos >= em.n] = 0
        assert np.all(em.column_id[pos] == column_id[m]), "column id missing from the eye map"
        square[m] = em.square[pos]
        uv[m] = em.uv[pos]

    # side sanity: a partner-placed photoreceptor should sit in the eye its classification says
    side_code = np.where(conn.side[idx] == "right", 1, np.where(conn.side[idx] == "left", 0, -1))
    side_mismatch = int(np.sum((side_code >= 0) & (side_code != eye_code)))

    for eye in eyes:
        empty = square_stats(maps[eye].square, field_squares(eye, opt["effective_field"]))["empty_squares"]
        if empty:
            warnings.warn(f"retina: {empty} square(s) of the {eye} eye's field receive no column "
                          f"({maps[eye].n} columns, binning={opt['binning']!r}); those squares are invisible "
                          "to that eye", stacklevel=2)
    stats = {
        "options": {k: (list(v) if isinstance(v, tuple) else v) for k, v in opt.items()},
        "columns_per_eye": {eye: int(maps[eye].n) for eye in eyes},
        "columns_per_square": {eye: square_stats(maps[eye].square, field_squares(eye, opt["effective_field"]))
                               for eye in eyes},
        "photoreceptors_per_square_per_eye": {
            eye: square_stats(square[eye_code == EYE_CODE[eye]], field_squares(eye, opt["effective_field"]))
            for eye in eyes},
        "photoreceptors_per_square": square_stats(square),
        "per_type": {},
        "side_mismatch": side_mismatch,
        "placed": n_placed,
        "n": len(idx),
    }
    for t in opt["types"]:
        tm = conn.cell_type == t
        stats["per_type"][t] = {
            "connectome": int(tm.sum()),
            "direct": int((tm & direct).sum()),
            "via_partner": int((tm & need & (via >= 0)).sum()),
            "unplaced": int((tm & need & (via < 0)).sum()),
            "inactive": int((tm & (placed_key >= 0) & ~ok).sum()),
        }
    return RetinaCandidates(idx=idx, square=square, uv=uv, cell_type=conn.cell_type[idx].astype(str),
                            eye=eye_code, column_id=column_id, via_partner=need[idx], stats=stats)


def square_stats(square: np.ndarray, squares: np.ndarray | None = None) -> dict:
    """min / median / max / empty-count of items per square (over all 64 squares, or ``squares``)."""
    counts = np.bincount(np.asarray(square, dtype=np.int64), minlength=64)
    if squares is not None:
        counts = counts[np.asarray(squares, dtype=np.int64)]
    return {"min": int(counts.min()), "median": float(np.median(counts)), "max": int(counts.max()),
            "empty_squares": int((counts == 0).sum()), "total": int(counts.sum())}


def strongest_partner(idx: np.ndarray, pre: np.ndarray, post: np.ndarray, syn: np.ndarray, n: int,
                      exclude: np.ndarray | None = None) -> np.ndarray:
    """Post-synaptic partner receiving the most synapses from each neuron of ``idx`` (-1 when none).

    ``pre / post / syn`` are edge arrays over a common index space of ``n`` neurons (one entry per
    (pre, post) pair); partners flagged in ``exclude`` (bool mask, e.g. the photoreceptors themselves)
    are ignored. Ties -> the smaller partner index.
    """
    idx = np.asarray(idx, dtype=np.int64)
    out = np.full(len(idx), -1, dtype=np.int64)
    if len(idx) == 0 or len(pre) == 0:
        return out
    need = np.zeros(n, dtype=bool)
    need[idx] = True
    e = need[np.asarray(pre)]
    if exclude is not None:
        e &= ~np.asarray(exclude, dtype=bool)[np.asarray(post)]
    if not e.any():
        return out
    p, q, w = np.asarray(pre)[e].astype(np.int64), np.asarray(post)[e].astype(np.int64), np.asarray(syn)[e]
    order = np.lexsort((q, -w.astype(np.float64), p))  # per pre: most synapses first, then smaller post
    p_o = p[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = p_o[1:] != p_o[:-1]
    best = np.full(n, -1, dtype=np.int64)
    best[p_o[first]] = q[order][first]
    return best[idx]


def cap_priority(cand: RetinaCandidates, partner: np.ndarray, total_syn: np.ndarray) -> list[np.ndarray]:
    """Lexicographic priority (all descending) for `balanced_cap` that makes capped photoreceptors
    SHARE their post-synaptic partners: within a square, candidates are grouped by ``partner`` and the
    groups ranked by their summed ``total_syn`` (then smaller partner index), the members of a group
    by ``total_syn``. Photoreceptors without a partner (``-1``) come last. ``total_syn`` is per
    candidate (same order as ``cand.idx``)."""
    partner = np.asarray(partner, dtype=np.int64)
    total_syn = np.asarray(total_syn, dtype=np.float64)
    has = partner >= 0
    key = cand.square.astype(np.int64) * (int(partner.max(initial=0)) + 2) + np.where(has, partner, -1) + 1
    uniq, inv = np.unique(key, return_inverse=True)
    group = np.bincount(inv, weights=total_syn, minlength=len(uniq))[inv]
    group = np.where(has, group, -1.0)  # partnerless: after every group
    return [group, -np.where(has, partner, np.iinfo(np.int64).max // 2).astype(np.float64), total_syn]


def round_robin_order(square: np.ndarray, priority, idx: np.ndarray) -> np.ndarray:
    """Positions ordered round-robin over squares: the best item (``priority`` desc, ``idx`` asc) of
    every square first (squares ascending), then the second best of every square, ... ``priority`` is
    one array or a list of arrays (lexicographic, first = most significant, all descending)."""
    square = np.asarray(square, dtype=np.int64)
    idx = np.asarray(idx)
    n = len(square)
    keys = list(priority) if isinstance(priority, (list, tuple)) else [priority]
    keys = [-np.asarray(k, dtype=np.float64) for k in keys]
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    order = np.lexsort((idx, *reversed(keys), square))
    sq_sorted = square[order]
    rank = np.zeros(n, dtype=np.int64)  # rank within square along the sorted order
    starts = np.flatnonzero(np.r_[True, sq_sorted[1:] != sq_sorted[:-1]])
    rank[order] = np.arange(n) - np.repeat(starts, np.diff(np.r_[starts, n]))
    return np.lexsort((idx, square, rank))


def balanced_cap(cand: RetinaCandidates, priority, max_neurons: int) -> RetinaCandidates:
    """Keep at most ``max_neurons`` candidates, round-robin over squares (deterministic).

    Within a square candidates are ordered by (``priority`` desc, connectome index asc); the k-th of
    every square is taken before the (k+1)-th of any square, so 64 * m candidates cover every square
    with m photoreceptors each (as far as the squares have that many). ``priority`` is one array or a
    list of arrays (lexicographic, first = most significant, every key descending; see `cap_priority`).
    """
    if max_neurons is None or cand.n <= max_neurons:
        return cand
    pick = round_robin_order(cand.square, priority, cand.idx)[:max_neurons]
    keep = np.zeros(cand.n, dtype=bool)
    keep[np.sort(pick)] = True
    out = cand.take(keep)
    out.stats["capped_from"] = int(cand.n)
    out.stats["cap"] = int(max_neurons)
    return out


def hops_to_targets(n: int, pre: np.ndarray, post: np.ndarray, targets: np.ndarray,
                    max_hops: int | None = None) -> np.ndarray:
    """Directed distance (number of synaptic hops along ``pre -> post`` edges) from every neuron to the
    nearest of ``targets``: 0 on the targets, -1 when unreachable (or farther than ``max_hops``)."""
    dist = np.full(n, -1, dtype=np.int64)
    targets = np.asarray(targets, dtype=np.int64)
    if len(targets) == 0:
        return dist
    dist[targets] = 0
    frontier = np.zeros(n, dtype=bool)
    frontier[targets] = True
    pre = np.asarray(pre)
    post = np.asarray(post)
    d = 0
    while frontier.any() and (max_hops is None or d < max_hops):
        d += 1
        cand = np.unique(pre[frontier[post]])
        cand = cand[dist[cand] < 0]
        frontier = np.zeros(n, dtype=bool)
        if len(cand) == 0:
            break
        dist[cand] = d
        frontier[cand] = True
    return dist


def hops_stats(hops: np.ndarray) -> dict:
    """min / median / max hops over the reachable items, plus how many are unreachable (-1)."""
    hops = np.asarray(hops, dtype=np.int64)
    ok = hops[hops >= 0]
    return {"min": int(ok.min()) if len(ok) else None, "median": float(np.median(ok)) if len(ok) else None,
            "max": int(ok.max()) if len(ok) else None, "unreachable": int((hops < 0).sum()), "n": len(hops)}


def _out_csr(n: int, pre: np.ndarray, post: np.ndarray, syn: np.ndarray):
    order = np.lexsort((np.asarray(post), np.asarray(pre)))
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(np.asarray(pre), minlength=n), out=indptr[1:])
    return indptr, np.asarray(post)[order].astype(np.int64), np.asarray(syn)[order].astype(np.float64)


def output_paths(start: np.ndarray, n: int, pre: np.ndarray, post: np.ndarray, syn: np.ndarray,
                 dist: np.ndarray, kept: np.ndarray, tiebreak: np.ndarray, budget: int | None = None,
                 free: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Force-keep a shortest path from every ``start`` neuron to the targets ``dist`` was computed for.

    Walks from each start (in the given order) along ``pre -> post`` edges to a neighbour one hop
    closer to the targets, preferring neighbours already ``kept`` (in the graph anyway), then the
    most synapses, then the smaller ``tiebreak`` (root id); every visited neuron is marked kept (and
    free), so later paths merge into earlier ones. ``budget`` bounds the number of neurons NOT in
    ``free`` (default: ``kept``; the graph builder passes its role neurons, which are kept
    unconditionally, so that every other neuron on a path costs one of the ``max_neurons`` slots) that
    the paths may claim in total: a path that would exceed it is skipped entirely (its start stays
    unconnected). Returns ``(nodes, connected)``: every neuron on an accepted path (sorted, excluding
    the starts; targets included) and a bool per start (False when the start has no path to a target
    at all or was skipped for the budget).
    """
    kept = np.asarray(kept, dtype=bool).copy()
    free = kept.copy() if free is None else np.asarray(free, dtype=bool).copy()
    dist = np.asarray(dist)
    tiebreak = np.asarray(tiebreak)
    start = np.asarray(start, dtype=np.int64)
    indptr, nbr, w = _out_csr(n, pre, post, syn)
    on_path = np.zeros(n, dtype=bool)
    connected = np.zeros(len(start), dtype=bool)
    left = np.inf if budget is None else int(budget)
    for i, s in enumerate(start):
        if dist[s] < 0:
            continue
        path: list[int] = []
        cur = int(s)
        while dist[cur] > 0:
            nb = nbr[indptr[cur]:indptr[cur + 1]]
            ww = w[indptr[cur]:indptr[cur + 1]]
            m = dist[nb] == dist[cur] - 1
            nb, ww = nb[m], ww[m]
            # prefer an already kept neighbour, then the strongest connection, then the smaller root id
            j = np.lexsort((tiebreak[nb], -ww, ~kept[nb]))[0]
            cur = int(nb[j])
            path.append(cur)
        cost = len({v for v in path if not free[v]})
        if cost > left:
            continue
        connected[i] = True
        kept[path] = True
        free[path] = True
        on_path[path] = True
        left -= cost
    return np.flatnonzero(on_path).astype(np.int64), connected


def candidates_to_arrays(cand: RetinaCandidates, new_index: np.ndarray) -> dict:
    """Map candidates (connectome indices) into graph indices via ``new_index`` (-1 = not in graph).

    Returns the `BrainGraph` retina arrays sorted by graph index.
    """
    gi = np.asarray(new_index)[cand.idx]
    keep = gi >= 0
    order = np.argsort(gi[keep], kind="stable")
    sel = np.flatnonzero(keep)[order]
    return {
        "retina_idx": gi[sel].astype(np.int32),
        "retina_square": cand.square[sel].astype(np.int8),
        "retina_uv": cand.uv[sel].astype(np.float32),
        "retina_type": cand.cell_type[sel].astype(str),
        "retina_eye": cand.eye[sel].astype(np.int8),
    }


def retina_meta(cand: RetinaCandidates, arrays: dict, columns: ColumnTable | None, t0: float | None = None) -> dict:
    """The ``meta['retina']`` block describing the mapping."""
    n_ret = int(arrays["retina_idx"].shape[0])
    types, counts = np.unique(arrays["retina_type"], return_counts=True)
    eyes, ecounts = np.unique(arrays["retina_eye"], return_counts=True)
    meta = {"enabled": True, "n_ret": n_ret}
    meta.update({k: v for k, v in cand.stats.items() if k != "n"})
    meta["candidates"] = int(cand.stats.get("n", n_ret))  # placed photoreceptors before graph restriction / cap
    meta.update({
        "types": {str(t): int(k) for t, k in zip(types, counts)},
        "eyes": {("left" if int(e) == 0 else "right"): int(k) for e, k in zip(eyes, ecounts)},
        "photoreceptors_per_square": square_stats(arrays["retina_square"]),  # final (graph-restricted, capped)
        "photoreceptors_per_square_per_eye": {
            eye: square_stats(arrays["retina_square"][arrays["retina_eye"] == code],
                              field_squares(eye, cand.stats.get("options", {}).get("effective_field", "full")))
            for eye, code in EYE_CODE.items() if np.any(arrays["retina_eye"] == code)},
        "coordinates": "u: file a -> h (left eye u<0.5 lateral->frontal, right eye u>0.5 frontal->lateral); "
                       "v: rank 1 -> 8 (ventral -> dorsal); X=(q-p)*sqrt(3)/2, Y=(p+q)/2 from hex (p,q)",
        "square": "rank*8+file in the mover's perspective (planes index), see docs/RETINA.md",
    })
    if columns is not None and columns.source:
        meta["source"] = dict(columns.source)
    if t0 is not None:
        meta["build_time_s"] = round(time.time() - t0, 3)
    return meta


def empty_retina() -> dict:
    return {
        "retina_idx": np.zeros(0, dtype=np.int32),
        "retina_square": np.zeros(0, dtype=np.int8),
        "retina_uv": np.zeros((0, 2), dtype=np.float32),
        "retina_type": np.zeros(0, dtype=str),
        "retina_eye": np.zeros(0, dtype=np.int8),
    }


def build_retina(conn: Connectome, graph: BrainGraph, cfg=None, columns: ColumnTable | None = None) -> dict:
    """Retina arrays for an EXISTING graph: ``{retina_idx, retina_square, retina_uv, retina_type,
    retina_eye, meta}`` (graph indices; only photoreceptors present in ``graph`` with at least one
    outgoing connection are used).

    ``cfg`` is a `GraphConfig` (its ``retina_*`` fields) or None for the defaults; ``columns`` the
    column table (loaded from ``data/connectome`` when None). ``retina_max`` caps the retina, keeping
    the photoreceptors with the most synapses in the graph, balanced over squares.
    """
    from flychess.connectome.load import load_column_assignment

    t0 = time.time()
    if columns is None:
        columns = load_column_assignment(paths.CONNECTOME_DIR)
    new_index = np.full(conn.n, -1, dtype=np.int64)
    new_index[conn.index_of(graph.root_ids)] = np.arange(graph.n)
    rows = np.repeat(np.arange(graph.n), np.diff(graph.csr_indptr))  # graph edges: pre = csr_indices, post = rows
    drives = np.zeros(conn.n, dtype=bool)  # photoreceptors with >= 1 outgoing connection in the graph
    drives[conn.index_of(graph.root_ids)] = np.bincount(graph.csr_indices, minlength=graph.n) > 0
    cand = retina_candidates(conn, cfg, columns, active=(new_index >= 0) & drives)
    cap = cfg.effective_retina_max() if hasattr(cfg, "effective_retina_max") else retina_options(cfg)["max"]
    if cap is not None:
        deg_syn = (np.bincount(graph.csr_indices, weights=graph.syn_count, minlength=graph.n)
                   + np.bincount(rows, weights=graph.syn_count, minlength=graph.n))  # in + out synapses
        gidx = new_index[cand.idx]
        is_cand = np.zeros(graph.n, dtype=bool)
        is_cand[gidx] = True
        partner = strongest_partner(gidx, graph.csr_indices, rows, graph.syn_count, graph.n, exclude=is_cand)
        cand = balanced_cap(cand, cap_priority(cand, partner, deg_syn[gidx]), cap)
    arrays = candidates_to_arrays(cand, new_index)
    arrays["meta"] = retina_meta(cand, arrays, columns, t0)
    arrays["meta"]["hops_to_output"] = hops_stats(
        hops_to_targets(graph.n, graph.csr_indices, rows, graph.output_idx)[arrays["retina_idx"]])
    return arrays
