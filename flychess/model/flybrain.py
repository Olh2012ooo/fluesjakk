"""FlyBrain: the fruit-fly connectome as a recurrent chess-playing network (docs/SPEC.md §4).

Every neuron of the :class:`BrainGraph` is a leaky rate unit; every synapse of the connectome is a
weight whose *existence* and *sign* (Dale's law) are fixed and whose magnitude is learned. The board
is injected each timestep into the sensory ``input_idx`` neurons (dense ``w_in``) and/or into the
photoreceptors of the graph's retina (``retina_idx``: each looks at one board square through
``w_ret``), and the move / value heads read the activity of the ``output_idx`` (descending & motor)
neurons at the readout timesteps (optionally plus a linear summary of the ``central`` neurons).
There is no other pathway from board to move: the fly brain is the player.

Dynamics (``h`` is kept neuron-major ``(n, B)`` internally, batch-first at the module boundary)::

    h_0 = 0
    for t in range(steps):
        ion  = W_ion @ h_t                                  # ionotropic synapses (all of them without neuromod)
        pre  = ion * (1 + tanh(W_mod @ h_t))  [neuromod]    # DA / SER / OCT synapses gate multiplicatively
        pre += bias ; pre[input_idx] += x @ w_in.T + b_in ; pre[retina_idx[k]] += w_ret[k] . planes[:, sq_k] + b_ret[k]
        h_{t+1} = (1 - a) * h_t + a * act(pre)              # a = sigmoid(leak_logit), per neuron
    feat  = concat_{t in readout_steps} h_t[output_idx]  (+ central_proj(h_T[central_idx]))
    policy_logits = policy_head(feat) ;  value = tanh(value_head(feat))
"""
from __future__ import annotations

import math
import os
import warnings
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from flychess.connectome.graph import BrainGraph
from flychess.model.config import BrainConfig
from flychess.model.spmm import SparseStructure, subset_csr

_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16, "float64": torch.float64}
_MIN_MAGNITUDE = 1e-6  # |w| floor before inverse_softplus (log1p(syn_count) is > 0 anyway)
MODULATORY_NT: tuple[str, ...] = ("DA", "SER", "OCT")   # presynaptic transmitters whose synapses gate (neuromod)
NUM_SQUARES = 64


def inverse_softplus(x: Tensor) -> Tensor:
    """``y`` such that ``softplus(y) == x`` for ``x > 0``: ``y = x + log(-expm1(-x))``."""
    return x + torch.log(-torch.expm1(-x))


def _gelu_tanh(t: Tensor) -> Tensor:
    """GELU with the tanh approximation (module-level so that models stay picklable)."""
    return F.gelu(t, approximate="tanh")


class _SatReLUFn(torch.autograd.Function):
    """``sat * tanh(relu(t) / sat)`` saving only ``t`` for backward (one tensor per timestep instead of four)."""

    @staticmethod
    def forward(ctx, t: Tensor, sat: float) -> Tensor:
        ctx.save_for_backward(t)
        ctx.sat = sat
        return torch.tanh(torch.relu(t) * (1.0 / sat)).mul_(sat)

    @staticmethod
    def backward(ctx, g: Tensor):
        (t,) = ctx.saved_tensors
        y = torch.tanh(torch.relu(t) * (1.0 / ctx.sat))
        return g * (1.0 - y * y) * (t > 0), None


class SatReLU(nn.Module):
    """Saturating rectifier ``sat * tanh(relu(x) / sat)``: non-negative rates with a ceiling of ``sat``.

    Biologically a firing rate saturates; numerically it keeps the recurrent activity bounded, which
    an unbounded ReLU in an excitatory-loop network does not (activity drifted 65 -> 300 over one epoch).
    Linear (slope 1) for small inputs, so Dale's-law signs keep their meaning.
    """

    def __init__(self, sat: float) -> None:
        super().__init__()
        self.sat = float(sat)

    def forward(self, t: Tensor) -> Tensor:
        return _SatReLUFn.apply(t, self.sat)

    def extra_repr(self) -> str:
        return f"sat={self.sat}"


def activation_fn(name: str, sat: float = 10.0):
    """The recurrent non-linearity. GELU uses the tanh approximation so that JS/numpy can match it."""
    if name == "relu":
        return torch.relu
    if name == "satrelu":
        return SatReLU(sat)
    if name == "tanh":
        return torch.tanh
    if name == "gelu":
        return _gelu_tanh
    raise ValueError(f"unknown activation {name!r}")


def activation_module(name: str, sat: float = 10.0) -> nn.Module:
    """``nn.Module`` version of :func:`activation_fn` (same math), used inside the value MLP."""
    if name == "relu":
        return nn.ReLU()
    if name == "satrelu":
        return SatReLU(sat)
    if name == "tanh":
        return nn.Tanh()
    if name == "gelu":
        return nn.GELU(approximate="tanh")
    raise ValueError(f"unknown activation {name!r}")


def _act_plain(name: str, sat: float):
    """Plain (compilable) elementwise activation used inside the fused update functions."""
    if name == "relu":
        return torch.relu
    if name == "tanh":
        return torch.tanh
    if name == "gelu":
        return _gelu_tanh
    inv, s = 1.0 / float(sat), float(sat)
    return lambda z: torch.tanh(torch.relu(z) * inv) * s


def _make_update(name: str, sat: float):
    act = _act_plain(name, sat)

    def _update(h: Tensor, pre: Tensor, bias: Tensor, a: Tensor) -> Tensor:
        return torch.lerp(h, act(pre + bias), a)

    return _update


class _GateRowsFn(torch.autograd.Function):
    """``pre[rows] *= 1 + tanh(mod)`` IN PLACE on the (n, B) pre-activation (``mod: (len(rows), B)``).

    The neuromodulatory gate only touches the rows that receive DA/SER/OCT synapses; doing it in place
    on the SpMM output (which SpMM does not save) costs one small gather/scatter instead of several
    full (n, B) passes, and the backward saves only the two ``(rows, B)`` tensors it needs.
    """

    @staticmethod
    def forward(ctx, pre: Tensor, mod: Tensor, rows: Tensor) -> Tensor:
        y = torch.tanh(mod)
        ion_rows = pre.index_select(0, rows)
        pre.index_copy_(0, rows, ion_rows * (1.0 + y))
        ctx.save_for_backward(ion_rows, y, rows)
        ctx.mark_dirty(pre)
        return pre

    @staticmethod
    def backward(ctx, g: Tensor):
        ion_rows, y, rows = ctx.saved_tensors
        g_rows = g.index_select(0, rows)
        d_mod = g_rows * ion_rows * (1.0 - y * y)
        g_pre = g.index_copy(0, rows, g_rows * (1.0 + y))
        return g_pre, d_mod, None


def modulatory_edge_mask(graph: BrainGraph) -> np.ndarray | None:
    """Boolean ``(nnz,)`` mask of the synapses whose PRE-synaptic neuron releases DA / SER / OCT
    (``graph.nt_type``, canonical edge order), or ``None`` when the graph carries no ``nt_type``."""
    nt = getattr(graph, "nt_type", None)
    if nt is None or np.size(nt) != int(graph.n):
        return None
    return np.isin(np.asarray(nt).astype(str)[np.asarray(graph.csr_indices)], MODULATORY_NT)


def retina_slots(retina_square: np.ndarray) -> tuple[np.ndarray, int]:
    """Group the photoreceptors by board square for a batched matmul.

    Returns ``(slot, max_k)``: photoreceptor ``k`` sits at row ``slot[k] = square * max_k + j`` of a
    ``(64 * max_k, NUM_PLANES)`` padded weight matrix, ``j`` being its rank among the photoreceptors
    of the same square and ``max_k`` the largest number of photoreceptors on one square.
    """
    sq = np.asarray(retina_square).astype(np.int64)
    if sq.size == 0:
        return np.zeros(0, dtype=np.int64), 0
    order = np.argsort(sq, kind="stable")
    counts = np.bincount(sq, minlength=NUM_SQUARES)
    starts = np.concatenate([[0], np.cumsum(counts)])
    slot = np.empty(sq.size, dtype=np.int64)
    slot[order] = np.arange(sq.size) - starts[sq[order]]
    return sq * int(counts.max()) + slot, int(counts.max())


def compute_ordering(graph: BrainGraph):
    """Reverse-Cuthill-McKee neuron permutation for the recurrent loop.

    Returns ``(node_perm, edge_perm, structure)`` where ``node_perm[i]`` is the canonical index of the
    i-th neuron in compute order, ``edge_perm`` maps compute-order CSR entries to canonical entries
    (``values_c = values[edge_perm]``) and ``structure`` is the compute-order :class:`SparseStructure`
    (CPU). ``(None, None, None)`` when disabled (``FLYCHESS_REORDER=0``) or scipy is unavailable.
    Measured on the full brain: the three sparse kernels per timestep go from 7.6 ms to 5.7 ms.
    """
    if os.environ.get("FLYCHESS_REORDER", "1") == "0" or graph.nnz == 0:
        return None, None, None
    try:
        import scipy.sparse as sp
        from scipy.sparse.csgraph import reverse_cuthill_mckee
    except ImportError:  # pragma: no cover
        return None, None, None
    n = int(graph.n)
    m = sp.csr_matrix((np.arange(graph.nnz, dtype=np.int64) + 1, graph.csr_indices, graph.csr_indptr), shape=(n, n))
    perm = np.asarray(reverse_cuthill_mckee(m + m.T, symmetric_mode=True), dtype=np.int64)
    mp = m[perm][:, perm].tocsr()
    mp.sort_indices()
    edge_perm = np.asarray(mp.data, dtype=np.int64) - 1
    return perm, edge_perm, SparseStructure.from_csr(mp.indptr.astype(np.int32), mp.indices.astype(np.int32), n)


_COMPILED: dict[tuple[str, float], Any] = {}


def _compiled_update(activation: str, sat: float):
    """One ``torch.compile``d update function per (activation, sat); ``dynamic=True`` so that any batch
    size (training, evaluation, MCTS leaves) reuses the same kernels."""
    key = (activation, float(sat) if activation == "satrelu" else 0.0)
    if key not in _COMPILED:
        _COMPILED[key] = torch.compile(_make_update(activation, sat), dynamic=True)
    return _COMPILED[key]


class FlyBrain(nn.Module):
    """Recurrent network on the fly connectome with policy and value heads.

    Parameters
    ----------
    syn_gain (nnz,)          synapse magnitude logits over ALL synapses in canonical graph order;
                             ``w = sign * softplus(syn_gain)`` (dale=True) or ``w = syn_gain`` (dale=False).
                             With ``neuromod`` the DA/SER/OCT synapses are always ``softplus(syn_gain)`` (positive
                             gains, whatever ``dale`` says) — the split is a fixed index partition of this one
                             parameter, so checkpoints and exports do not depend on it.
    bias (n,)                per-neuron bias, init 0.
    leak_logit (n,)          per-neuron leak logit, init ``logit(alpha)``.
    w_in (n_in, input_dim), b_in (n_in,)   board projection into the sensory neurons (``sensory_input``;
                             absent — not created, not saved — when ``sensory_input=False``).
    w_ret (n_ret, num_planes), b_ret (n_ret,)  per-photoreceptor plane weights (``vision``); photoreceptor
                             ``k`` receives ``w_ret[k] . planes[:, retina_square[k]] + b_ret[k]``.
                             Init ``N(0, 1/sqrt(num_planes))`` / 0.
    central_proj             ``Linear(n_central, central_dim)`` on the final activity of the ``central``
                             neurons (``central_dim > 0``), concatenated to the head input.
    policy_head              ``Linear(head_in, num_moves)`` with ``head_in = n_out * len(readout_steps) + central_dim``.
    value_head               ``Linear(head_in, value_hidden) → act → Linear(value_hidden, 1)`` (or ``Linear(head_in, 1)``),
                             followed by ``tanh``. ``act`` is the *same* non-linearity as the recurrence
                             (``config.activation``); the export records it as ``header['value_activation']``
                             so the JS/numpy engines never have to guess it.
    ``syn_gain``, ``bias`` and ``leak_logit`` are always float32 regardless of ``config.dtype`` (which
    only sets the dtype of ``w_in``/``b_in``/``w_ret``/``b_ret`` and the heads): under bf16 their
    optimiser steps would be smaller than an ulp and the connectome would never learn.
    Buffers: ``sign (nnz,) float``, ``input_idx (n_in,) int64``, ``output_idx (n_out,) int64``,
    ``syn_count (nnz,) float`` (persistent); plus the non-persistent, graph-determined index data: the
    canonical CSR pattern (``csr_indptr``, ``csr_indices``, ``crow_t``, ``col_t``, ``perm_t``), the
    compute-order copies (``c_*``, ``node_perm``, ``node_inv``, ``edge_perm``), the ionotropic /
    modulatory split (``ion_*``, ``mod_*``, ``modr_*`` + ``mod_rows``/``mod_cols``, ``ion_edges``, ``mod_edges``,
    ``mod_mask``), the retina
    (``retina_idx``, ``retina_square``, ``ret_slot``) and ``central_idx``.
    """

    def __init__(self, graph: BrainGraph, config: BrainConfig) -> None:
        super().__init__()
        self.config = config
        self.n = int(graph.n)
        self.nnz = int(graph.nnz)
        self.n_in = int(graph.n_in)
        self.n_out = int(graph.n_out)
        self.steps = int(config.steps)
        self.dale = bool(config.dale)
        self.act = activation_fn(config.activation, config.sat)
        self.fused = os.environ.get("FLYCHESS_FUSED", "1") != "0"   # torch.compile'd recurrent update on CUDA
        self._update_fn = None
        dtype = _DTYPES[config.dtype]
        self.num_planes = int(config.input_dim // NUM_SQUARES)
        self.readout_steps: tuple[int, ...] = tuple(config.effective_readout_steps)
        self.sensory_input = bool(config.sensory_input)

        # ---- fixed connectome data (buffers) ----
        self.register_buffer("sign", torch.as_tensor(graph.sign, dtype=torch.float32))
        self.register_buffer("syn_count", torch.as_tensor(graph.syn_count, dtype=torch.float32))
        self.register_buffer("input_idx", torch.as_tensor(graph.input_idx, dtype=torch.int64))
        self.register_buffer("output_idx", torch.as_tensor(graph.output_idx, dtype=torch.int64))
        structure = SparseStructure(graph, device="cpu")
        for name, tensor in structure.tensors().items():
            self.register_buffer({"crow": "csr_indptr", "col": "csr_indices"}.get(name, name), tensor, persistent=False)
        # Cache-friendly compute ordering: the recurrent loop runs on a reverse-Cuthill-McKee permutation
        # of the neurons (the sparse kernels gather one 1.5 KB state row per connection; neighbours
        # that sit close in memory hit L2). Parameters, buffers, checkpoints and the export keep the
        # canonical graph order — the permutation is applied on the way in and out of the loop only.
        node_perm, edge_perm, compute = compute_ordering(graph)
        self.reordered = node_perm is not None
        if self.reordered:
            self.register_buffer("node_perm", torch.as_tensor(node_perm, dtype=torch.int64), persistent=False)
            self.register_buffer("node_inv", torch.argsort(self.node_perm), persistent=False)
            self.register_buffer("edge_perm", torch.as_tensor(edge_perm, dtype=torch.int64), persistent=False)
            for name, tensor in compute.tensors().items():
                self.register_buffer(f"c_{name}", tensor, persistent=False)
        self._structures: dict[tuple[str, torch.device], SparseStructure] = {}
        # Recompute each recurrent step in backward instead of storing its activations: ~2x less
        # activation memory for ~+25 % time. Set by the trainer when batch size / VRAM demands it.
        self.grad_checkpoint: bool = False

        # ---- neuromodulation: split the synapses by the presynaptic transmitter ----
        self.neuromod = bool(config.neuromod)
        self.nnz_mod = 0
        if self.neuromod:
            mask = modulatory_edge_mask(graph)
            if mask is None:
                warnings.warn("neuromod=True but the graph has no nt_type: no modulatory synapses (plain dynamics)")
                mask = np.zeros(self.nnz, dtype=bool)
            self.nnz_mod = int(mask.sum())
            self.register_buffer("mod_mask", torch.as_tensor(mask), persistent=False)
            # the loop's CSRs (compute order when reordered) split into ionotropic / modulatory sub-CSRs;
            # ``ion_edges`` / ``mod_edges`` gather their values straight from the canonical ``w``
            if self.reordered:
                loop_indptr, loop_indices = _np(self.c_crow), _np(self.c_col)
                loop_to_canonical, loop_mask = edge_perm, mask[edge_perm]
            else:
                loop_indptr, loop_indices = np.asarray(graph.csr_indptr), np.asarray(graph.csr_indices)
                loop_to_canonical, loop_mask = np.arange(self.nnz, dtype=np.int64), mask
            for kind, keep in (("ion", ~loop_mask), ("mod", loop_mask)):
                indptr_s, indices_s, sel = subset_csr(loop_indptr, loop_indices, keep)
                sub = SparseStructure.from_csr(indptr_s.astype(np.int32), indices_s.astype(np.int32), self.n)
                for name, tensor in sub.tensors().items():
                    self.register_buffer(f"{kind}_{name}", tensor, persistent=False)
                self.register_buffer(f"{kind}_edges", torch.as_tensor(loop_to_canonical[sel], dtype=torch.int64),
                                     persistent=False)
            # The loop multiplies with the modulatory CSR compressed to the rows that receive any
            # modulatory synapse (``mod_rows``, ~8 % of the neurons) and the columns that emit one
            # (``mod_cols``, the DA/SER/OCT neurons, ~1 %): a cuSPARSE SpMM costs about as much as its
            # output whatever its nnz (measured: the 25k-synapse full-size product cost as much as the
            # 2.7M-synapse one), and the gate then only touches those rows.
            counts = np.diff(indptr_s)                       # the last iteration was "mod"
            mod_rows = np.flatnonzero(counts > 0)
            mod_cols = np.unique(indices_s)
            indptr_r = np.concatenate([[0], np.cumsum(counts[mod_rows])]).astype(np.int32)
            col_r = np.searchsorted(mod_cols, indices_s).astype(np.int32)
            compressed = SparseStructure.from_csr(indptr_r, col_r, int(mod_cols.size))
            for name, tensor in compressed.tensors().items():
                self.register_buffer(f"modr_{name}", tensor, persistent=False)
            self.register_buffer("mod_rows", torch.as_tensor(mod_rows, dtype=torch.int64), persistent=False)
            self.register_buffer("mod_cols", torch.as_tensor(mod_cols, dtype=torch.int64), persistent=False)
            self.n_mod_rows, self.n_mod_cols = int(mod_rows.size), int(mod_cols.size)

        # ---- retina (vision) ----
        raw_ret = getattr(graph, "retina_idx", None)            # optional BrainGraph field (older npz: absent)
        retina_idx = np.asarray(raw_ret if raw_ret is not None else np.zeros(0, np.int32)).astype(np.int64)
        self.n_ret = int(retina_idx.size)
        self.vision = bool(config.vision) and self.n_ret > 0
        if bool(config.vision) and not self.vision:
            # The graph builder silently produces no retina when column_assignment.csv.gz is missing:
            # never let that turn into a network that trains for hours without seeing the board.
            if not self.sensory_input:
                raise ValueError("config.sensory_input=False needs a graph with a retina (graph.has_retina is "
                                 "False): the board could not reach the network")
            warnings.warn("config.vision=True but the graph has no retina (graph.has_retina is False): "
                          "running with the dense sensory input (w_in) only")
        if self.vision:
            retina_square = np.asarray(graph.retina_square).astype(np.int64)
            if retina_square.shape != (self.n_ret,) or retina_square.min() < 0 or retina_square.max() >= NUM_SQUARES:
                raise ValueError("graph.retina_square must be (n_ret,) board squares in 0..63")
            slot, max_k = retina_slots(retina_square)
            self.ret_max_k = max_k
            self.register_buffer("retina_idx", torch.as_tensor(retina_idx), persistent=False)
            self.register_buffer("retina_square", torch.as_tensor(retina_square), persistent=False)
            self.register_buffer("ret_slot", torch.as_tensor(slot), persistent=False)
        else:
            self.n_ret = 0

        # ---- central summary features ----
        self.central_dim = int(config.central_dim)
        central_idx = np.flatnonzero(np.asarray(graph.super_class).astype(str) == "central").astype(np.int64)
        self.n_central = int(central_idx.size) if self.central_dim > 0 else 0
        if self.central_dim > 0:
            if self.n_central == 0:
                raise ValueError("central_dim > 0 but the graph has no 'central' neurons")
            self.register_buffer("central_idx", torch.as_tensor(central_idx), persistent=False)
        self.head_in = self.n_out * len(self.readout_steps) + self.central_dim

        # ---- learnable parameters ----
        init_gain = self.init_syn_gain(self.syn_count, self.csr_indptr, config.weight_init_scale)
        if not self.dale:
            free = self.sign * F.softplus(init_gain)  # free-sign weights start at the Dale init
            init_gain = torch.where(self.mod_mask, init_gain, free) if self.neuromod else free
        # The recurrent parameters are always float32, whatever ``config.dtype`` says: the forward
        # upcasts them anyway (see ``effective_weights``/``leak``), and in bf16 the initial gains
        # (~ -2.5..-4.5, ulp ~ 0.016..0.03) would swallow every optimiser step (~ lr = 1e-3), silently
        # freezing the whole connectome while biases/heads keep learning. ``config.dtype`` only
        # governs the dense parts (``w_in``/``b_in`` and the heads); prefer float32 + autocast.
        self.syn_gain = nn.Parameter(init_gain.float())
        # Homeostatic intrinsic gain per (post-synaptic) neuron, multiplying its whole ionotropic input.
        # Always present in the state dict (zeros = gain 1) so checkpoints stay interchangeable.
        self.homeostatic = bool(getattr(config, "homeostatic", False))
        if self.homeostatic:
            self.log_gain = nn.Parameter(torch.zeros(self.n, dtype=torch.float32))
        else:
            self.register_buffer("log_gain", torch.zeros(self.n, dtype=torch.float32))   # gain 1, not trained
        self.register_buffer("edge_row", torch.repeat_interleave(
            torch.arange(self.n, dtype=torch.int64), (self.csr_indptr[1:] - self.csr_indptr[:-1]).to(torch.int64)),
            persistent=False)
        self.bias = nn.Parameter(torch.zeros(self.n, dtype=torch.float32))
        self.leak_logit = nn.Parameter(
            torch.full((self.n,), math.log(config.alpha / (1 - config.alpha)), dtype=torch.float32)
        )
        if self.sensory_input:   # a vision-only model has no w_in / b_in at all (nothing to train or save)
            self.w_in = nn.Parameter(torch.empty(self.n_in, config.input_dim, dtype=dtype))
            self.b_in = nn.Parameter(torch.zeros(self.n_in, dtype=dtype))
            nn.init.kaiming_uniform_(self.w_in, a=math.sqrt(5))
        self.policy_head = nn.Linear(self.head_in, config.num_moves, dtype=dtype)
        if config.value_hidden > 0:
            self.value_head = nn.Sequential(
                nn.Linear(self.head_in, config.value_hidden, dtype=dtype),
                activation_module(config.activation, config.sat),
                nn.Linear(config.value_hidden, 1, dtype=dtype),
            )
        else:
            self.value_head = nn.Linear(self.head_in, 1, dtype=dtype)
        # the optional parts draw their random init last, so a seeded model's w_in / heads do not depend
        # on whether the graph has a retina or the config asks for central features
        if self.vision:
            self.w_ret = nn.Parameter(torch.randn(self.n_ret, self.num_planes, dtype=dtype) / math.sqrt(self.num_planes))
            self.b_ret = nn.Parameter(torch.zeros(self.n_ret, dtype=dtype))
        if self.central_dim > 0:
            self.central_proj = nn.Linear(self.n_central, self.central_dim, dtype=dtype)

    # ---- initialisation ------------------------------------------------------------------------
    @staticmethod
    def init_syn_gain(syn_count: Tensor, csr_indptr: Tensor, scale: float = 1.0) -> Tensor:
        """Initial ``syn_gain`` from synapse counts.

        ``|w_k| = scale · log1p(c_k) / rownorm`` with ``rownorm = (1/n) Σ_k log1p(c_k)`` — the mean over
        *all* neurons (zero in-degree included) of the row-sum of ``log1p(c)`` — so that the mean
        over neurons of ``Σ_pre |w[post, pre]|`` equals ``scale`` (1.0 by default): each neuron's
        total incoming drive is O(1), a crude spectral-radius control. Magnitudes are floored at
        1e-6 before ``inverse_softplus`` so the logit stays finite.
        """
        n = int(csr_indptr.shape[0]) - 1
        lc = torch.log1p(syn_count.float())
        rownorm = lc.sum() / max(n, 1)
        rownorm = torch.clamp(rownorm, min=_MIN_MAGNITUDE)
        magnitude = torch.clamp(scale * lc / rownorm, min=_MIN_MAGNITUDE)
        return inverse_softplus(magnitude)

    # ---- structure -----------------------------------------------------------------------------
    def _structure_named(self, prefix: str) -> SparseStructure:
        """CSR views ``<prefix>crow`` ... on the current device (rebuilt after ``.to(device)``)."""
        dev = self.csr_indices.device
        key = (prefix, dev)
        if key not in self._structures:
            names = {"crow": "csr_indptr", "col": "csr_indices"} if prefix == "" else {}
            tensors = {k: getattr(self, names.get(k, f"{prefix}{k}"))
                       for k in ("crow", "col", "crow_t", "col_t", "perm_t")}
            n_cols = self.n_mod_cols if prefix == "modr_" else self.n
            self._structures[key] = SparseStructure(tensors=tensors, n=n_cols)
        return self._structures[key]

    def structure(self) -> SparseStructure:
        """The :class:`SparseStructure` the recurrent loop multiplies with: all synapses (compute order
        when reordered), or only the ionotropic ones under ``neuromod``."""
        if self.neuromod:
            return self._structure_named("ion_")
        return self._structure_named("c_" if self.reordered else "")

    def mod_structure(self, compressed: bool = False) -> SparseStructure:
        """The modulatory (DA/SER/OCT) sub-CSR in the loop's neuron order (``neuromod`` only): all ``n``
        rows (export / reference), or ``compressed=True`` for the ``(n_mod_rows, n_mod_cols)`` version
        the loop uses (row ``i`` = neuron ``mod_rows[i]``, column ``j`` = neuron ``mod_cols[j]``)."""
        if not self.neuromod:
            raise RuntimeError("mod_structure() needs config.neuromod=True")
        return self._structure_named("modr_" if compressed else "mod_")

    def canonical_structure(self) -> SparseStructure:
        """CSR views in the graph's own neuron order (tests / reference computations)."""
        return self._structure_named("")

    # ---- weights -------------------------------------------------------------------------------
    def effective_weights(self) -> Tensor:
        """Signed synaptic weights ``w (nnz,)`` in float32, ordered like ``graph.csr_indices``.

        Under ``neuromod`` the modulatory synapses are ``softplus(syn_gain)`` (positive gains) even with
        ``dale=False``; with ``dale=True`` that is what the sign rule (+1 for DA/SER/OCT) gives anyway."""
        gain = self.syn_gain.float()
        if self.dale:
            w = self.sign * F.softplus(gain)
        elif self.neuromod:
            w = torch.where(self.mod_mask, F.softplus(gain), gain)
        else:
            w = gain
        if self.homeostatic or bool((self.log_gain != 0).any()):
            g = torch.exp(self.log_gain.float())[self.edge_row]        # post-synaptic neuron's gain per synapse
            if self.neuromod:
                g = torch.where(self.mod_mask, torch.ones_like(g), g)  # gating synapses are not scaled
            w = w * g
        return w

    @torch.no_grad()
    def calibrate_gains(self, x: Tensor, target: float = 0.5, iters: int = 8, min_gain: float = 0.2,
                        max_gain: float = 200.0, min_std: float = 1e-7) -> dict[str, float]:
        """Homeostatic calibration of ``log_gain`` on a batch of positions ``x (B, input_dim)``.

        Each pass runs the network, measures per neuron the standard deviation *across positions* of its
        ionotropic synaptic input (``W @ h_T``), and rescales the neuron's gain towards ``target`` (damped,
        clipped to ``[min_gain, max_gain]``). Neurons whose input does not vary with the position
        (std < ``min_std``) are left alone — only information-carrying pathways are amplified. With the
        retina this is what lets the visual signal reach the central brain: at the row-normalised init it
        attenuates ~20x per synaptic hop. Returns summary statistics of the final gains.
        """
        was_training = self.training
        self.eval()
        try:
            for _ in range(iters):
                _, _, h = self.forward(x, return_activity=True)          # (B, n) canonical order
                w = self.effective_weights()
                if self.neuromod:
                    w = torch.where(self.mod_mask, torch.zeros_like(w), w)
                syn = self.canonical_structure().to(x.device).spmm(w, h.t().contiguous())   # (n, B)
                std = syn.std(dim=1)
                factor = torch.where(std > min_std, target / (std + 1e-12), torch.ones_like(std))
                factor = factor.clamp(0.25, 4.0)                          # damped steps
                new = (torch.exp(self.log_gain) * factor).clamp(min_gain, max_gain)
                self.log_gain.copy_(torch.log(new))
        finally:
            self.train(was_training)
        g = torch.exp(self.log_gain)
        return {"gain_mean": float(g.mean()), "gain_median": float(g.median()), "gain_max": float(g.max()),
                "gain_min": float(g.min()), "frac_amplified": float((g > 1.5).float().mean())}

    def loop_weights(self) -> tuple[Tensor, Tensor | None]:
        """``(w_ion, w_mod)`` as the recurrent loop consumes them (loop CSR order); ``w_mod`` is ``None``
        without ``neuromod`` and ``w_ion`` then holds every synapse."""
        w = self.effective_weights()
        if self.neuromod:
            return w[self.ion_edges], w[self.mod_edges]
        return (w[self.edge_perm] if self.reordered else w), None

    def leak(self) -> Tensor:
        """Per-neuron leak ``a = sigmoid(leak_logit)`` in float32."""
        return torch.sigmoid(self.leak_logit.float())

    @torch.no_grad()
    def round_weights_to_f16_(self) -> FlyBrain:
        """Round every weight that the web export stores as f16 to its f16 value, in place.

        After this the torch forward uses exactly the values the JS/numpy engines see, which is how
        the parity tests and ``export/testvectors.py`` obtain matching numbers. ``syn_gain`` is set to
        ``inverse_softplus(|round_f16(w)|)`` (dale, and the modulatory synapses) or ``round_f16(w)``
        (free sign).
        """
        w = self.effective_weights().half().float()
        logit = inverse_softplus(torch.clamp(w.abs(), min=_MIN_MAGNITUDE))
        if self.dale:
            new = logit
        elif self.neuromod:
            new = torch.where(self.mod_mask, logit, w)
        else:
            new = w
        self.syn_gain.copy_(new.to(self.syn_gain.dtype))
        for lin in self._dense_linears():
            lin.weight.copy_(lin.weight.half().to(lin.weight.dtype))
        if self.sensory_input:
            self.w_in.copy_(self.w_in.half().to(self.w_in.dtype))
        if self.vision:
            self.w_ret.copy_(self.w_ret.half().to(self.w_ret.dtype))
        return self

    def _dense_linears(self) -> list[nn.Linear]:
        lins = [self.policy_head]
        if isinstance(self.value_head, nn.Sequential):
            lins += [m for m in self.value_head if isinstance(m, nn.Linear)]
        else:
            lins.append(self.value_head)
        if self.central_dim > 0:
            lins.append(self.central_proj)
        return lins

    # ---- forward -------------------------------------------------------------------------------
    def retina_input(self, x: Tensor) -> Tensor:
        """Photoreceptor drive ``(n_ret, B)``: ``w_ret[k] . planes[b, :, retina_square[k]] + b_ret[k]``.

        Computed as one batched matmul per board square (photoreceptors grouped by square and padded to
        ``ret_max_k`` per square, see :func:`retina_slots`) so that nothing of size ``B × planes × n_ret``
        is ever materialised or saved for backward."""
        B = x.shape[0]
        w_pad = torch.zeros(NUM_SQUARES * self.ret_max_k, self.num_planes, dtype=self.w_ret.dtype, device=x.device)
        w_pad = w_pad.index_put((self.ret_slot,), self.w_ret)                       # differentiable scatter
        planes = x.to(self.w_ret.dtype).view(B, self.num_planes, NUM_SQUARES).permute(2, 0, 1)   # (64, B, P)
        out = torch.bmm(planes, w_pad.view(NUM_SQUARES, self.ret_max_k, self.num_planes).transpose(1, 2))  # (64, B, k)
        r = out.permute(0, 2, 1).reshape(NUM_SQUARES * self.ret_max_k, B).index_select(0, self.ret_slot)
        return r.float() + self.b_ret.float().unsqueeze(1)

    def forward(self, x: Tensor, return_activity: bool = False) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
        """``x: (B, input_dim)`` → ``(policy_logits (B, num_moves), value (B, 1)[, h_T (B, n)])``.

        The recurrent part always runs in float32 (the SpMM upcasts); under autocast the dense
        input projections and the heads may run in bf16.
        """
        if x.dim() != 2 or x.shape[1] != self.config.input_dim:
            raise ValueError(f"expected x of shape (B, {self.config.input_dim}), got {tuple(x.shape)}")
        B = x.shape[0]
        w_ion, w_mod = self.loop_weights()                        # (nnz_ion,) [, (nnz_mod,)] fp32
        a = self.leak().unsqueeze(1)                              # (n, 1) fp32
        bias = self.bias.float().unsqueeze(1)                     # (n, 1)
        h_in = r_in = None
        if self.sensory_input:
            h_in = F.linear(x.to(self.w_in.dtype), self.w_in, self.b_in).float().t().contiguous()  # (n_in, B)
        if self.vision:
            r_in = self.retina_input(x)                           # (n_ret, B)
        input_idx, output_idx = self.input_idx, self.output_idx
        retina_idx = self.retina_idx if self.vision else None
        central_idx = self.central_idx if self.central_dim > 0 else None
        if self.reordered:                                        # switch to the compute ordering
            a, bias = a[self.node_perm], bias[self.node_perm]
            input_idx, output_idx = self.node_inv[input_idx], self.node_inv[output_idx]
            retina_idx = self.node_inv[retina_idx] if retina_idx is not None else None
            central_idx = self.node_inv[central_idx] if central_idx is not None else None
        if w_mod is not None and self.nnz_mod == 0:
            w_mod = None                                          # neuromod on, but nothing modulatory here

        use_ckpt = self.grad_checkpoint and torch.is_grad_enabled()
        h = torch.zeros(self.n, B, dtype=torch.float32, device=x.device)
        feats = []
        for t in range(self.steps):
            if use_ckpt and t > 0:
                h = checkpoint(self._step, h, w_ion, w_mod, a, bias, h_in, r_in, input_idx, retina_idx,
                               use_reentrant=False)
            else:
                h = self._step(h, w_ion, w_mod, a, bias, h_in, r_in, input_idx, retina_idx, first=t == 0)
            if t + 1 in self.readout_steps:
                feats.append(h.index_select(0, output_idx).t())   # (B, n_out) fp32
        if central_idx is not None:
            feats.append(self.central_proj(h.index_select(0, central_idx).t().to(self.central_proj.weight.dtype)).float())
        feat = feats[0] if len(feats) == 1 else torch.cat(feats, dim=1)
        feat_dense = feat.to(self.policy_head.weight.dtype)
        policy_logits = self.policy_head(feat_dense)
        value = torch.tanh(self.value_head(feat_dense).float())
        if return_activity:
            if self.reordered:
                h = h.index_select(0, self.node_inv)              # back to the canonical neuron order
            return policy_logits, value, h.t()
        return policy_logits, value

    def _step(self, h: Tensor, w_ion: Tensor, w_mod: Tensor | None, a: Tensor, bias: Tensor,
              h_in: Tensor | None, r_in: Tensor | None, input_idx: Tensor, retina_idx: Tensor | None,
              first: bool = False) -> Tensor:
        """One recurrent timestep on the neuron-major state ``h: (n, B)`` (loop ordering)."""
        # W @ h_0 is identically zero (h_0 = 0): skipping the SpMMs on the first step changes nothing.
        pre = torch.zeros_like(h) if first else self.structure().spmm(w_ion, h)
        if w_mod is not None and not first:                         # pre[rows] *= 1 + tanh(W_mod @ h)[rows]
            mod = self.mod_structure(compressed=True).spmm(w_mod, h.index_select(0, self.mod_cols))  # (rows, B)
            pre = _GateRowsFn.apply(pre, mod, self.mod_rows)        # in place on the SpMM output
        if h_in is not None:
            pre = pre.index_add_(0, input_idx, h_in)                # in-place: SpMM does not save its output
        if r_in is not None:
            pre = pre.index_add_(0, retina_idx, r_in)
        return self._apply_update(h, pre, bias, a)

    def _apply_update(self, h: Tensor, pre: Tensor, bias: Tensor, a: Tensor) -> Tensor:
        update = self._fused_update()
        if update is not None:
            return update(h, pre, bias, a)
        return torch.lerp(h, self.act(pre + bias), a)                # (1 - a) * h + a * act(pre + bias)

    def _fused_update(self):
        """``torch.compile``d elementwise update (bias + activation + leak blend in ONE kernel, with a
        fused backward). Eager PyTorch runs it as ~5 passes over the (n, B) state and saves every
        intermediate; measured 7.5 ms per timestep at B=384 — more than the sparse matmul itself.
        CUDA only; falls back to eager if compilation is unavailable. Same math, so checkpoints from
        either path are interchangeable."""
        if self._update_fn is not None or not self.fused or not torch.cuda.is_available():
            return self._update_fn
        try:
            self._update_fn = _compiled_update(self.config.activation, self.config.sat)
        except Exception as exc:  # noqa: BLE001 - depends on the local toolchain
            warnings.warn(f"fused recurrent update unavailable ({exc}); using eager PyTorch")
            self.fused = False
        return self._update_fn

    # ---- convenience ---------------------------------------------------------------------------
    def __getstate__(self) -> dict[str, Any]:
        """Pickle without the per-process caches (compiled update kernels, device structure views)."""
        state = dict(self.__dict__)
        state.update(_update_fn=None, _structures={})
        return state

    @classmethod
    def from_checkpoint(cls, state_dict: Mapping[str, Any], config: BrainConfig, graph: BrainGraph,
                        strict: bool = True) -> FlyBrain:
        """Build a FlyBrain for ``graph``/``config`` and load ``state_dict`` into it.

        A checkpoint written before the retina existed (no ``w_ret``) is loaded with ``vision=False``
        (with a warning) even when the graph has since been rebuilt with a retina: the checkpoint's
        weights never saw one, and the model's ``config`` then says so. (Only with ``sensory_input=True``:
        a vision-only config has no other way in, so the strict load then fails as a config error.)
        A vision-only model (``sensory_input=False``) ignores ``w_in`` / ``b_in`` found in the checkpoint.

        ``input_idx`` / ``output_idx`` are persistent buffers, so the model plays with the CHECKPOINT's
        sensory / motor sets even when the graph on disk has since been rebuilt with different ones
        (e.g. inputs that became photoreceptors): a warning says so, and the export writes the
        model's sets, never the graph's.
        """
        state_dict = dict(state_dict)
        has_retina = getattr(graph, "retina_idx", None) is not None and np.size(graph.retina_idx) > 0
        if config.vision and config.sensory_input and has_retina and "w_ret" not in state_dict:
            warnings.warn("checkpoint has no retina parameters (w_ret): loading it with vision=False")
            config = config.replace(vision=False)
        if not config.sensory_input:
            for name in ("w_in", "b_in"):
                state_dict.pop(name, None)      # written by a sensory model (or an older vision-only one)
        for name in ("input_idx", "output_idx"):
            saved, current = state_dict.get(name), np.asarray(getattr(graph, name))
            if saved is None:
                continue
            saved = _np(torch.as_tensor(saved)).reshape(-1)
            if saved.size != current.size:
                how = f"{saved.size} vs {current.size} neurons"
            elif not np.array_equal(saved, current):
                how = f"{np.setdiff1d(saved, current).size} of {saved.size} neurons are not in the graph's set"
            else:
                continue
            warnings.warn(f"checkpoint {name} differs from the graph's ({how}): the model keeps the checkpoint's "
                          "neurons, so its w_in / heads stay paired with the neurons they were trained on")
        model = cls(graph, config)
        model.load_state_dict(state_dict, strict=strict)
        return model

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Checkpoints written before the homeostatic gain existed: gain 1 everywhere.
        if prefix + "log_gain" not in state_dict:
            state_dict[prefix + "log_gain"] = torch.zeros_like(self.log_gain)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def count_parameters(self) -> dict[str, int]:
        """Trainable parameter counts per group plus ``total``."""
        counts = {
            "syn_gain": self.syn_gain.numel(),
            "bias": self.bias.numel(),
            "leak_logit": self.leak_logit.numel(),
            "policy_head": sum(p.numel() for p in self.policy_head.parameters()),
            "value_head": sum(p.numel() for p in self.value_head.parameters()),
        }
        if self.sensory_input:
            counts["input_proj"] = self.w_in.numel() + self.b_in.numel()
        if self.vision:
            counts["retina_proj"] = self.w_ret.numel() + self.b_ret.numel()
        if self.central_dim > 0:
            counts["central_proj"] = sum(p.numel() for p in self.central_proj.parameters())
        if self.homeostatic:
            counts["log_gain"] = self.log_gain.numel()
        counts["total"] = sum(counts.values())
        return counts

    def extra_repr(self) -> str:
        return (f"n={self.n}, nnz={self.nnz}, n_in={self.n_in}, n_out={self.n_out}, n_ret={self.n_ret}, "
                f"steps={self.steps}, readout_steps={self.readout_steps}, activation={self.config.activation}, "
                f"dale={self.dale}, neuromod={self.neuromod} (nnz_mod={self.nnz_mod}), central_dim={self.central_dim}")


def _np(t: Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


# ---- loss helpers ------------------------------------------------------------------------------
def masked_policy_log_softmax(logits: Tensor, legal_mask: Tensor | None) -> Tensor:
    """Log-softmax over moves with illegal ones (``legal_mask == False``) set to ``-inf`` first.

    Raises ``ValueError`` if a row of ``legal_mask`` has no legal move at all (a terminal position):
    such a row would silently turn into NaNs and poison the whole batch's gradients. Callers (MCTS,
    trainers) must handle terminal positions before asking the brain for a policy.
    """
    logits = logits.float()
    if legal_mask is not None:
        legal_mask = legal_mask.to(torch.bool)
        if not bool(legal_mask.any(dim=-1).all()):  # one small host sync; cheap next to the forward
            raise ValueError("legal_mask has a row with no legal moves (terminal position?)")
        logits = logits.masked_fill(~legal_mask, float("-inf"))
    return F.log_softmax(logits, dim=-1)


def policy_value_loss(
    logits: Tensor,
    value: Tensor,
    target_move: Tensor,
    target_value: Tensor,
    legal_mask: Tensor | None = None,
    value_weight: float = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Cross-entropy on the target move (illegal moves masked) + MSE on the value.

    Returns ``(loss, metrics)`` with ``metrics = {loss, policy_loss, value_loss, top1, top3}`` as
    *detached 0-d tensors on the same device* — no host/device synchronisation happens here, so the
    trainer can keep the GPU queue full and convert only when it logs (:func:`metrics_to_float`,
    one sync for all five; ``MetricsLogger`` also accepts the tensors directly).
    ``target_move`` must be legal whenever ``legal_mask`` is given (otherwise the loss is +inf).
    """
    logp = masked_policy_log_softmax(logits, legal_mask)
    target_move = target_move.to(torch.int64).view(-1)
    policy_loss = F.nll_loss(logp, target_move)
    v = value.float().view(-1)
    value_loss = F.mse_loss(v, target_value.float().view(-1))
    loss = policy_loss + value_weight * value_loss
    with torch.no_grad():
        k = min(3, logp.shape[-1])
        topk = logp.topk(k, dim=-1).indices
        hit = topk == target_move.unsqueeze(1)
        metrics = {
            "loss": loss.detach(),
            "policy_loss": policy_loss.detach(),
            "value_loss": value_loss.detach(),
            "top1": hit[:, 0].float().mean(),
            "top3": hit.any(dim=1).float().mean(),
        }
    return loss, metrics


def metrics_to_float(metrics: Mapping[str, Tensor]) -> dict[str, float]:
    """``{name: float}`` from the tensor metrics of :func:`policy_value_loss` with a single device sync."""
    keys = list(metrics)
    values = torch.stack([metrics[k].detach().float().reshape(()) for k in keys]).tolist()
    return dict(zip(keys, values, strict=True))
