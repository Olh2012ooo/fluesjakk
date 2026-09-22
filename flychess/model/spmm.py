"""Sparse matrix-matrix product with a memory-safe custom backward (docs/SPEC.md §4).

The fly brain's recurrent step is ``pre = W @ h`` where ``W`` is the (n, n) connectome adjacency
in CSR form (rows = post-synaptic neuron, columns = pre-synaptic neuron) and ``h`` is the hidden
state kept neuron-major, ``(n, B)``. Only the *values* of ``W`` are learnable; its sparsity pattern
is the connectome and never changes.

Why a custom ``autograd.Function``: PyTorch's built-in autograd for ``torch.sparse_csr_tensor``
values densifies the gradient w.r.t. the values into an (n, n) matrix — 134k² × 4 bytes ≈ 73 GiB.
Here the backward is computed sparsely:

* ``dvalues[k] = Σ_b g[post_k, b] · h[pre_k, b]`` — an SDDMM on the same CSR pattern, computed with
  ``torch.sparse.sampled_addmm`` on a zero-valued CSR tensor (cuSPARSE on GPU, MKL on CPU) or, as a
  fallback, by chunked gather + row-wise sums (never materialising more than ``chunk × B`` floats).
* ``dh = Wᵀ @ g`` — an SpMM with a *precomputed transposed CSR* (``crow_t, col_t``) whose values are
  ``values[perm_t]`` (a gather, no sort at run time).

Everything here runs in float32 (or float64 for gradient checks); callers under bf16 autocast get
their inputs upcast. cuSPARSE accepts int32 and int64 indices; int32 measured ~15 % faster and
halves index memory, so it is the default.
"""
from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import torch

from flychess.connectome.graph import BrainGraph

SddmmMode = Literal["auto", "sddmm", "gather"]

# One-time PyTorch notices about the (beta) CSR layout; we rely on it deliberately.
warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")
warnings.filterwarnings("ignore", message="Sparse invariant checks are implicitly disabled")


def transpose_csr(indptr: np.ndarray, indices: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(crow_t, col_t, perm_t)`` of the transposed CSR so that ``values_t = values[perm_t]``.

    ``n`` is the number of COLUMNS of the matrix (= rows of the transpose); the number of rows comes
    from ``indptr`` (rectangular matrices are fine). ``perm_t`` is a stable argsort of the column
    indices; within each transposed row the entries are therefore sorted by original row, i.e. the
    transposed CSR is canonical as well.
    """
    n_rows = int(indptr.shape[0]) - 1
    counts_t = np.bincount(np.asarray(indices, dtype=np.int64), minlength=n)
    crow_t = np.concatenate([[0], np.cumsum(counts_t)]).astype(np.int64)
    perm_t = np.argsort(indices, kind="stable")
    rows = np.repeat(np.arange(n_rows, dtype=np.int64), np.diff(indptr))
    col_t = rows[perm_t]
    return crow_t, col_t, perm_t


def subset_csr(indptr: np.ndarray, indices: np.ndarray, keep: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The sub-CSR made of the entries where ``keep`` (bool, (nnz,)) is True, same rows.

    Returns ``(indptr_sub, indices_sub, sel)`` with ``sel = flatnonzero(keep)`` so that
    ``values_sub = values[sel]``. Row order and the within-row column order are preserved, so a
    canonical CSR stays canonical. Used to split the connectome into ionotropic and modulatory synapses.
    """
    keep = np.asarray(keep, dtype=bool)
    n = int(indptr.shape[0]) - 1
    rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr))
    sel = np.flatnonzero(keep)
    counts = np.bincount(rows[sel], minlength=n)
    indptr_sub = np.concatenate([[0], np.cumsum(counts)]).astype(indptr.dtype)
    return indptr_sub, np.asarray(indices)[sel], sel


class SparseStructure:
    """Immutable CSR pattern of a :class:`BrainGraph` on one device, with its transpose.

    Attributes
    ----------
    n : int                 number of neurons = columns of the matrix (the state ``h`` has ``n`` rows).
    n_rows : int            rows of the matrix — ``n`` for the connectome itself, fewer for a row-compressed
                            sub-matrix (e.g. only the neurons that receive modulatory synapses).
    nnz : int               number of synapses.
    crow, col : Tensor      forward CSR (``index_dtype``), rows = post, cols = pre.
    crow_t, col_t : Tensor  transposed CSR (rows = pre, cols = post).
    perm_t : Tensor(int64)  ``values_t = values[perm_t]``.
    """

    def __init__(
        self,
        graph: BrainGraph | None = None,
        device: torch.device | str = "cpu",
        index_dtype: torch.dtype = torch.int32,
        sddmm_mode: SddmmMode = "auto",
        sddmm_chunk: int = 1 << 19,
        *,
        tensors: dict[str, torch.Tensor] | None = None,
        n: int | None = None,
    ) -> None:
        self.index_dtype = index_dtype
        self.sddmm_mode: SddmmMode = sddmm_mode
        self.sddmm_chunk = int(sddmm_chunk)
        if tensors is not None:
            assert n is not None, "n is required when building from tensors"
            self.n = int(n)
            self.crow = tensors["crow"]
            self.col = tensors["col"]
            self.crow_t = tensors["crow_t"]
            self.col_t = tensors["col_t"]
            self.perm_t = tensors["perm_t"]
        else:
            assert graph is not None, "graph or tensors must be given"
            device = torch.device(device)
            self.n = int(graph.n)
            crow_t, col_t, perm_t = transpose_csr(graph.csr_indptr, graph.csr_indices, self.n)
            self.crow = torch.as_tensor(graph.csr_indptr, dtype=index_dtype).to(device)
            self.col = torch.as_tensor(graph.csr_indices, dtype=index_dtype).to(device)
            self.crow_t = torch.as_tensor(crow_t, dtype=index_dtype).to(device)
            self.col_t = torch.as_tensor(col_t, dtype=index_dtype).to(device)
            self.perm_t = torch.as_tensor(perm_t, dtype=torch.int64).to(device)
        self.nnz = int(self.col.shape[0])
        self.n_rows = int(self.crow.shape[0]) - 1
        self._rows: torch.Tensor | None = None  # lazily built for the gather fallback
        self._sddmm_failed = False

    @classmethod
    def from_csr(cls, indptr: np.ndarray, indices: np.ndarray, n: int, device: torch.device | str = "cpu",
                 index_dtype: torch.dtype = torch.int32) -> SparseStructure:
        """Structure of an arbitrary CSR pattern with ``n`` columns (e.g. a :func:`subset_csr` of the
        connectome, or a row-compressed one: ``len(indptr) - 1`` rows may be fewer than ``n``)."""
        indptr, indices = np.asarray(indptr), np.asarray(indices)
        crow_t, col_t, perm_t = transpose_csr(indptr, indices, int(n))
        device = torch.device(device)
        tensors = {
            "crow": torch.as_tensor(indptr, dtype=index_dtype).to(device),
            "col": torch.as_tensor(indices, dtype=index_dtype).to(device),
            "crow_t": torch.as_tensor(crow_t, dtype=index_dtype).to(device),
            "col_t": torch.as_tensor(col_t, dtype=index_dtype).to(device),
            "perm_t": torch.as_tensor(perm_t, dtype=torch.int64).to(device),
        }
        return cls(tensors=tensors, n=int(n), index_dtype=index_dtype)

    # ---- helpers ------------------------------------------------------------------------------
    @property
    def device(self) -> torch.device:
        return self.col.device

    def tensors(self) -> dict[str, torch.Tensor]:
        return {"crow": self.crow, "col": self.col, "crow_t": self.crow_t, "col_t": self.col_t, "perm_t": self.perm_t}

    def to(self, device: torch.device | str) -> SparseStructure:
        device = torch.device(device)
        if device == self.device:
            return self
        return SparseStructure(
            tensors={k: v.to(device) for k, v in self.tensors().items()},
            n=self.n,
            index_dtype=self.index_dtype,
            sddmm_mode=self.sddmm_mode,
            sddmm_chunk=self.sddmm_chunk,
        )

    def csr(self, values: torch.Tensor) -> torch.Tensor:
        """Forward CSR tensor ``W`` with the given values (no copy of the indices)."""
        return torch.sparse_csr_tensor(self.crow, self.col, values, (self.n_rows, self.n))

    def csr_t(self, values: torch.Tensor) -> torch.Tensor:
        """Transposed CSR tensor ``Wᵀ`` for forward-ordered ``values``."""
        return torch.sparse_csr_tensor(self.crow_t, self.col_t, values[self.perm_t], (self.n, self.n_rows))

    def rows(self) -> torch.Tensor:
        """Row (post-synaptic) index of every synapse, int64 — built lazily (gather fallback only)."""
        if self._rows is None:
            crow = self.crow.to(torch.int64)
            self._rows = torch.repeat_interleave(torch.arange(self.n_rows, device=self.device), crow[1:] - crow[:-1])
        return self._rows

    def dense(self, values: torch.Tensor) -> torch.Tensor:
        """Dense (n, n) copy of ``W`` — reference for tests only."""
        return self.csr(values).to_dense()

    # ---- products -----------------------------------------------------------------------------
    def spmm(self, values: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """``W @ h`` with ``h: (n, B)`` → ``(n_rows, B)``, differentiable w.r.t. both arguments."""
        return SpMM.apply(values, h, self)

    def spmm_t(self, values: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """``Wᵀ @ g`` (no autograd) — used by the backward pass."""
        return torch.mm(self.csr_t(values), g)

    def sddmm(self, g: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """``dvalues[k] = Σ_b g[post_k, b] · h[pre_k, b]`` for ``g, h: (n, B)`` → ``(nnz,)``."""
        mode = self.sddmm_mode
        if mode == "auto" and not self._sddmm_failed:
            try:
                return self._sddmm_sampled(g, h)
            except (RuntimeError, NotImplementedError):
                self._sddmm_failed = True  # remember; fall through to the gather path
        if mode == "sddmm":
            return self._sddmm_sampled(g, h)
        return self._sddmm_gather(g, h)

    def _sddmm_sampled(self, g: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        zero = torch.sparse_csr_tensor(self.crow, self.col, torch.zeros(self.nnz, dtype=g.dtype, device=g.device),
                                       (self.n_rows, self.n))
        return torch.sparse.sampled_addmm(zero, g, h.t()).values()

    def _sddmm_gather(self, g: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        rows = self.rows()
        col = self.col.to(torch.int64)
        out = torch.empty(self.nnz, dtype=g.dtype, device=g.device)
        for s in range(0, self.nnz, self.sddmm_chunk):
            e = min(self.nnz, s + self.sddmm_chunk)
            out[s:e] = (g.index_select(0, rows[s:e]) * h.index_select(0, col[s:e])).sum(1)
        return out


class SpMM(torch.autograd.Function):
    """``out = W(values) @ h`` — CSR forward, sparse SDDMM / transposed-CSR backward."""

    @staticmethod
    def forward(ctx, values: torch.Tensor, h: torch.Tensor, structure: SparseStructure) -> torch.Tensor:  # type: ignore[override]
        if h.dtype in (torch.float16, torch.bfloat16):
            h = h.float()
        if values.dtype != h.dtype:
            values = values.to(h.dtype)
        h = h.contiguous()
        with torch.autocast(device_type=h.device.type, enabled=False):
            out = torch.mm(structure.csr(values), h)
        ctx.save_for_backward(values, h)
        ctx.structure = structure
        return out

    @staticmethod
    def backward(ctx, g: torch.Tensor):  # type: ignore[override]
        values, h = ctx.saved_tensors
        structure: SparseStructure = ctx.structure
        g = g.to(h.dtype).contiguous()
        d_values = d_h = None
        with torch.autocast(device_type=h.device.type, enabled=False):
            if ctx.needs_input_grad[0]:
                d_values = structure.sddmm(g, h)
            if ctx.needs_input_grad[1]:
                d_h = structure.spmm_t(values, g)
        return d_values, d_h, None


def spmm(values: torch.Tensor, h: torch.Tensor, structure: SparseStructure) -> torch.Tensor:
    """Functional alias of :meth:`SparseStructure.spmm`."""
    return SpMM.apply(values, h, structure)


def spmm_dense_reference(values: torch.Tensor, h: torch.Tensor, structure: SparseStructure) -> torch.Tensor:
    """Dense reference ``W @ h`` using ordinary autograd — tests only (O(n²) memory)."""
    rows = structure.rows()
    col = structure.col.to(torch.int64)
    w = torch.zeros(structure.n_rows, structure.n, dtype=values.dtype, device=values.device)
    w = w.index_put((rows, col), values)  # differentiable scatter of the values
    return w @ h
