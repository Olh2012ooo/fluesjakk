"""Export a trained FlyBrain to the static-website format (docs/SPEC.md §8).

``brain.json`` (header, JSON) + ``brain.flyb`` (little-endian binary blob, every array 8-byte aligned)
+ ``brain.flyb.gz``. Array order and dtypes are fixed by the SPEC so that ``web/engine/loader.js``
can parse the blob with nothing but the header::

    csr_indptr i32 [n+1] · csr_indices i32 [nnz] · w f16 [nnz] (signed) · bias f32 [n] · alpha f32 [n]
    · input_idx i32 [n_in] · output_idx i32 [n_out] · w_in f16 [n_in | 0, input_dim] · b_in f32 [n_in | 0]
    · policy_w f16 [num_moves, n_out] · policy_b f32 [num_moves] · value_w f16 [hidden|1, n_out]
    · value_b f32 [hidden|1] · (value_w2 f16 [1, hidden] · value_b2 f32 [1] if MLP)
    · positions f16 [n, 3] in [0, 1] · super_class u8 [n] · node_perm i32 [n]
    · retina_idx i32 [n_ret] · retina_square u8 [n_ret] · retina_uv f16 [n_ret, 2] · retina_eye u8 [n_ret]
    · w_ret f16 [n_ret, num_planes] · b_ret f32 [n_ret]
    · mod_indptr i32 [n+1 | 0] · mod_indices i32 [nnz_mod] · w_mod f16 [nnz_mod]
    · central_idx i32 [n_central] · central_w f16 [central_dim, n_central] · central_b f32 [central_dim]
    · retina_type u8 [n_ret]

The arrays after ``node_perm`` are the optional features (retina input, neuromodulation, central
summary); a feature that is off still writes its arrays with a zero length so the parser is uniform,
and a blob written before they existed simply lacks them (feature off). Under ``neuromod`` the main
CSR (``csr_indptr``/``csr_indices``/``w``, ``header.nnz``) holds the ionotropic synapses only and the
DA/SER/OCT synapses live in the ``mod_*`` CSR (``header.nnz_mod``; ``header.nnz_total`` = all).
``input_idx`` / ``output_idx`` are the MODEL's buffers (the checkpoint's sets), not the graph's; with
``sensory_input=false`` ``w_in`` / ``b_in`` have zero rows (``input_idx`` is still written).

``numpy_forward`` is the reference implementation of what the JS engine must compute, using the
rounded weights exactly as stored in the blob.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

# torch and the model classes are needed by export_web alone. Keeping them out of the module
# namespace means read_flyb / numpy_forward — and therefore the cross-language test vectors and
# `fly test-vectors --model-dir` — work with numpy alone, without a GPU stack installed.
if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

    from flychess.connectome.graph import BrainGraph
    from flychess.model.flybrain import FlyBrain

FORMAT_VERSION = 1
ALIGN = 8
# Legend for the u8 super_class array (index = code); "unknown" catches anything not listed.
SUPER_CLASS_LEGEND: tuple[str, ...] = (
    "unknown", "optic", "central", "sensory", "visual_projection", "ascending", "descending",
    "sensory_ascending", "visual_centrifugal", "motor", "endocrine",
)
_NP_DTYPE = {"f16": np.float16, "f32": np.float32, "i32": np.int32, "u8": np.uint8, "i8": np.int8}


def _pad(nbytes: int) -> int:
    return (-nbytes) % ALIGN


def _quantise(x: np.ndarray, quant: str) -> tuple[np.ndarray, str, float | list[float] | None]:
    """Return ``(stored array, dtype tag, scale)`` for a floating weight array.

    ``i8`` is symmetric: ``q = rint(x / scale)`` with ``scale = max|x| / 127``. 2-D arrays (``w_in``,
    ``policy_w``, ``value_w``...) get one scale **per row** (``scale`` is a list of ``shape[0]``
    floats, dequantised as ``q[i, :] * scale[i]`` — ``loader.js`` supports exactly this), so a few
    large rows cannot flush the others to zero. 1-D arrays (the synaptic ``w``) keep a single scale;
    i8 is therefore lossy for heavy-tailed trained synapses and is documented as experimental — f16
    (the default) is exact to 3 decimal digits and is the parity path.
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    if quant == "f16":
        return x.astype(np.float16), "f16", None
    if quant == "f32":
        return x, "f32", None
    if quant == "i8":
        if x.ndim == 2 and x.shape[0] > 1:
            row_scale = np.abs(x).max(axis=1) / 127.0                      # (rows,)
            row_scale = np.where(row_scale > 0, row_scale, 1.0).astype(np.float32)
            q = np.clip(np.rint(x / row_scale[:, None]), -127, 127).astype(np.int8)
            return q, "i8", [float(s) for s in row_scale]
        scale = float(np.abs(x).max()) / 127.0 if x.size else 1.0
        scale = scale or 1.0
        return np.clip(np.rint(x / scale), -127, 127).astype(np.int8), "i8", scale
    raise ValueError(f"quant must be 'f16', 'f32' or 'i8', got {quant!r}")


def _dequantise(x: np.ndarray, entry: dict[str, Any]) -> np.ndarray:
    """Stored array → float32, applying the i8 ``scale`` (scalar, or per-row list for 2-D arrays)."""
    if entry["dtype"] == "i8":
        scale = entry["scale"]
        if isinstance(scale, (list, tuple)):
            rows = len(scale)
            return (x.reshape(rows, -1).astype(np.float32) * np.asarray(scale, np.float32)[:, None]).reshape(x.shape)
        return x.astype(np.float32) * np.float32(scale)
    return x.astype(np.float32)


def normalise_positions(position: np.ndarray) -> np.ndarray:
    """Per-axis min-max normalisation to [0, 1]; NaN (unknown position) → 0.5."""
    pos = np.asarray(position, dtype=np.float32).copy()
    out = np.full_like(pos, 0.5)
    for ax in range(pos.shape[1]):
        col = pos[:, ax]
        ok = np.isfinite(col)
        if ok.sum() == 0:
            continue
        lo, hi = float(col[ok].min()), float(col[ok].max())
        if hi > lo:
            out[ok, ax] = (col[ok] - lo) / (hi - lo)
        else:
            out[ok, ax] = 0.5
    return out


def super_class_codes(super_class: np.ndarray) -> np.ndarray:
    legend = {name: i for i, name in enumerate(SUPER_CLASS_LEGEND)}
    return np.array([legend.get(str(s), 0) for s in super_class], dtype=np.uint8)


def _t(x: torch.Tensor) -> np.ndarray:
    return x.detach().float().cpu().numpy()


def export_web(
    model: FlyBrain,
    graph: BrainGraph,
    out_dir: str | Path,
    quant: str = "f16",
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write ``brain.json``, ``brain.flyb`` and ``brain.flyb.gz`` into ``out_dir``; return the header."""
    import torch

    if quant not in ("f16", "f32", "i8"):
        raise ValueError(f"quant must be 'f16', 'f32' or 'i8', got {quant!r}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    extra_meta = dict(extra_meta or {})
    cfg = model.config
    if graph.n != model.n or graph.nnz != model.nnz:
        raise ValueError("graph does not match the model (n / nnz differ)")

    # ---- gather arrays in SPEC order ----
    arrays: list[tuple[str, np.ndarray, str, float | list[float] | None]] = []

    def add(name: str, arr: np.ndarray, dtype: str, scale: float | list[float] | None = None) -> None:
        arrays.append((name, np.ascontiguousarray(arr), dtype, scale))

    def add_q(name: str, arr: np.ndarray) -> None:
        q, tag, scale = _quantise(arr, quant)
        add(name, q, tag, scale)

    # Neuron order of the blob: the model's cache-friendly compute ordering (reverse Cuthill-McKee)
    # when available — the JS engine does the same gather-bound sparse products and gains the same
    # locality. Every per-neuron / per-connection array below is permuted consistently, so the blob is
    # self-contained; ``node_perm`` maps blob index -> canonical graph index for anyone who needs it.
    if getattr(model, "reordered", False):
        node_perm = _t(model.node_perm).astype(np.int64)
        node_inv = _t(model.node_inv).astype(np.int64)
        edge_perm = _t(model.edge_perm).astype(np.int64)
        csr_indptr, csr_indices = _t(model.c_crow).astype(np.int32), _t(model.c_col).astype(np.int32)
        neuron_order = "rcm"
    else:
        node_perm = np.arange(graph.n, dtype=np.int64)
        node_inv, edge_perm = node_perm, np.arange(graph.nnz, dtype=np.int64)
        csr_indptr, csr_indices = graph.csr_indptr.astype(np.int32), graph.csr_indices.astype(np.int32)
        neuron_order = "canonical"
    w_all = _t(model.effective_weights())
    neuromod = bool(getattr(model, "neuromod", False))
    if neuromod:  # main CSR = ionotropic synapses only; the modulatory ones get their own CSR below
        ion, mod = model._structure_named("ion_"), model._structure_named("mod_")
        csr_indptr, csr_indices = _t(ion.crow).astype(np.int32), _t(ion.col).astype(np.int32)
        w_main = w_all[_t(model.ion_edges).astype(np.int64)]
        mod_indptr, mod_indices = _t(mod.crow).astype(np.int32), _t(mod.col).astype(np.int32)
        w_mod = w_all[_t(model.mod_edges).astype(np.int64)]
    else:
        w_main = w_all[edge_perm]
        mod_indptr, mod_indices, w_mod = np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros(0, np.float32)
    add("csr_indptr", csr_indptr, "i32")
    add("csr_indices", csr_indices, "i32")
    add_q("w", w_main)
    add("bias", _t(model.bias).astype(np.float32)[node_perm], "f32")
    add("alpha", _t(model.leak()).astype(np.float32)[node_perm], "f32")
    # The MODEL's sensory / motor sets (persistent buffers restored from the checkpoint), not the
    # graph's: a graph rebuilt after training (e.g. with a retina) can list different input neurons,
    # and w_in / the heads are paired with the neurons they were trained on.
    input_idx, output_idx = _t(model.input_idx).astype(np.int64), _t(model.output_idx).astype(np.int64)
    sensory_input = bool(getattr(model, "sensory_input", True))
    add("input_idx", node_inv[input_idx].astype(np.int32), "i32")
    add("output_idx", node_inv[output_idx].astype(np.int32), "i32")
    if sensory_input:
        add_q("w_in", _t(model.w_in))
        add("b_in", _t(model.b_in).astype(np.float32), "f32")
    else:                       # vision-only model: no dense projection (input_idx stays, for the visualiser)
        add_q("w_in", np.zeros((0, int(cfg.input_dim)), np.float32))
        add("b_in", np.zeros(0, np.float32), "f32")
    add_q("policy_w", _t(model.policy_head.weight))
    add("policy_b", _t(model.policy_head.bias).astype(np.float32), "f32")
    if isinstance(model.value_head, torch.nn.Sequential):
        l1, l2 = model.value_head[0], model.value_head[2]
        add_q("value_w", _t(l1.weight))
        add("value_b", _t(l1.bias).astype(np.float32), "f32")
        add_q("value_w2", _t(l2.weight))
        add("value_b2", _t(l2.bias).astype(np.float32), "f32")
        value_head = "mlp"
    else:
        add_q("value_w", _t(model.value_head.weight))
        add("value_b", _t(model.value_head.bias).astype(np.float32), "f32")
        value_head = "linear"
    add("positions", normalise_positions(graph.position).astype(np.float16)[node_perm], "f16")
    add("super_class", super_class_codes(graph.super_class)[node_perm], "u8")
    add("node_perm", node_perm.astype(np.int32), "i32")
    # ---- optional features (always present; zero-length when off) ----
    num_planes = int(cfg.input_dim // 64)
    vision = bool(getattr(model, "vision", False))
    n_ret = int(model.n_ret) if vision else 0
    if vision:
        retina_idx = np.asarray(graph.retina_idx).astype(np.int64)
        retina_uv = np.asarray(getattr(graph, "retina_uv", np.zeros((n_ret, 2), np.float32)), np.float32)
        retina_eye = np.asarray(getattr(graph, "retina_eye", np.zeros(n_ret, np.int8)))
        retina_type = np.asarray(getattr(graph, "retina_type", np.array([""] * n_ret))).astype(str)
        add("retina_idx", node_inv[retina_idx].astype(np.int32), "i32")
        add("retina_square", np.asarray(graph.retina_square).astype(np.uint8), "u8")
        add("retina_uv", retina_uv.reshape(n_ret, 2).astype(np.float16), "f16")
        add("retina_eye", retina_eye.reshape(n_ret).astype(np.uint8), "u8")
        add_q("w_ret", _t(model.w_ret).reshape(n_ret, num_planes))
        add("b_ret", _t(model.b_ret).astype(np.float32), "f32")
    else:
        retina_type = np.zeros(0, dtype=str)
        add("retina_idx", np.zeros(0, np.int32), "i32")
        add("retina_square", np.zeros(0, np.uint8), "u8")
        add("retina_uv", np.zeros((0, 2), np.float16), "f16")
        add("retina_eye", np.zeros(0, np.uint8), "u8")
        add_q("w_ret", np.zeros((0, num_planes), np.float32))
        add("b_ret", np.zeros(0, np.float32), "f32")
    add("mod_indptr", mod_indptr, "i32")
    add("mod_indices", mod_indices, "i32")
    add_q("w_mod", w_mod)
    central_dim = int(getattr(model, "central_dim", 0))
    n_central = int(model.n_central) if central_dim > 0 else 0
    if central_dim > 0:
        add("central_idx", node_inv[_t(model.central_idx).astype(np.int64)].astype(np.int32), "i32")
        add_q("central_w", _t(model.central_proj.weight).reshape(central_dim, n_central))
        add("central_b", _t(model.central_proj.bias).astype(np.float32), "f32")
    else:
        add("central_idx", np.zeros(0, np.int32), "i32")
        add_q("central_w", np.zeros((0, 0), np.float32))
        add("central_b", np.zeros(0, np.float32), "f32")
    type_legend = sorted(set(retina_type.tolist()))
    add("retina_type", np.array([type_legend.index(t) for t in retina_type.tolist()], dtype=np.uint8), "u8")

    # ---- lay out the blob ----
    entries: list[dict[str, Any]] = []
    chunks: list[bytes] = []
    offset = 0
    for name, arr, dtype, scale in arrays:
        data = arr.astype(arr.dtype.newbyteorder("<"), copy=False).tobytes(order="C")
        entry: dict[str, Any] = {"name": name, "dtype": dtype, "shape": list(arr.shape),
                                 "offset": offset, "length_bytes": len(data)}
        if scale is not None:
            entry["scale"] = scale
        entries.append(entry)
        chunks.append(data)
        pad = _pad(len(data))
        if pad:
            chunks.append(b"\0" * pad)
        offset += len(data) + pad
    blob = b"".join(chunks)
    gz = gzip.compress(blob, compresslevel=6, mtime=0)

    header: dict[str, Any] = {
        "format": "flyb",
        "version": FORMAT_VERSION,
        "byte_order": "little",
        "quant": quant,
        "steps": int(cfg.steps),
        "activation": cfg.activation,
        "gelu_approximate": "tanh",
        "activation_sat": float(cfg.sat),
        "neuron_order": neuron_order,
        "value_head": value_head,
        # non-linearity between value_w and value_w2 (MLP head); the engines must read it from here
        "value_activation": cfg.activation,
        "value_hidden": int(cfg.value_hidden),
        "dale": bool(cfg.dale),
        "n": int(graph.n),
        # nnz = entries of the main CSR (what the recurrent loop iterates): all synapses, or only the
        # ionotropic ones under neuromod (the DA/SER/OCT synapses are then in mod_* / nnz_mod)
        "nnz": int(csr_indices.shape[0]),
        "nnz_mod": int(mod_indices.shape[0]),
        "nnz_total": int(graph.nnz),
        "total_synapses": int(graph.meta.get("total_syn_count") or float(graph.syn_count.sum())),
        "n_in": int(input_idx.size),
        "n_out": int(output_idx.size),
        "num_moves": int(cfg.num_moves),
        "input_dim": int(cfg.input_dim),
        "num_planes": num_planes,
        "vision": vision,
        "sensory_input": sensory_input,
        "readout_steps": [int(t) for t in getattr(model, "readout_steps", (cfg.steps,))],
        "neuromod": neuromod,
        "central_dim": central_dim,
        "n_ret": n_ret,
        "n_central": n_central,
        "retina_type_legend": type_legend,
        "retina": dict(graph.meta.get("retina", {})) if isinstance(graph.meta.get("retina", {}), dict) else {},
        "run_name": extra_meta.pop("run_name", ""),
        "train_steps": int(extra_meta.pop("train_steps", 0)),
        "exported_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "elo_estimates": extra_meta.pop("elo_estimates", {}),
        "super_class_legend": list(SUPER_CLASS_LEGEND),
        "graph_meta": dict(graph.meta),
        "total_bytes": len(blob),
        # the web loader verifies the downloaded blob against these (stale-cache / out-of-sync guard)
        "blob_sha256": hashlib.sha256(blob).hexdigest(),
        "gzip_bytes": len(gz),
        "arrays": entries,
    }
    header.update(extra_meta)  # any remaining caller metadata (must be JSON-serialisable)
    features = [name for name, on in (("vision", vision), ("neuromod", neuromod), ("central_dim", central_dim > 0),
                                      ("readout_steps", len(header["readout_steps"]) > 1),
                                      ("sensory_input=false", not sensory_input)) if on]
    if features:
        warnings.warn("the exported brain relies on the SPEC §8 optional features " + ", ".join(features)
                      + ": the browser engine must implement them (numpy_forward is the reference)")

    (out_dir / "brain.flyb").write_bytes(blob)
    (out_dir / "brain.flyb.gz").write_bytes(gz)
    (out_dir / "brain.json").write_text(json.dumps(header, indent=1, default=_json_default))
    return json.loads((out_dir / "brain.json").read_text())


def _json_default(o: Any) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def read_flyb(out_dir: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Read ``brain.json`` + ``brain.flyb`` back → ``(arrays, header)``; arrays keep their stored dtypes."""
    out_dir = Path(out_dir)
    header = json.loads((out_dir / "brain.json").read_text())
    blob = (out_dir / "brain.flyb").read_bytes()
    if len(blob) != header["total_bytes"]:
        raise ValueError(f"brain.flyb has {len(blob)} bytes, header says {header['total_bytes']}")
    arrays: dict[str, np.ndarray] = {}
    for e in header["arrays"]:
        dt = np.dtype(_NP_DTYPE[e["dtype"]]).newbyteorder("<")
        arr = np.frombuffer(blob, dtype=dt, count=int(np.prod(e["shape"])) if e["shape"] else 1, offset=e["offset"])
        arrays[e["name"]] = arr.reshape(e["shape"]).astype(dt.newbyteorder("="))
    return arrays, header


def _np_act(name: str, sat: float = 10.0):
    if name == "relu":
        return lambda z: np.maximum(z, 0.0)
    if name == "satrelu":
        return lambda z: sat * np.tanh(np.maximum(z, 0.0) / sat)
    if name == "tanh":
        return np.tanh
    if name == "gelu":
        c = math.sqrt(2.0 / math.pi)
        return lambda z: 0.5 * z * (1.0 + np.tanh(c * (z + 0.044715 * z ** 3)))
    raise ValueError(name)


def numpy_forward(arrays: dict[str, np.ndarray], header: dict[str, Any], x1280: np.ndarray
                  ) -> tuple[np.ndarray, float, np.ndarray]:
    """Pure-numpy forward pass mirroring ``web/engine/flybrain.js`` → ``(policy (num_moves,), value, h (n,))``.

    Single position. Uses the stored (quantised) weights, float32 state, SpMV via CSR gather + bincount.
    Implements every optional feature the header may declare (all default to off, so old blobs still
    run): ``vision`` (retina injection ``w_ret[k] . planes[:, retina_square[k]] + b_ret[k]`` into
    ``retina_idx``), ``sensory_input`` (the dense ``w_in`` path; default on), ``readout_steps``
    (concatenated ``output_idx`` activity at those 1-based steps; default final step), ``neuromod``
    (``pre = ion * (1 + tanh(mod))`` with the ``mod_*`` CSR) and ``central_dim`` (``central_w @
    h_T[central_idx] + central_b`` appended to the head input).
    The value MLP's hidden non-linearity is ``header['value_activation']`` (falls back to ``activation``).
    """
    entries = {e["name"]: e for e in header["arrays"]}
    deq = lambda name: _dequantise(arrays[name], entries[name])
    n, steps = header["n"], header["steps"]
    num_planes = int(header.get("num_planes", 20))
    indptr = arrays["csr_indptr"].astype(np.int64)
    col = arrays["csr_indices"].astype(np.int64)
    rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr))
    w = deq("w")
    bias, alpha = deq("bias"), deq("alpha")
    input_idx = arrays["input_idx"].astype(np.int64)
    output_idx = arrays["output_idx"].astype(np.int64)
    act = _np_act(header["activation"], float(header.get("activation_sat", 10.0)))
    sensory = bool(header.get("sensory_input", True))
    vision = bool(header.get("vision", False)) and arrays.get("retina_idx", np.zeros(0)).size > 0
    neuromod = bool(header.get("neuromod", False)) and arrays.get("mod_indices", np.zeros(0)).size > 0
    central_dim = int(header.get("central_dim", 0))
    readout = [int(t) for t in (header.get("readout_steps") or [steps])]

    x = np.asarray(x1280, dtype=np.float32).reshape(-1)
    h_in = deq("w_in") @ x + deq("b_in") if sensory else None                # (n_in,)
    if vision:
        retina_idx = arrays["retina_idx"].astype(np.int64)
        square = arrays["retina_square"].astype(np.int64)
        planes = x.reshape(num_planes, 64)
        r_in = (deq("w_ret") * planes[:, square].T).sum(1) + deq("b_ret")   # (n_ret,)
    if neuromod:
        mod_indptr = arrays["mod_indptr"].astype(np.int64)
        mod_col = arrays["mod_indices"].astype(np.int64)
        mod_rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(mod_indptr))
        w_mod = deq("w_mod")
    h = np.zeros(n, dtype=np.float32)
    feats = []
    for t in range(steps):
        if t > 0:
            pre = np.bincount(rows, weights=w * h[col], minlength=n).astype(np.float32)
            if neuromod:
                mod = np.bincount(mod_rows, weights=w_mod * h[mod_col], minlength=n).astype(np.float32)
                pre = pre * (1.0 + np.tanh(mod))
        else:
            pre = np.zeros(n, np.float32)
        pre = pre + bias
        if sensory:
            pre[input_idx] += h_in
        if vision:
            np.add.at(pre, retina_idx, r_in)   # a photoreceptor may also be a sensory input neuron
        h = ((1.0 - alpha) * h + alpha * act(pre)).astype(np.float32)
        if t + 1 in readout:
            feats.append(h[output_idx])
    if central_dim > 0:
        feats.append(deq("central_w") @ h[arrays["central_idx"].astype(np.int64)] + deq("central_b"))
    out = np.concatenate(feats).astype(np.float32)
    policy = deq("policy_w") @ out + deq("policy_b")
    v = deq("value_w") @ out + deq("value_b")
    if header.get("value_head") == "mlp":
        v = _np_act(header.get("value_activation", header["activation"]), float(header.get("activation_sat", 10.0)))(v)
        v = deq("value_w2") @ v + deq("value_b2")
    value = float(np.tanh(v[0]))
    return policy.astype(np.float32), value, h


def flyb_size_report(header: dict[str, Any]) -> str:
    """Human-readable per-array byte sizes (for the CLI)."""
    lines = [f"{e['name']:<12} {e['dtype']:<4} {e['shape']!s:<18} {e['length_bytes'] / 1e6:8.2f} MB"
             for e in header["arrays"]]
    lines.append(f"{'total':<12} {'':<4} {'':<18} {header['total_bytes'] / 1e6:8.2f} MB")
    return "\n".join(lines)

