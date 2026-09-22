// Synthetic `fly export-web` output for browser-level tests (same layout as SPEC §8, tiny sizes).
// Not a real fly: random CSR wiring and heads, just enough for worker.js to load and pick moves.
// The optional features (retina / neuromod / readout_steps / central summary / sensory_input=false)
// are written exactly as export_web does — always present after node_perm, zero-length when off —
// unless `legacy` is set, which reproduces a pre-feature blob (no feature arrays or flags at all).
import { createHash } from 'node:crypto';
import { floatToHalf } from '../../engine/loader.js';

function rng(seed) { let x = seed >>> 0 || 1; return () => { x ^= x << 13; x >>>= 0; x ^= x >>> 17; x ^= x << 5; x >>>= 0; return x / 4294967296; }; }

/**
 * @param {object} [o]
 * @param {number} [o.nRet=0]        photoreceptors (vision); each looks at a square, 2 eyes, 3 types
 * @param {number} [o.nnzMod=0]      modulatory synapses (neuromod) in a separate CSR over random rows
 * @param {number[]} [o.readoutSteps]  1-based readout steps (default: final step only)
 * @param {number} [o.centralDim=0]  central summary width (nCentral neurons = every 3rd neuron)
 * @param {boolean} [o.sensoryInput=true]  false: w_in / b_in with zero rows
 * @param {boolean} [o.legacy=false] omit the feature arrays and flags entirely (old export)
 * @returns {{ header: object, blob: Buffer }} brain.json + brain.flyb bytes
 */
export function synthModel({ n = 64, nnz = 400, nIn = 6, nOut = 7, numMoves = 4168, inputDim = 1280, valueHidden = 4, steps = 3, seed = 1,
  nRet = 0, nnzMod = 0, readoutSteps = null, centralDim = 0, sensoryInput = true, legacy = false, activation = 'relu' } = {}) {
  const r = rng(seed);
  const rows = Array.from({ length: n }, () => new Set());
  let placed = 0;
  while (placed < nnz) {
    const post = Math.floor(r() * n), pre = Math.floor(r() * n);
    if (post === pre || rows[post].has(pre)) continue;
    rows[post].add(pre); placed++;
  }
  const csr = (rowSets, count, scale, positive = false) => {
    const indptr = new Int32Array(n + 1), indices = new Int32Array(count), w = new Float32Array(count);
    let e = 0;
    for (let i = 0; i < n; i++) {
      for (const c of [...rowSets[i]].sort((a, b) => a - b)) { indices[e] = c; w[e] = positive ? r() * scale : (r() - 0.4) * scale; e++; }
      indptr[i + 1] = e;
    }
    return { indptr, indices, w };
  };
  const main = csr(rows, nnz, 0.6);
  const perm = Array.from({ length: n }, (_, i) => i).sort(() => r() - 0.5);
  const inputIdx = Int32Array.from(perm.slice(0, nIn)).sort(), outputIdx = Int32Array.from(perm.slice(nIn, nIn + nOut)).sort();
  // photoreceptors: disjoint from the inputs (may overlap the outputs, like the real graph's R7/R8)
  const retinaIdx = Int32Array.from(perm.slice(nIn + nOut, nIn + nOut + nRet)).sort();
  if (retinaIdx.length !== nRet) throw new Error('n too small for nRet');
  const numPlanes = inputDim / 64;
  const f32 = (len, scale = 0.5) => Float32Array.from({ length: len }, () => (r() - 0.5) * scale);
  const readout = readoutSteps || [steps];
  const nCentral = centralDim > 0 ? Math.floor(n / 3) : 0;
  const headIn = nOut * readout.length + centralDim;
  const arrays = [
    ['csr_indptr', 'i32', [n + 1], main.indptr], ['csr_indices', 'i32', [nnz], main.indices], ['w', 'f16', [nnz], main.w],
    ['bias', 'f32', [n], f32(n, 0.2)], ['alpha', 'f32', [n], Float32Array.from({ length: n }, () => 0.2 + 0.7 * r())],
    ['input_idx', 'i32', [nIn], inputIdx], ['output_idx', 'i32', [nOut], outputIdx],
    ['w_in', 'f16', [sensoryInput ? nIn : 0, inputDim], f32(sensoryInput ? nIn * inputDim : 0, 0.1)], ['b_in', 'f32', [sensoryInput ? nIn : 0], f32(sensoryInput ? nIn : 0)],
    ['policy_w', 'f16', [numMoves, headIn], f32(numMoves * headIn, 1)], ['policy_b', 'f32', [numMoves], f32(numMoves, 0.1)],
    ['value_w', 'f16', [valueHidden, headIn], f32(valueHidden * headIn, 1)], ['value_b', 'f32', [valueHidden], f32(valueHidden)],
    ['value_w2', 'f16', [1, valueHidden], f32(valueHidden, 1)], ['value_b2', 'f32', [1], f32(1)],
    ['positions', 'f16', [n, 3], Float32Array.from({ length: n * 3 }, () => r())],
    ['super_class', 'u8', [n], Uint8Array.from({ length: n }, () => Math.floor(r() * 4))],
  ];
  const typeLegend = ['R1-6', 'R7', 'R8'];
  let modSets = null;
  if (!legacy) {
    arrays.push(['node_perm', 'i32', [n], Int32Array.from({ length: n }, (_, i) => i)]);
    const eye = Uint8Array.from({ length: nRet }, (_, k) => (k % 2));
    const square = Uint8Array.from({ length: nRet }, () => Math.floor(r() * 64));
    const uv = new Float32Array(nRet * 2);
    for (let k = 0; k < nRet; k++) { uv[2 * k] = eye[k] ? 0.5 + 0.5 * r() : 0.499 * r(); uv[2 * k + 1] = r(); }
    arrays.push(['retina_idx', 'i32', [nRet], retinaIdx], ['retina_square', 'u8', [nRet], square], ['retina_uv', 'f16', [nRet, 2], uv],
      ['retina_eye', 'u8', [nRet], eye], ['w_ret', 'f16', [nRet, numPlanes], f32(nRet * numPlanes, 0.6)], ['b_ret', 'f32', [nRet], f32(nRet, 0.2)]);
    modSets = Array.from({ length: n }, () => new Set());
    let placedMod = 0;
    while (placedMod < nnzMod) {
      const post = Math.floor(r() * n), pre = Math.floor(r() * n);
      if (post === pre || modSets[post].has(pre) || rows[post].has(pre)) continue;
      modSets[post].add(pre); placedMod++;
    }
    const mod = csr(modSets, nnzMod, 0.5, true);
    arrays.push(['mod_indptr', 'i32', [nnzMod ? n + 1 : 0], nnzMod ? mod.indptr : new Int32Array(0)], ['mod_indices', 'i32', [nnzMod], mod.indices], ['w_mod', 'f16', [nnzMod], mod.w]);
    const centralIdx = Int32Array.from({ length: nCentral }, (_, j) => 3 * j);
    arrays.push(['central_idx', 'i32', [nCentral], centralIdx], ['central_w', 'f16', [centralDim, nCentral], f32(centralDim * nCentral, 0.4)], ['central_b', 'f32', [centralDim], f32(centralDim, 0.2)]);
    arrays.push(['retina_type', 'u8', [nRet], Uint8Array.from({ length: nRet }, () => Math.floor(r() * 3))]);
  }
  const specs = [], parts = [];
  let offset = 0;
  for (const [name, dtype, shape, data] of arrays) {
    let bytes;
    if (dtype === 'i32') bytes = new Uint8Array(Int32Array.from(data).buffer);
    else if (dtype === 'f32') bytes = new Uint8Array(Float32Array.from(data).buffer);
    else if (dtype === 'u8') bytes = Uint8Array.from(data);
    else { const u = new Uint16Array(data.length); for (let i = 0; i < data.length; i++) u[i] = floatToHalf(data[i]); bytes = new Uint8Array(u.buffer); }
    specs.push({ name, dtype, shape, offset, length_bytes: bytes.byteLength });
    parts.push([offset, bytes]);
    offset += Math.ceil(bytes.byteLength / 8) * 8;
  }
  const blob = Buffer.alloc(offset);
  for (const [off, bytes] of parts) blob.set(bytes, off);
  const header = {
    format: 'flyb', version: 1, run_name: 'synth', exported_at: '2026-01-01T00:00:00+00:00',
    n, nnz, steps, activation, input_dim: inputDim, num_moves: numMoves, num_planes: numPlanes, n_in: nIn, n_out: nOut,
    total_bytes: blob.byteLength, blob_sha256: createHash('sha256').update(blob).digest('hex'),
    super_class_legend: ['optic', 'central', 'sensory', 'descending'],
    arrays: specs,
  };
  if (!legacy) {
    Object.assign(header, {
      vision: nRet > 0, sensory_input: sensoryInput, readout_steps: readout, neuromod: nnzMod > 0, central_dim: centralDim,
      n_ret: nRet, n_central: nCentral, nnz_mod: nnzMod, nnz_total: nnz + nnzMod, retina_type_legend: nRet ? typeLegend : [],
    });
  }
  return { header, blob };
}
