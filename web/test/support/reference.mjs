// Naive float64 reference of SPEC §4 with every §8 optional feature (a port of
// flychess.export.web.numpy_forward), for the node tests of both engines. Returns the per-step
// states too, so trace mode can be checked step by step.
import { modelFeatures } from '../../engine/loader.js';

export function rng(seed) {
  let s = seed >>> 0;
  return () => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; };
}

/** A random 0/1 board with the constant plane set (like a real encoding). */
export function randomInput(seed) {
  const r = rng(seed);
  const x = new Float32Array(1280);
  for (let i = 0; i < 1280; i++) x[i] = r() < 0.15 ? 1 : 0;
  for (let i = 17 * 64; i < 18 * 64; i++) x[i] = 1;
  return x;
}

export function erfRef(x) {
  const ax = Math.abs(x);
  if (ax < 3) {
    let sum = 0, term = ax;
    for (let k = 0; k < 60; k++) { sum += term / (2 * k + 1); term *= -ax * ax / (k + 1); }
    return Math.sign(x) * (2 / Math.sqrt(Math.PI)) * sum;
  }
  return Math.sign(x) * (1 - Math.exp(-ax * ax) / (ax * Math.sqrt(Math.PI)) * (1 - 1 / (2 * ax * ax)));
}

export function actRef(header, name) {
  const sat = Number(header.activation_sat ?? 10);
  const geluTanh = /^tanh$/i.test(String(header.gelu_approximate || ''));
  switch (name) {
    case 'tanh': return Math.tanh;
    case 'gelu': return geluTanh
      ? (v) => 0.5 * v * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (v + 0.044715 * v ** 3)))
      : (v) => 0.5 * v * (1 + erfRef(v / Math.SQRT2));
    case 'satrelu': return (v) => (v > 0 ? sat * Math.tanh(v / sat) : 0);
    default: return (v) => Math.max(0, v);
  }
}

/**
 * @returns {{policy: Float64Array, value: number, activity: Float64Array, states: Float64Array[], retinaDrive: Float64Array|null, feat: Float64Array}}
 */
export function reference(header, a, x) {
  const F = modelFeatures(header, a);
  const n = header.n, act = actRef(header, header.activation), vact = actRef(header, header.value_activation || header.activation);
  const D = x.length, P = F.numPlanes;
  // injections (constant over the timesteps)
  const inj = new Float64Array(n);
  if (F.sensoryInput) {
    for (let k = 0; k < a.input_idx.length; k++) {
      let s = a.b_in[k];
      for (let j = 0; j < D; j++) s += a.w_in[k * D + j] * x[j];
      inj[a.input_idx[k]] += s;
    }
  }
  let retinaDrive = null;
  if (F.vision) {
    retinaDrive = new Float64Array(F.nRet);
    for (let k = 0; k < F.nRet; k++) {
      let s = a.b_ret[k];
      for (let p = 0; p < P; p++) s += a.w_ret[k * P + p] * x[p * 64 + a.retina_square[k]];
      retinaDrive[k] = s;
      inj[a.retina_idx[k]] += s;
    }
  }
  let h = new Float64Array(n);
  const states = [], feats = [];
  for (let t = 0; t < header.steps; t++) {
    const hn = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let ion = 0;
      for (let e = a.csr_indptr[i]; e < a.csr_indptr[i + 1]; e++) ion += a.w[e] * h[a.csr_indices[e]];
      if (F.neuromod) {
        let mod = 0;
        for (let e = a.mod_indptr[i]; e < a.mod_indptr[i + 1]; e++) mod += a.w_mod[e] * h[a.mod_indices[e]];
        ion *= 1 + Math.tanh(mod);
      }
      const pre = ion + a.bias[i] + inj[i];
      hn[i] = (1 - a.alpha[i]) * h[i] + a.alpha[i] * act(pre);
    }
    h = hn;
    states.push(h);
    if (F.readoutSteps.includes(t + 1)) feats.push(Array.from(a.output_idx, (i) => h[i]));
  }
  if (F.centralDim > 0) {
    const c = [];
    for (let k = 0; k < F.centralDim; k++) {
      let s = a.central_b[k];
      for (let j = 0; j < F.nCentral; j++) s += a.central_w[k * F.nCentral + j] * h[a.central_idx[j]];
      c.push(s);
    }
    feats.push(c);
  }
  const feat = Float64Array.from(feats.flat());
  const K = feat.length;
  const policy = new Float64Array(header.num_moves);
  for (let m = 0; m < header.num_moves; m++) {
    let s = a.policy_b[m];
    for (let j = 0; j < K; j++) s += a.policy_w[m * K + j] * feat[j];
    policy[m] = s;
  }
  let v;
  if (a.value_w2) {
    const H = a.value_w.length / K;
    v = a.value_b2[0];
    for (let k = 0; k < H; k++) {
      let s = a.value_b[k];
      for (let j = 0; j < K; j++) s += a.value_w[k * K + j] * feat[j];
      v += a.value_w2[k] * vact(s);
    }
  } else {
    v = a.value_b[0];
    for (let j = 0; j < K; j++) v += a.value_w[j] * feat[j];
  }
  return { policy, value: Math.tanh(v), activity: h, states, retinaDrive, feat };
}

export const maxAbsDiff = (a, b) => { let m = 0; for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i])); return m; };

/** Feature combinations every engine test runs through (label → synthModel options). */
export const FEATURE_CASES = [
  ['legacy blob (no feature arrays)', { legacy: true }],
  ['feature arrays present, all off', {}],
  ['vision', { nRet: 12 }],
  ['vision only (sensory_input=false)', { nRet: 12, sensoryInput: false }],
  ['neuromod', { nnzMod: 60 }],
  ['readout_steps [1,3]', { readoutSteps: [1, 3] }],
  ['central_dim 5', { centralDim: 5 }],
  ['everything (satrelu)', { nRet: 12, nnzMod: 60, readoutSteps: [2, 3], centralDim: 5, activation: 'satrelu' }],
  ['everything, 4 steps, readout [1,2,4], tanh', { nRet: 20, nnzMod: 90, readoutSteps: [1, 2, 4], centralDim: 3, steps: 4, activation: 'tanh', n: 96, nnz: 700 }],
];
