// node --test web/test/flybrain-gpu.test.mjs
// There is no WebGPU under node, so this checks the *math* of the WGSL kernels through their
// line-by-line f32 JS mirrors (emulateInject / emulateStep / emulateMatvec / emulateForward in
// flybrain-gpu.js) against a float64 reference of SPEC §4 and against FlyBrain, for every
// activation, on a tiny synthetic .flyb blob — including the SPEC §8 feature kernels (retina drive,
// base with both injections, gated step rows for neuromod, readout / trace gather, central matvec)
// and the end-to-end mirror on every feature combination. The shader text itself is exercised in
// the browser by web/test/browser/gpu-parity.mjs (real model, real device). Also covers the
// promise-aware MCTS.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import path from 'node:path';

import { parseArrays } from '../engine/loader.js';
import { FlyBrain } from '../engine/flybrain.js';
import {
  FlyBrainGPU, activationKind, activationWGSL, activationF32, inverseInputMap, modelShape, bucketRows,
  emulateInject, emulateStep, emulateMatvec, emulateForward, emulateRetina, emulateGather, valueFromHidden,
} from '../engine/flybrain-gpu.js';
import { runMCTS, runMCTSAsync, createSearch, createSearchAsync } from '../engine/mcts.js';
import { synthModel } from './support/synthmodel.mjs';
import { reference as referenceAll, FEATURE_CASES } from './support/reference.mjs';

/** Node's ESM loader needs a file:// URL, not a Windows path. */
const fil = (p) => pathToFileURL(p).href;
const here = path.dirname(fileURLToPath(import.meta.url));
const chessPath = path.join(here, '..', 'vendor', 'chess.js');
const encPath = path.join(here, '..', 'engine', 'encoding.js');

function rng(seed) {
  let s = seed >>> 0;
  return () => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; };
}

/** Synthetic blob → {header, arrays} with the requested activation header fields. */
function model(headerExtra = {}, blobOpts = {}) {
  const { header, blob } = synthModel({ seed: 11, steps: 4, ...blobOpts });
  const buffer = new ArrayBuffer(blob.byteLength);
  new Uint8Array(buffer).set(blob);
  Object.assign(header, headerExtra);
  return { header, arrays: parseArrays(header, buffer) };
}

function erfRef(x) {
  const ax = Math.abs(x);
  if (ax < 3) {
    let sum = 0, term = ax;
    for (let k = 0; k < 60; k++) { sum += term / (2 * k + 1); term *= -ax * ax / (k + 1); }
    return Math.sign(x) * (2 / Math.sqrt(Math.PI)) * sum;
  }
  return Math.sign(x) * (1 - Math.exp(-ax * ax) / (ax * Math.sqrt(Math.PI)) * (1 - 1 / (2 * ax * ax)));
}

function actRef(header, name) {
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

/** Naive float64 reference of SPEC §4 (same as flybrain.test.mjs, plus value_activation). */
function reference(header, a, x) {
  const n = header.n, act = actRef(header, header.activation), vact = actRef(header, header.value_activation || header.activation);
  let h = new Float64Array(n);
  const D = x.length;
  for (let t = 0; t < header.steps; t++) {
    const pre = new Float64Array(n);
    for (let post = 0; post < n; post++) {
      let s = a.bias[post];
      for (let e = a.csr_indptr[post]; e < a.csr_indptr[post + 1]; e++) s += a.w[e] * h[a.csr_indices[e]];
      pre[post] = s;
    }
    for (let k = 0; k < a.input_idx.length; k++) {
      let s = a.b_in[k];
      for (let j = 0; j < D; j++) s += a.w_in[k * D + j] * x[j];
      pre[a.input_idx[k]] += s;
    }
    const hn = new Float64Array(n);
    for (let i = 0; i < n; i++) hn[i] = (1 - a.alpha[i]) * h[i] + a.alpha[i] * act(pre[i]);
    h = hn;
  }
  const out = Array.from(a.output_idx, (i) => h[i]);
  const nOut = out.length;
  const policy = new Float64Array(header.num_moves);
  for (let m = 0; m < header.num_moves; m++) {
    let s = a.policy_b[m];
    for (let j = 0; j < nOut; j++) s += a.policy_w[m * nOut + j] * out[j];
    policy[m] = s;
  }
  let v;
  if (a.value_w2) {
    const H = a.value_w.length / nOut;
    v = a.value_b2[0];
    for (let k = 0; k < H; k++) {
      let s = a.value_b[k];
      for (let j = 0; j < nOut; j++) s += a.value_w[k * nOut + j] * out[j];
      v += a.value_w2[k] * vact(s);
    }
  } else {
    v = a.value_b[0];
    for (let j = 0; j < nOut; j++) v += a.value_w[j] * out[j];
  }
  return { policy, value: Math.tanh(v), activity: h };
}

function randomInput(seed) {
  const r = rng(seed);
  const x = new Float32Array(1280);
  for (let i = 0; i < 1280; i++) x[i] = r() < 0.15 ? 1 : 0;
  for (let i = 17 * 64; i < 18 * 64; i++) x[i] = 1;
  return x;
}

const maxAbsDiff = (a, b) => { let m = 0; for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i])); return m; };

// ---------------------------------------------------------------------------------------------
test('activation kinds follow the header like FlyBrain (gelu → tanh approximation, value_activation)', () => {
  assert.equal(activationKind({ activation: 'relu' }), 'relu');
  assert.equal(activationKind({ activation: 'gelu' }), 'gelu');
  assert.equal(activationKind({ activation: 'gelu', gelu_approximate: 'tanh' }), 'gelu_tanh');
  assert.equal(activationKind({ activation: 'satrelu', value_activation: 'relu' }, 'relu'), 'relu');
  assert.equal(activationKind({}), 'relu');
  assert.throws(() => activationKind({ activation: 'swish' }), /unknown activation/);
  const S = modelShape(model({ activation: 'satrelu', value_activation: 'gelu', gelu_approximate: 'tanh', activation_sat: 6 }));
  assert.equal(S.kind, 'satrelu'); assert.equal(S.valueKind, 'gelu_tanh'); assert.equal(S.sat, 6); assert.equal(S.steps, 4);
  assert.equal(S.valueHidden, 4); assert.equal(S.nOut, 7); assert.equal(S.numMoves, 4168);
});

test('WGSL activation bodies are distinct, complete and bake the saturation constant', () => {
  const codes = ['relu', 'tanh', 'gelu', 'gelu_tanh', 'satrelu', 'identity'].map((k) => activationWGSL(k, 10));
  assert.equal(new Set(codes).size, 6);
  for (const c of codes) assert.match(c, /^fn act\(v: f32\) -> f32 \{ return .*; \}\n$/);
  assert.match(activationWGSL('satrelu', 10), /10\.0 \* tanh_\(v \/ 10\.0\)/);
  assert.match(activationWGSL('satrelu', 2.5), /2\.5 \* tanh_/);
  assert.match(activationWGSL('relu', 10, 'vact'), /^fn vact/);
  assert.throws(() => activationWGSL('swish', 10), /unknown activation/);
});

test('f32 activation mirrors match the float64 functions to f32 precision', () => {
  const hdr = { activation_sat: 10 };
  const xs = [-30, -8, -3, -1, -0.5, -1e-3, 0, 1e-3, 0.25, 1, 2.5, 7, 15, 40, 1e3];
  for (const [kind, name] of [['relu', 'relu'], ['tanh', 'tanh'], ['gelu', 'gelu'], ['satrelu', 'satrelu']]) {
    const g = activationF32(kind, 10), r = actRef(hdr, name);
    for (const x of xs) assert.ok(Math.abs(g(x) - r(x)) <= 2e-6 * Math.max(1, Math.abs(r(x))), `${kind}(${x}): ${g(x)} vs ${r(x)}`);
  }
  const gt = activationF32('gelu_tanh'), rt = actRef({ gelu_approximate: 'tanh' }, 'gelu');
  for (const x of xs) assert.ok(Math.abs(gt(x) - rt(x)) <= 2e-6 * Math.max(1, Math.abs(rt(x))), `gelu_tanh(${x})`);
  // saturation: satrelu never exceeds sat, and is 0 for v <= 0
  const s = activationF32('satrelu', 4);
  assert.equal(s(-1), 0); assert.equal(s(0), 0); assert.ok(s(1e6) <= 4 && s(1e6) > 3.999);
});

test('inverseInputMap: bijection on the inputs, NONE elsewhere, rejects duplicates', () => {
  const inv = inverseInputMap(Int32Array.from([5, 2, 9]), 10);
  assert.equal(inv[5], 0); assert.equal(inv[2], 1); assert.equal(inv[9], 2);
  assert.equal(inv[0], 0xffffffff); assert.equal(inv.length, 10);
  assert.throws(() => inverseInputMap(Int32Array.from([1, 1]), 4), /duplicate/);
  assert.throws(() => inverseInputMap(Int32Array.from([4]), 4), /out of range/);
});

test('inject kernel = bias + w_in·x + b_in on the input neurons only', () => {
  const m = model();
  const { n } = modelShape(m);
  const x = randomInput(3);
  const base = emulateInject(m, inverseInputMap(m.arrays.input_idx, n), x, new Float32Array(n));
  const ref = Float64Array.from(m.arrays.bias);
  for (let k = 0; k < m.arrays.input_idx.length; k++) {
    let s = m.arrays.b_in[k];
    for (let j = 0; j < 1280; j++) s += m.arrays.w_in[k * 1280 + j] * x[j];
    ref[m.arrays.input_idx[k]] += s;
  }
  assert.ok(maxAbsDiff(base, ref) < 1e-5);
});

test('step kernel = one leaky recurrent update (f32) for every activation', () => {
  for (const activation of ['relu', 'tanh', 'gelu', 'satrelu']) {
    const m = model({ activation, activation_sat: 3 });
    const S = modelShape(m);
    const act = activationF32(S.kind, S.sat), ref = actRef(m.header, activation);
    const r = rng(5);
    const base = Float32Array.from({ length: S.n }, () => (r() - 0.5) * 4);
    const h = Float32Array.from({ length: S.n }, () => (r() - 0.5) * 6);
    const out = emulateStep(m, S.alpha, act, base, h, new Float32Array(S.n));
    for (let i = 0; i < S.n; i++) {
      let s = base[i];
      for (let e = m.arrays.csr_indptr[i]; e < m.arrays.csr_indptr[i + 1]; e++) s += m.arrays.w[e] * h[m.arrays.csr_indices[e]];
      const want = (1 - S.alpha[i]) * h[i] + S.alpha[i] * ref(s);
      assert.ok(Math.abs(out[i] - want) < 1e-5, `${activation} row ${i}: ${out[i]} vs ${want}`);
    }
  }
});

test('matvec kernel (64-lane reduction) = W·h[output_idx] + b, optionally activated', () => {
  const m = model();
  const S = modelShape(m);
  const r = rng(8);
  const h = Float32Array.from({ length: S.n }, () => (r() - 0.5) * 2);
  const y = emulateMatvec(m.arrays.policy_w, m.arrays.policy_b, S.numMoves, S.nOut, h, m.arrays.output_idx, activationF32('identity'), new Float32Array(S.numMoves));
  for (let mv = 0; mv < S.numMoves; mv += 97) {
    let s = m.arrays.policy_b[mv];
    for (let j = 0; j < S.nOut; j++) s += m.arrays.policy_w[mv * S.nOut + j] * h[m.arrays.output_idx[j]];
    assert.ok(Math.abs(y[mv] - s) < 1e-5);
  }
  // wide rows exercise the strided lanes (nOut > 64): n_out = 150
  const wide = model({}, { nOut: 150, n: 200, nnz: 900 });
  const W = modelShape(wide);
  const hw = Float32Array.from({ length: W.n }, () => (r() - 0.5) * 2);
  const yw = emulateMatvec(wide.arrays.value_w, wide.arrays.value_b, W.valueHidden, W.nOut, hw, wide.arrays.output_idx, activationF32('relu'), new Float32Array(W.valueHidden));
  for (let k = 0; k < W.valueHidden; k++) {
    let s = wide.arrays.value_b[k];
    for (let j = 0; j < W.nOut; j++) s += wide.arrays.value_w[k * W.nOut + j] * hw[wide.arrays.output_idx[j]];
    assert.ok(Math.abs(yw[k] - Math.max(0, s)) < 1e-5);
  }
  // value head tail: MLP dot vs linear
  assert.ok(Math.abs(valueFromHidden({ value_w2: Float32Array.from([0.5, -1]), value_b2: Float32Array.from([0.1]) }, 2, Float32Array.from([2, 1])) - Math.tanh(0.1)) < 1e-6);
  assert.ok(Math.abs(valueFromHidden({}, 0, Float32Array.from([0.3])) - Math.tanh(0.3)) < 1e-6);
});

for (const [activation, extra] of [['relu', {}], ['tanh', {}], ['gelu', {}], ['gelu', { gelu_approximate: 'tanh' }], ['satrelu', { activation_sat: 10 }], ['satrelu', { activation_sat: 10, value_activation: 'satrelu' }]]) {
  const label = `${activation}${extra.gelu_approximate ? '/tanh' : ''}${extra.value_activation ? '+value_activation' : ''}`;
  test(`emulated GPU forward matches the float64 reference and FlyBrain (${label})`, () => {
    const m = model({ activation, ...extra });
    const js = new FlyBrain(m);
    for (let s = 0; s < 3; s++) {
      const x = randomInput(200 + s);
      const got = emulateForward(m, x);
      const ref = reference(m.header, m.arrays, x);
      const cpu = js.forward(x);
      assert.equal(got.policy.length, 4168);
      assert.ok(maxAbsDiff(got.policy, ref.policy) < 1e-3, `policy vs reference ${maxAbsDiff(got.policy, ref.policy)}`);
      assert.ok(maxAbsDiff(got.policy, cpu.policy) < 1e-3, `policy vs FlyBrain ${maxAbsDiff(got.policy, cpu.policy)}`);
      assert.ok(Math.abs(got.value - ref.value) < 1e-4, `value ${got.value} vs ${ref.value}`);
      assert.ok(Math.abs(got.value - cpu.value) < 1e-4);
      assert.ok(maxAbsDiff(got.activity, ref.activity) < 1e-4, `activity ${maxAbsDiff(got.activity, ref.activity)}`);
    }
  });
}

// ---------------------------------------------------------------------------------------------
// SPEC §8 feature kernels
test('bucketRows separates the rows with modulatory inputs', () => {
  const m = model({}, { nnzMod: 60 });
  const plain = bucketRows(m.arrays.csr_indptr, m.header.n);
  assert.equal(plain.length, 3);
  assert.ok(plain.every((b) => b.modRows === 0));
  for (const b of plain) assert.deepEqual(Array.from(b.rows), Array.from(b.rows).sort((a, c) => a - c), 'ascending rows');
  const withMod = bucketRows(m.arrays.csr_indptr, m.header.n, m.arrays.mod_indptr);
  assert.equal(withMod.length, 3);
  const all = withMod.flatMap((b) => Array.from(b.rows)).sort((a, b) => a - b);
  assert.deepEqual(all, Array.from({ length: m.header.n }, (_, i) => i), 'the buckets partition the rows');
  let gated = 0;
  for (const b of withMod) {
    for (const [k, i] of Array.from(b.rows).entries()) assert.equal(m.arrays.mod_indptr[i + 1] > m.arrays.mod_indptr[i], k < b.modRows, `gated rows first (bucket ${b.lanes}, slot ${k})`);
    gated += b.modRows;
    const head = Array.from(b.rows).slice(0, b.modRows), tail = Array.from(b.rows).slice(b.modRows);
    assert.deepEqual(head, [...head].sort((a, c) => a - c)); assert.deepEqual(tail, [...tail].sort((a, c) => a - c));
  }
  assert.ok(gated > 0);
});

test('retina kernel mirror = w_ret·planes[:, square] + b_ret; gather mirror copies with an offset', () => {
  const m = model({}, { nRet: 12 });
  const x = randomInput(4);
  const drive = emulateRetina(m, x, new Float32Array(12));
  const ref = referenceAll(m.header, m.arrays, x).retinaDrive;
  assert.ok(maxAbsDiff(drive, ref) < 1e-5);
  const dst = new Float32Array(10).fill(-1);
  emulateGather(Float32Array.from([5, 6, 7, 8]), Uint32Array.from([3, 0, 2]), 3, 4, dst);
  assert.deepEqual(Array.from(dst), [-1, -1, -1, -1, 8, 5, 7, -1, -1, -1]);
  assert.throws(() => inverseInputMap(Int32Array.from([1, 1]), 4, 'retina_idx'), /retina_idx has duplicate/);
});

test('step kernel mirror gates the rows with modulatory inputs: pre = ion·(1 + tanh(mod)) + base', () => {
  const m = model({}, { nnzMod: 60, activation: 'satrelu' });
  const S = modelShape(m);
  const act = activationF32(S.kind, S.sat);
  const r = rng(5);
  const base = Float32Array.from({ length: S.n }, () => (r() - 0.5) * 4);
  const h = Float32Array.from({ length: S.n }, () => (r() - 0.5) * 6);
  const out = emulateStep(m, S.alpha, act, base, h, new Float32Array(S.n));
  const a = m.arrays;
  let gated = 0;
  for (let i = 0; i < S.n; i++) {
    let ion = 0;
    for (let e = a.csr_indptr[i]; e < a.csr_indptr[i + 1]; e++) ion += a.w[e] * h[a.csr_indices[e]];
    let mod = 0;
    for (let e = a.mod_indptr[i]; e < a.mod_indptr[i + 1]; e++) mod += a.w_mod[e] * h[a.mod_indices[e]];
    if (a.mod_indptr[i + 1] > a.mod_indptr[i]) gated++;
    const pre = ion * (1 + Math.tanh(mod)) + base[i];
    const want = (1 - S.alpha[i]) * h[i] + S.alpha[i] * (pre > 0 ? 10 * Math.tanh(pre / 10) : 0);
    assert.ok(Math.abs(out[i] - want) < 1e-5, `row ${i}: ${out[i]} vs ${want}`);
  }
  assert.ok(gated > 5, `${gated} gated rows`);
});

test('inject kernel with the retina: base = bias + inj[inv_in] + drive[inv_ret]; sensory_input=false skips w_in', () => {
  const m = model({}, { nRet: 12 });
  const S = modelShape(m);
  const x = randomInput(6);
  const invIn = inverseInputMap(m.arrays.input_idx, S.n), invRet = inverseInputMap(m.arrays.retina_idx, S.n, 'retina_idx');
  const drive = emulateRetina(m, x, new Float32Array(12));
  const base = emulateInject(m, invIn, x, new Float32Array(S.n), invRet, drive);
  const ref = referenceAll(m.header, m.arrays, x);
  const want = Float64Array.from(m.arrays.bias);
  for (let k = 0; k < m.arrays.input_idx.length; k++) {
    let s = m.arrays.b_in[k];
    for (let j = 0; j < 1280; j++) s += m.arrays.w_in[k * 1280 + j] * x[j];
    want[m.arrays.input_idx[k]] += s;
  }
  for (let k = 0; k < 12; k++) want[m.arrays.retina_idx[k]] += ref.retinaDrive[k];
  assert.ok(maxAbsDiff(base, want) < 1e-5);
  const vo = model({}, { nRet: 12, sensoryInput: false });
  assert.equal(vo.arrays.w_in.length, 0);
  const none = new Uint32Array(S.n).fill(0xffffffff);
  const base2 = emulateInject(vo, none, x, new Float32Array(S.n), invRet, emulateRetina(vo, x, new Float32Array(12)));
  const want2 = Float64Array.from(vo.arrays.bias);
  const ref2 = referenceAll(vo.header, vo.arrays, x);
  for (let k = 0; k < 12; k++) want2[vo.arrays.retina_idx[k]] += ref2.retinaDrive[k];
  assert.ok(maxAbsDiff(base2, want2) < 1e-5);
});

for (const [label, opts] of FEATURE_CASES) {
  test(`emulated GPU forward with the §8 features matches the float64 reference and FlyBrain (${label})`, () => {
    const m = model({}, opts);
    const js = new FlyBrain(m);
    const S = modelShape(m);
    assert.equal(S.direct, !opts.readoutSteps && !opts.centralDim, 'direct head input only without readouts / central summary');
    for (let s = 0; s < 2; s++) {
      const x = randomInput(400 + s);
      const got = emulateForward(m, x);
      const ref = referenceAll(m.header, m.arrays, x);
      const cpu = js.forward(x);
      assert.ok(maxAbsDiff(got.policy, ref.policy) < 1e-3, `policy vs reference ${maxAbsDiff(got.policy, ref.policy)}`);
      assert.ok(maxAbsDiff(got.policy, cpu.policy) < 1e-3, `policy vs FlyBrain ${maxAbsDiff(got.policy, cpu.policy)}`);
      assert.ok(Math.abs(got.value - ref.value) < 1e-4, `value ${got.value} vs ${ref.value}`);
      assert.ok(maxAbsDiff(got.activity, ref.activity) < 1e-4, `activity ${maxAbsDiff(got.activity, ref.activity)}`);
    }
  });
}

test('linear value head (value_hidden = 0) and scalar alpha', () => {
  const { header, blob } = synthModel({ seed: 4, steps: 3 });
  // drop the MLP tail so the value head is Linear(n_out, 1): value_w [1, nOut]
  header.arrays = header.arrays.filter((a) => a.name !== 'value_w2' && a.name !== 'value_b2');
  const vw = header.arrays.find((a) => a.name === 'value_w');
  vw.shape = [1, 7]; vw.length_bytes = 14;
  header.arrays.find((a) => a.name === 'value_b').shape = [1];
  header.arrays.find((a) => a.name === 'value_b').length_bytes = 4;
  const buffer = new ArrayBuffer(blob.byteLength);
  new Uint8Array(buffer).set(blob);
  const arrays = parseArrays(header, buffer);
  arrays.alpha = new Float32Array([0.5]);
  const m = { header, arrays };
  assert.equal(modelShape(m).valueHidden, 0);
  const x = randomInput(9);
  const got = emulateForward(m, x);
  const js = new FlyBrain(m).forward(x);
  assert.ok(maxAbsDiff(got.policy, js.policy) < 1e-3);
  assert.ok(Math.abs(got.value - js.value) < 1e-4);
});

test('FlyBrainGPU.create rejects cleanly without WebGPU (node)', async () => {
  await assert.rejects(FlyBrainGPU.create(model()), /WebGPU is not available/);
});

// ---------------------------------------------------------------------------------------------
// promise-aware MCTS: an async brain must give the same search as the sync one
const asyncStub = (value, log) => ({
  forward: (x, opts) => { log?.push(opts); return Promise.resolve({ policy: new Float32Array(4168), value, activity: null }); },
});
const syncStub = (value) => ({ forward: () => ({ policy: new Float32Array(4168), value, activity: new Float32Array(1) }) });

test('runMCTSAsync with a promise-returning brain matches the synchronous search', { skip: !existsSync(chessPath) && 'web/vendor/chess.js not present' }, async () => {
  const { Chess } = await import(fil(chessPath));
  const enc = await import(fil(encPath));
  const sync = runMCTS(syncStub(0.1), new Chess(), enc, { sims: 40, rnd: rng(3) });
  const log = [];
  const chess = new Chess();
  const async = await runMCTSAsync(asyncStub(0.1, log), chess, enc, { sims: 40, rnd: rng(3), yieldEvery: 7 });
  assert.deepEqual(async.visits, sync.visits);
  assert.equal(async.move, sync.move);
  assert.equal(async.sims, 40);
  assert.equal(chess.history().length, 0, 'board restored');
  assert.ok(log.length > 0 && log.every((o) => o && o.activity === false), 'the search asks for no hidden state');
  // mate in one with the async brain
  const m = await runMCTSAsync(asyncStub(0), new Chess('6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1'), enc, { sims: 120, rnd: rng(1) });
  assert.equal(m.move, 'a1a8');
  // createSearch refuses an async brain; createSearchAsync takes both
  assert.throws(() => createSearch(asyncStub(0), new Chess(), enc, { sims: 5 }), /synchronous brain/);
  const s1 = await createSearchAsync(syncStub(0), new Chess(), enc, { sims: 5 });
  assert.equal(typeof s1.step(), 'boolean');
  const s2 = await createSearchAsync(asyncStub(0), new Chess(), enc, { sims: 5 });
  const r = s2.step();
  assert.ok(r && typeof r.then === 'function');
  assert.equal(await r, true);
  // early stop still works with an async brain
  let calls = 0;
  const stopped = await runMCTSAsync(asyncStub(0.1), new Chess(), enc, { sims: 200, yieldEvery: 10, shouldStop: () => ++calls >= 2 });
  assert.equal(stopped.stopped, true);
  assert.ok(stopped.sims >= 10 && stopped.sims <= 20);
});
