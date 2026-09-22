// node --test web/test/flybrain.test.mjs
// Builds a tiny random brain as a real .flyb blob (SPEC §8), parses it with the loader, and checks
// FlyBrain.forward against a naive float64 reference. Also checks the softmax helpers and that
// MCTS returns a legal move (and finds a mate in one) with a constant-policy stub brain.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import path from 'node:path';

import { parseArrays, floatToHalf, halfToFloat, decodeF16 } from '../engine/loader.js';
import { FlyBrain, policyForLegal, argmax, sampleIndex, topK, erf } from '../engine/flybrain.js';
import { runMCTS, runMCTSAsync, createSearch } from '../engine/mcts.js';

/** Node's ESM loader needs a file:// URL, not a Windows path. */
const fil = (p) => pathToFileURL(p).href;
const here = path.dirname(fileURLToPath(import.meta.url));
const chessPath = path.join(here, '..', 'vendor', 'chess.js');
const encPath = path.join(here, '..', 'engine', 'encoding.js');

// ---------------------------------------------------------------------------------------------
// deterministic RNG
function rng(seed) {
  let s = seed >>> 0;
  return () => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; };
}

/** Build a random brain and serialise it exactly like `fly export-web` does. */
function buildBlob({ n = 64, nnz = 400, nIn = 6, nOut = 7, numMoves = 4168, inputDim = 1280, valueHidden = 4, steps = 3, activation = 'relu', seed = 1 } = {}) {
  const r = rng(seed);
  // CSR: rows sorted, columns sorted & unique, no self loops
  const rows = [];
  for (let i = 0; i < n; i++) rows.push(new Set());
  let placed = 0;
  while (placed < nnz) {
    const post = Math.floor(r() * n), pre = Math.floor(r() * n);
    if (post === pre || rows[post].has(pre)) continue;
    rows[post].add(pre); placed++;
  }
  const indptr = new Int32Array(n + 1);
  const indices = new Int32Array(nnz);
  const w = new Float32Array(nnz);
  let e = 0;
  for (let i = 0; i < n; i++) {
    const cols = [...rows[i]].sort((a, b) => a - b);
    for (const c of cols) { indices[e] = c; w[e] = (r() - 0.4) * 0.6; e++; }
    indptr[i + 1] = e;
  }
  const perm = Array.from({ length: n }, (_, i) => i).sort(() => r() - 0.5);
  const inputIdx = Int32Array.from(perm.slice(0, nIn)).sort();
  const outputIdx = Int32Array.from(perm.slice(nIn, nIn + nOut)).sort();
  const f32 = (len, scale = 0.5) => Float32Array.from({ length: len }, () => (r() - 0.5) * scale);
  const arrays = [
    ['csr_indptr', 'i32', [n + 1], indptr],
    ['csr_indices', 'i32', [nnz], indices],
    ['w', 'f16', [nnz], w],
    ['bias', 'f32', [n], f32(n, 0.2)],
    ['alpha', 'f32', [n], Float32Array.from({ length: n }, () => 0.2 + 0.7 * r())],
    ['input_idx', 'i32', [nIn], inputIdx],
    ['output_idx', 'i32', [nOut], outputIdx],
    ['w_in', 'f16', [nIn, inputDim], f32(nIn * inputDim, 0.1)],
    ['b_in', 'f32', [nIn], f32(nIn)],
    ['policy_w', 'f16', [numMoves, nOut], f32(numMoves * nOut, 1)],
    ['policy_b', 'f32', [numMoves], f32(numMoves, 0.1)],
    ['value_w', 'f16', [valueHidden || 1, nOut], f32((valueHidden || 1) * nOut, 1)],
    ['value_b', 'f32', [valueHidden || 1], f32(valueHidden || 1)],
  ];
  if (valueHidden) {
    arrays.push(['value_w2', 'f16', [1, valueHidden], f32(valueHidden, 1)]);
    arrays.push(['value_b2', 'f32', [1], f32(1)]);
  }
  arrays.push(['positions', 'f16', [n, 3], Float32Array.from({ length: n * 3 }, () => r())]);
  arrays.push(['super_class', 'u8', [n], Uint8Array.from({ length: n }, () => Math.floor(r() * 4))]);

  // layout
  const specs = [];
  let offset = 0;
  const parts = [];
  for (const [name, dtype, shape, data] of arrays) {
    const count = data.length;
    let bytes;
    if (dtype === 'i32') bytes = new Uint8Array(Int32Array.from(data).buffer);
    else if (dtype === 'f32') bytes = new Uint8Array(Float32Array.from(data).buffer);
    else if (dtype === 'u8') bytes = Uint8Array.from(data);
    else if (dtype === 'f16') { const u = new Uint16Array(count); for (let i = 0; i < count; i++) u[i] = floatToHalf(data[i]); bytes = new Uint8Array(u.buffer); }
    specs.push({ name, dtype, shape, offset, length_bytes: bytes.byteLength });
    parts.push([offset, bytes]);
    offset += Math.ceil(bytes.byteLength / 8) * 8;
  }
  const buffer = new ArrayBuffer(offset);
  const view = new Uint8Array(buffer);
  for (const [off, bytes] of parts) view.set(bytes, off);
  const header = {
    n, nnz, n_in: nIn, n_out: nOut, num_moves: numMoves, num_planes: inputDim / 64, steps, activation,
    value_hidden: valueHidden, run_name: 'test', exported_at: '2026-01-01T00:00:00Z',
    super_class_legend: ['central', 'sensory', 'descending', 'optic'], arrays: specs,
  };
  return { header, buffer };
}

/** erf by series / continued fraction (independent of the engine's polynomial). */
function erfRef(x) {
  const ax = Math.abs(x);
  if (ax < 3) {                      // Maclaurin series, converges fast for |x| < 3
    let sum = 0, term = ax;
    for (let k = 0; k < 60; k++) { sum += term / (2 * k + 1); term *= -ax * ax / (k + 1); }
    return Math.sign(x) * (2 / Math.sqrt(Math.PI)) * sum;
  }
  return Math.sign(x) * (1 - Math.exp(-ax * ax) / (ax * Math.sqrt(Math.PI)) * (1 - 1 / (2 * ax * ax)));
}

/** Naive float64 reference of SPEC §4. */
function reference(header, a, x) {
  const sat = Number(header.activation_sat ?? 10);
  const n = header.n, act = header.activation === 'tanh' ? Math.tanh : header.activation === 'gelu'
    ? (v) => 0.5 * v * (1 + erfRef(v / Math.SQRT2)) : header.activation === 'satrelu'
    ? (v) => (v > 0 ? sat * Math.tanh(v / sat) : 0) : (v) => Math.max(0, v);
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
    const H = header.value_hidden;
    v = a.value_b2[0];
    for (let k = 0; k < H; k++) {
      let s = a.value_b[k];
      for (let j = 0; j < nOut; j++) s += a.value_w[k * nOut + j] * out[j];
      v += a.value_w2[k] * act(s);
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

// ---------------------------------------------------------------------------------------------
test('erf approximation', () => {
  for (const x of [-4, -2, -1, -0.5, -0.1, 0, 0.1, 0.5, 1, 2, 4]) assert.ok(Math.abs(erf(x) - erfRef(x)) < 1e-6, `erf(${x})`);
});

test('f16 round trip', () => {
  for (const v of [0, 1, -1, 0.5, 65504, 1e-5, 3.14159, -0.333, 2 ** -20, 1e-8]) {
    const back = halfToFloat(floatToHalf(v));
    assert.ok(Math.abs(back - v) <= Math.max(Math.abs(v) * 1e-3, 1e-7), `${v} -> ${back}`);
  }
  assert.equal(floatToHalf(1), 0x3c00);
  assert.equal(halfToFloat(0x3c00), 1);
  assert.equal(halfToFloat(0xc000), -2);
  const u = new Uint16Array([0x3c00, 0x4000, 0x3800, 0x0000]);
  assert.deepEqual(Array.from(decodeF16(u.buffer, 0, 4)), [1, 2, 0.5, 0]);
});

test('loader parses the .flyb layout and dequantises i8', () => {
  const { header, buffer } = buildBlob({ n: 20, nnz: 40 });
  const a = parseArrays(header, buffer);
  assert.equal(a.csr_indptr.length, 21);
  assert.equal(a.csr_indptr[20], 40);
  assert.ok(a.w instanceof Float32Array && a.w.length === 40);
  assert.equal(a.w_in.length, 6 * 1280);
  assert.equal(a.super_class.length, 20);
  // i8 with scalar scale
  const buf = new ArrayBuffer(8);
  new Int8Array(buf).set([127, -128, 0, 1]);
  const q = parseArrays({ arrays: [{ name: 'q', dtype: 'i8', shape: [4], offset: 0, length_bytes: 4, scale: 0.5 }] }, buf);
  assert.deepEqual(Array.from(q.q), [63.5, -64, 0, 0.5]);
  // i8 with per-row scale
  const q2 = parseArrays({ arrays: [{ name: 'q', dtype: 'i8', shape: [2, 2], offset: 0, length_bytes: 4, scale: [1, 2] }] }, buf);
  assert.deepEqual(Array.from(q2.q), [127, -128, 0, 2]);
});

for (const activation of ['relu', 'tanh', 'gelu', 'satrelu']) {
  test(`FlyBrain.forward matches the reference implementation (${activation})`, () => {
    const { header, buffer } = buildBlob({ activation, seed: 7, steps: 4 });
    const arrays = parseArrays(header, buffer);
    const brain = new FlyBrain({ header, arrays });
    for (let s = 0; s < 3; s++) {
      const x = randomInput(100 + s);
      const got = brain.forward(x);
      const ref = reference(header, arrays, x);
      assert.equal(got.policy.length, 4168);
      let maxErr = 0;
      for (let i = 0; i < 4168; i++) maxErr = Math.max(maxErr, Math.abs(got.policy[i] - ref.policy[i]));
      assert.ok(maxErr < 1e-3, `policy max err ${maxErr}`);
      assert.ok(Math.abs(got.value - ref.value) < 1e-4, `value ${got.value} vs ${ref.value}`);
      let actErr = 0;
      for (let i = 0; i < header.n; i++) actErr = Math.max(actErr, Math.abs(got.activity[i] - ref.activity[i]));
      assert.ok(actErr < 1e-4, `activity max err ${actErr}`);
      assert.ok(Math.abs(got.value) <= 1);
    }
  });
}

test('linear value head (value_hidden = 0) and scalar alpha', () => {
  const { header, buffer } = buildBlob({ valueHidden: 0, seed: 3 });
  const arrays = parseArrays(header, buffer);
  arrays.alpha = new Float32Array([0.5]);              // scalar export
  const brain = new FlyBrain({ header, arrays });
  assert.equal(brain.valueHidden, 0);
  const x = randomInput(9);
  const got = brain.forward(x);
  const a2 = { ...arrays, alpha: new Float32Array(header.n).fill(0.5) };
  const ref = reference(header, a2, x);
  assert.ok(Math.abs(got.value - ref.value) < 1e-4);
  const batched = brain.forwardBatched([x, x]);
  assert.equal(batched.length, 2);
  assert.equal(batched[0].value, batched[1].value);
});

test('policyForLegal / argmax / sampleIndex / topK', () => {
  const logits = new Float32Array(4168);
  logits[10] = 2; logits[20] = 1; logits[30] = -1; logits[40] = 50;   // 40 is illegal and must be ignored
  const legal = [10, 20, 30];
  const p = policyForLegal(logits, legal, 1);
  const sum = p[0] + p[1] + p[2];
  assert.ok(Math.abs(sum - 1) < 1e-6);
  assert.ok(p[0] > p[1] && p[1] > p[2]);
  assert.ok(Math.abs(p[0] - Math.exp(2) / (Math.exp(2) + Math.exp(1) + Math.exp(-1))) < 1e-6);
  const flat = policyForLegal(logits, legal, 1000);
  assert.ok(Math.abs(flat[0] - 1 / 3) < 1e-3);
  const sharp = policyForLegal(logits, legal, 0.01);
  assert.ok(sharp[0] > 0.999);
  assert.equal(argmax(p), 0);
  assert.equal(sampleIndex(Float32Array.from([0, 0, 1]), () => 0.7), 2);
  assert.equal(sampleIndex(Float32Array.from([0.5, 0.5]), () => 0.2), 0);
  assert.deepEqual(topK(Float32Array.from([1, 5, 3, 4]), 2), [1, 3]);
  assert.equal(policyForLegal(logits, [], 1).length, 0);
});

// ---------------------------------------------------------------------------------------------
// MCTS with a stub brain — needs chess.js (vendored by another module); skipped when absent.
const stubBrain = (value = 0) => ({
  forward: () => ({ policy: new Float32Array(4168), value, activity: new Float32Array(1) }),
  n: 1, lastStepMs: 0,
});

/** Minimal perspective-free encoding, used only when web/engine/encoding.js is not there yet. */
const fallbackEnc = {
  encodeBoard: () => new Float32Array(1280),
  moveToIndex: (m) => {
    const sq = (s) => (s.charCodeAt(1) - 49) * 8 + (s.charCodeAt(0) - 97);
    return sq(m.from) * 64 + sq(m.to);
  },
  legalMoveIndices: (chess) => chess.moves({ verbose: true }).map((m) => fallbackEnc.moveToIndex(m)),
};

test('MCTS returns a legal move and finds a mate in one with a constant-policy stub brain', { skip: !existsSync(chessPath) && 'web/vendor/chess.js not present' }, async () => {
  const { Chess } = await import(fil(chessPath));
  const enc = existsSync(encPath) ? await import(fil(encPath)) : fallbackEnc;
  const chess = new Chess();
  const out = runMCTS(stubBrain(0), chess, enc, { sims: 30, rnd: rng(5), dirichletAlpha: 0.3 });
  const legal = new Set(chess.moves({ verbose: true }).map((m) => m.from + m.to + (m.promotion || '')));
  assert.ok(legal.has(out.move), `${out.move} must be legal`);
  assert.equal(chess.fen(), new Chess().fen(), 'search must leave the board untouched');
  assert.equal(out.visits.reduce((s, v) => s + v.n, 0), 30);
  assert.equal(out.sims, 30);

  chess.load('6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1');
  const mate = runMCTS(stubBrain(0), chess, enc, { sims: 120, rnd: rng(1) });
  assert.equal(mate.move, 'a1a8', `expected Ra8#, got ${mate.move} (${mate.visits.slice(0, 3).map((v) => v.uci + ':' + v.n)})`);
  assert.ok(mate.rootValue > 0.5);

  // black to move, mate in one for black: Qh4# style — test perspective symmetry
  chess.load('rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq g3 0 2');
  const m2 = runMCTS(stubBrain(0), chess, enc, { sims: 120, rnd: rng(2) });
  assert.equal(m2.move, 'd8h4');
});

test('MCTS reports no move on a finished game', { skip: !existsSync(chessPath) && 'web/vendor/chess.js not present' }, async () => {
  const { Chess } = await import(fil(chessPath));
  const enc = existsSync(encPath) ? await import(fil(encPath)) : fallbackEnc;
  const chess = new Chess('7k/5Q2/6K1/8/8/8/8/8 b - - 0 1');   // black is checkmated
  const out = runMCTS(stubBrain(0), chess, enc, { sims: 10 });
  assert.equal(out.move, null);
});

test('MCTS treats a threefold repetition as a draw even when chess.js\' hash counter misses it', { skip: !existsSync(chessPath) && 'web/vendor/chess.js not present' }, async () => {
  const { Chess } = await import(fil(chessPath));
  const enc = await import(fil(encPath));
  // d2d4 sets an ep square chess.js hashes although e4xd3 is illegal (Ra4 pins the pawn): the
  // position after the third Nb1 is a threefold repetition for python-chess / FIDE but not for chess.js
  const chess = new Chess('6n1/8/8/8/R3p2k/8/3P4/1N5K w - - 0 1');
  for (const u of 'd2d4 g8f6 b1c3 f6g8 c3b1 g8f6 b1c3 f6g8 c3b1'.split(' ')) chess.move(enc.uciToMove(u));
  assert.equal(chess.isThreefoldRepetition(), false);
  const out = runMCTS(stubBrain(0.9), chess, enc, { sims: 20 });
  assert.equal(out.move, null, 'the root is terminal (draw)');
  assert.equal(out.rootValue, 0);
  assert.equal(out.sims, 0);
  // one ply earlier the drawing move must be scored 0 for the mover, not the brain's +0.9 leaf value
  chess.undo();
  const search = createSearch(stubBrain(0.9), chess, enc, { sims: 40 });
  while (search.step()) { /* run */ }
  const res = search.result();
  const draw = res.visits.find((v) => v.uci === 'c3b1');
  assert.ok(draw && draw.n > 0);
  assert.ok(Math.abs(draw.q) < 1e-9, `Nb1 repeats the position: q must be 0, got ${draw.q}`);
  assert.equal(chess.history().length, 8, 'the board is restored after the search');
});

test('runMCTSAsync matches runMCTS and stops early on shouldStop', { skip: !existsSync(chessPath) && 'web/vendor/chess.js not present' }, async () => {
  const { Chess } = await import(fil(chessPath));
  const enc = existsSync(encPath) ? await import(fil(encPath)) : fallbackEnc;
  const chess = new Chess();
  const sync = runMCTS(stubBrain(0.1), chess, enc, { sims: 30, rnd: rng(3) });
  const async = await runMCTSAsync(stubBrain(0.1), new Chess(), enc, { sims: 30, rnd: rng(3), yieldEvery: 7 });
  assert.equal(async.stopped, false);
  assert.equal(async.sims, 30);
  assert.deepEqual(async.visits, sync.visits);
  assert.equal(async.move, sync.move);
  // cancel after the first batch: the search stops within one batch, the board is left at the root
  let calls = 0;
  const progress = [];
  const stopped = await runMCTSAsync(stubBrain(0.1), chess, enc, {
    sims: 200, yieldEvery: 10, shouldStop: () => ++calls >= 2, onProgress: (d) => progress.push(d),
  });
  assert.equal(stopped.stopped, true);
  assert.ok(stopped.sims >= 10 && stopped.sims <= 20, `ran ${stopped.sims} sims`);
  assert.equal(progress.at(-1), stopped.sims);
  assert.ok(stopped.move, 'a best-so-far move is still reported');
  assert.equal(chess.history().length, 0);
});
