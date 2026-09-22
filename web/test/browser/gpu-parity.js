// Browser side of web/test/browser/gpu-parity.mjs: loads the exported model from ../../model/,
// builds FlyBrain (plain JS) and FlyBrainGPU on the same arrays and compares them on a few
// positions (logits, value, final activity, the per-step activity trace and the retina drive),
// compares both engines with the Python reference vectors when the runner serves /vectors.json,
// then times both. Exposes window.gpuParity(opts) → result object (JSON-serialisable).
import { loadBrain } from '../../engine/loader.js';
import { FlyBrain } from '../../engine/flybrain.js';
import { FlyBrainGPU, emulateForward } from '../../engine/flybrain-gpu.js';
import { Chess } from '../../vendor/chess.js';
import * as enc from '../../engine/encoding.js';

const POSITIONS = [
  { name: 'start', moves: [] },
  { name: 'italian', moves: ['e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1c4', 'f8c5'] },
  { name: 'black to move', moves: ['d2d4', 'g8f6', 'c2c4', 'e7e6', 'b1c3', 'f8b4', 'd1c2'] },
  { name: 'middlegame', fen: 'r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP3PPP/R2QKB1R w KQ - 2 8' },
  { name: 'endgame', fen: '8/5pk1/6p1/8/3K4/8/5PPP/8 w - - 0 40' },
];
const TRACE_N = 2048;   // like the worker's activity sample

function encode(pos) {
  const chess = new Chess();
  if (pos.moves) for (const u of pos.moves) chess.move(enc.uciToMove(u));
  else chess.load(pos.fen);
  const moves = chess.moves({ verbose: true });
  return { x: enc.encodeBoard(chess), legal: moves.map((m) => enc.moveToIndex(m, chess)) };
}

const maxAbs = (a, b) => { let m = 0; for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i])); return m; };
const argmaxLegal = (p, legal) => legal.reduce((best, i) => (p[i] > p[best] ? i : best), legal[0]);
const log = (s) => { document.getElementById('out').textContent += s + '\n'; console.log(s); };

/** Spread sample of neuron indices (deterministic). */
function sampleIdx(n, k) {
  const out = new Int32Array(Math.min(k, n));
  const stride = n / out.length;
  for (let i = 0; i < out.length; i++) out[i] = Math.min(n - 1, Math.floor(i * stride + (i * 7919) % Math.max(1, Math.floor(stride))));
  return out;
}

/** Compare an engine's output with one Python vector (SPEC §8 tolerance semantics). */
function vsPython(policy, value, legal, vec) {
  let best = -1, bestV = -Infinity;
  for (const idx of legal) if (policy[idx] > bestV) { bestV = policy[idx]; best = idx; }
  let dLogit = 0;
  for (const [idx, , logit] of vec.top) dLogit = Math.max(dLogit, Math.abs(policy[idx] - logit));
  return { dLogit, dValue: Math.abs(value - vec.value), sameArgmax: best === vec.argmax, nLegal: legal.length === vec.n_legal };
}

window.gpuParity = async ({ reps = 20, emulate = true } = {}) => {
  const model = await loadBrain(new URL('../../model/', location.href).href);
  const js = new FlyBrain(model);
  const t0 = performance.now();
  const gpu = await FlyBrainGPU.create(model);
  const createMs = performance.now() - t0;
  const adapterInfo = gpu.adapter?.info ? JSON.parse(JSON.stringify(gpu.adapter.info)) : null;
  const F = js.features;
  const features = { vision: F.vision, sensoryInput: F.sensoryInput, readoutSteps: F.readoutSteps, neuromod: F.neuromod, centralDim: F.centralDim, nRet: F.nRet, nCentral: F.nCentral };
  log(`model ${model.header.run_name}: n=${js.n} nnz=${js.nnz} steps=${js.steps} act=${js.activation} features=${JSON.stringify(features)}; GPU ready in ${createMs.toFixed(0)} ms, timestamps=${gpu.timestamps}`);
  const trace = sampleIdx(js.n, TRACE_N);

  const positions = [];
  for (const pos of POSITIONS) {
    const { x, legal } = encode(pos);
    const a = js.forward(x, { trace });
    const aPolicy = a.policy.slice(), aActivity = a.activity.slice(), aTrace = a.trace, aRet = a.retinaDrive ? a.retinaDrive.slice() : null;
    const b = await gpu.forward(x, { trace });
    const r = {
      name: pos.name,
      dLogit: maxAbs(aPolicy, b.policy),
      dLogitLegal: Math.max(...legal.map((i) => Math.abs(aPolicy[i] - b.policy[i]))),
      dValue: Math.abs(a.value - b.value),
      dActivity: maxAbs(aActivity, b.activity),
      dTrace: b.trace && b.trace.length === js.steps * trace.length ? maxAbs(aTrace, b.trace) : Infinity,
      dRetina: F.vision ? (b.retinaDrive && b.retinaDrive.length === F.nRet ? maxAbs(aRet, b.retinaDrive) : Infinity) : (b.retinaDrive === null ? 0 : Infinity),
      sameArgmax: argmaxLegal(aPolicy, legal) === argmaxLegal(b.policy, legal),
      valueJs: a.value, valueGpu: b.value,
      maxActivity: b.activity.reduce((m, v) => Math.max(m, v), 0),
    };
    // the last trace row is the final activity of the sampled neurons
    let dLast = 0;
    for (let j = 0; j < trace.length; j++) dLast = Math.max(dLast, Math.abs(b.trace[(js.steps - 1) * trace.length + j] - b.activity[trace[j]]));
    r.dTraceLast = dLast;
    if (dLast > 0) r.dTrace = Infinity;
    if (emulate) {
      const e = emulateForward(model, x);
      r.dLogitEmu = maxAbs(e.policy, b.policy);
      r.dActivityEmu = maxAbs(e.activity, b.activity);
      r.dValueEmu = Math.abs(e.value - b.value);
    }
    positions.push(r);
    log(`${pos.name.padEnd(14)} |Δlogit| ${r.dLogit.toExponential(2)} (legal ${r.dLogitLegal.toExponential(2)})  |Δvalue| ${r.dValue.toExponential(2)}  |Δh| ${r.dActivity.toExponential(2)}  |Δtrace| ${r.dTrace.toExponential(2)}  |Δretina| ${r.dRetina.toExponential(2)}  argmax ${r.sameArgmax ? 'same' : 'DIFFERENT'}` + (emulate ? `  vs emulation: logit ${r.dLogitEmu.toExponential(2)} h ${r.dActivityEmu.toExponential(2)}` : ''));
  }
  // a plain forward (no trace) must give the same answer as a traced one
  {
    const { x } = encode(POSITIONS[1]);
    const plain = await gpu.forward(x), traced = await gpu.forward(x, { trace });
    positions[1].dPlainVsTraced = maxAbs(plain.policy, traced.policy);
    if (positions[1].dPlainVsTraced > 0 || plain.trace !== null) positions[1].dTrace = Infinity;
    log(`plain vs traced GPU forward: |Δlogit| ${positions[1].dPlainVsTraced.toExponential(2)}, plain.trace = ${plain.trace}`);
  }

  // the Python reference vectors, when the runner serves them
  let python = null;
  const vr = await fetch(new URL('/vectors.json', location.href)).catch(() => null);
  if (vr && vr.ok) {
    const vectors = await vr.json();
    python = { runName: vectors.run_name, positions: [], gpu: { logit: 0, value: 0 }, js: { logit: 0, value: 0 } };
    for (const vec of vectors.vectors) {
      const chess = new Chess();
      if (vec.moves && vec.moves.length) for (const u of vec.moves) chess.move(enc.uciToMove(u)); else chess.load(vec.fen);
      const legal = enc.legalMoveIndices(chess);
      const x = enc.encodeBoard(chess);
      const a = js.forward(x);
      const aPolicy = a.policy.slice(), aValue = a.value;
      const b = await gpu.forward(x, { activity: false });
      const p = { name: vec.moves ? vec.moves.join(' ') : vec.fen, js: vsPython(aPolicy, aValue, legal, vec), gpu: vsPython(b.policy, b.value, legal, vec) };
      python.positions.push(p);
      for (const k of ['js', 'gpu']) { python[k].logit = Math.max(python[k].logit, p[k].dLogit); python[k].value = Math.max(python[k].value, p[k].dValue); }
      log(`python ${p.name.slice(0, 30).padEnd(30)} gpu |Δlogit| ${p.gpu.dLogit.toExponential(2)} |Δvalue| ${p.gpu.dValue.toExponential(2)} argmax ${p.gpu.sameArgmax ? 'same' : 'DIFFERENT'} | js |Δlogit| ${p.js.dLogit.toExponential(2)} |Δvalue| ${p.js.dValue.toExponential(2)} argmax ${p.js.sameArgmax ? 'same' : 'DIFFERENT'}`);
    }
  }

  // latency (single forward, serial; the MCTS pattern)
  const { x } = encode(POSITIONS[1]);
  const time = async (fn, n) => {
    const ts = [];
    for (let i = 0; i < n; i++) { const t = performance.now(); await fn(); ts.push(performance.now() - t); }
    ts.sort((p, q) => p - q);
    return { median: ts[ts.length >> 1], min: ts[0], max: ts[ts.length - 1], mean: ts.reduce((s, v) => s + v, 0) / ts.length };
  };
  await gpu.forward(x); await gpu.forward(x);
  const gpuFull = await time(() => gpu.forward(x), reps);
  const gpuStepMs = gpu.lastStepMs;
  const gpuNoAct = await time(() => gpu.forward(x, { activity: false }), reps);
  const gpuTrace = await time(() => gpu.forward(x, { trace }), reps);
  const jsReps = Math.max(3, Math.min(reps, 5));
  const jsT = await time(() => js.forward(x), jsReps);
  const jsStep = js.lastStepMs;
  const jsTrace = await time(() => js.forward(x, { trace }), jsReps);
  const timing = { gpuForwardMs: gpuFull, gpuForwardNoActivityMs: gpuNoAct, gpuForwardTraceMs: gpuTrace, gpuStepMs, jsForwardMs: jsT, jsForwardTraceMs: jsTrace, jsStepMs: jsStep, createMs };
  log(`GPU forward: median ${gpuFull.median.toFixed(2)} ms (min ${gpuFull.min.toFixed(2)}, max ${gpuFull.max.toFixed(2)}), without activity readback ${gpuNoAct.median.toFixed(2)} ms, with trace ${gpuTrace.median.toFixed(2)} ms; GPU step ${gpuStepMs.toFixed(3)} ms${gpu.timestamps ? ' (timestamp query)' : ' (estimate)'}`);
  log(`JS  forward: median ${jsT.median.toFixed(1)} ms (with trace ${jsTrace.median.toFixed(1)} ms), step ${jsStep.toFixed(2)} ms`);
  const limits = gpu.device.limits;
  return { header: { run_name: model.header.run_name, n: js.n, nnz: js.nnz, steps: js.steps, activation: js.activation }, features, adapterInfo, timestamps: gpu.timestamps, limits: { maxStorageBufferBindingSize: limits.maxStorageBufferBindingSize, maxComputeWorkgroupsPerDimension: limits.maxComputeWorkgroupsPerDimension }, positions, python, timing };
};
