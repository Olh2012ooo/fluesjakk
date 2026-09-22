// node --test web/test/features.test.mjs
// The SPEC §8 optional features in both browser engines: retina input (vision, with and without the
// dense sensory path), neuromodulation, readout_steps, the central summary and their combinations,
// on synthetic blobs written exactly like `fly export-web` (web/test/support/synthmodel.mjs), against
// a float64 reference of flychess.export.web.numpy_forward (support/reference.mjs). Also covers
// loader.modelFeatures (old blobs = every feature off) and the activity trace / retina-drive outputs.
// The individual new kernels of the WebGPU engine are covered in flybrain-gpu.test.mjs.
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { parseArrays, modelFeatures } from '../engine/loader.js';
import { FlyBrain } from '../engine/flybrain.js';
import { emulateForward, modelShape } from '../engine/flybrain-gpu.js';
import { synthModel } from './support/synthmodel.mjs';
import { reference, randomInput, maxAbsDiff, FEATURE_CASES } from './support/reference.mjs';

function model(opts = {}, headerExtra = {}) {
  const { header, blob } = synthModel({ seed: 21, steps: 3, ...opts });
  const buffer = new ArrayBuffer(blob.byteLength);
  new Uint8Array(buffer).set(blob);
  Object.assign(header, headerExtra);
  return { header, arrays: parseArrays(header, buffer) };
}

// ---------------------------------------------------------------------------------------------
test('modelFeatures: an old blob has every feature off; flags need their arrays; readout_steps validated', () => {
  const legacy = model({ legacy: true });
  assert.equal(legacy.arrays.retina_idx, undefined);
  assert.deepEqual(modelFeatures(legacy.header, legacy.arrays), { vision: false, sensoryInput: true, readoutSteps: [3], neuromod: false, centralDim: 0, nRet: 0, nCentral: 0, numPlanes: 20, steps: 3, headIn: 7 });
  const off = model();
  assert.equal(off.arrays.retina_idx.length, 0);
  assert.equal(off.arrays.w_ret.length, 0);
  assert.equal(off.arrays.mod_indptr.length, 0);
  assert.deepEqual(modelFeatures(off.header, off.arrays), modelFeatures(legacy.header, legacy.arrays));
  // a header claiming a feature without the arrays runs without it (like numpy_forward)
  assert.equal(modelFeatures({ ...off.header, vision: true, neuromod: true }, off.arrays).vision, false);
  assert.equal(modelFeatures({ ...off.header, vision: true, neuromod: true }, off.arrays).neuromod, false);
  const all = model({ nRet: 12, nnzMod: 60, readoutSteps: [1, 3], centralDim: 5 });
  const F = modelFeatures(all.header, all.arrays);
  assert.deepEqual({ ...F }, { vision: true, sensoryInput: true, readoutSteps: [1, 3], neuromod: true, centralDim: 5, nRet: 12, nCentral: 21, numPlanes: 20, steps: 3, headIn: 7 * 2 + 5 });
  const vo = model({ nRet: 12, sensoryInput: false });
  assert.equal(vo.arrays.w_in.length, 0);
  assert.equal(vo.arrays.input_idx.length, 6, 'input_idx stays for the visualiser');
  assert.equal(modelFeatures(vo.header, vo.arrays).sensoryInput, false);
  for (const bad of [[0, 3], [3, 1], [2, 2], [4]]) assert.throws(() => modelFeatures({ ...all.header, readout_steps: bad }, all.arrays), /readout_steps/);
  assert.throws(() => new FlyBrain(model({ sensoryInput: false })), /no input path/);
  assert.throws(() => modelShape(model({ sensoryInput: false })), /no input path/);
});

for (const [label, opts] of FEATURE_CASES) {
  test(`FlyBrain and the WebGPU mirror match the float64 reference (${label})`, () => {
    const m = model(opts);
    const js = new FlyBrain(m);
    for (let s = 0; s < 3; s++) {
      const x = randomInput(300 + s);
      const ref = reference(m.header, m.arrays, x);
      const cpu = js.forward(x);
      const gpu = emulateForward(m, x);
      assert.equal(cpu.policy.length, 4168);
      assert.ok(maxAbsDiff(cpu.policy, ref.policy) < 1e-3, `FlyBrain policy ${maxAbsDiff(cpu.policy, ref.policy)}`);
      assert.ok(Math.abs(cpu.value - ref.value) < 1e-4, `FlyBrain value ${cpu.value} vs ${ref.value}`);
      assert.ok(maxAbsDiff(cpu.activity, ref.activity) < 1e-4, `FlyBrain activity ${maxAbsDiff(cpu.activity, ref.activity)}`);
      assert.ok(maxAbsDiff(gpu.policy, ref.policy) < 1e-3, `mirror policy ${maxAbsDiff(gpu.policy, ref.policy)}`);
      assert.ok(Math.abs(gpu.value - ref.value) < 1e-4, `mirror value ${gpu.value} vs ${ref.value}`);
      assert.ok(maxAbsDiff(gpu.activity, ref.activity) < 1e-4, `mirror activity ${maxAbsDiff(gpu.activity, ref.activity)}`);
      if (ref.retinaDrive) {
        assert.ok(maxAbsDiff(cpu.retinaDrive, ref.retinaDrive) < 1e-5);
        assert.ok(maxAbsDiff(gpu.retinaDrive, ref.retinaDrive) < 1e-5);
      } else {
        assert.equal(cpu.retinaDrive, null); assert.equal(gpu.retinaDrive, null);
      }
      assert.equal(cpu.trace, null); assert.equal(gpu.trace, null);
    }
  });
}

test('the features change the answer (a blob with them is not silently run without them)', () => {
  const x = randomInput(77);
  const base = new FlyBrain(model()).forward(x);
  const basePolicy = base.policy.slice(), baseActivity = base.activity.slice();
  // same seed → same wiring, heads and inputs where the shapes agree; the feature must move the state
  for (const opts of [{ nRet: 12 }, { nnzMod: 60 }]) {
    const r = new FlyBrain(model(opts)).forward(x);
    assert.ok(maxAbsDiff(r.activity, baseActivity) > 1e-4, JSON.stringify(opts));
  }
  assert.ok(maxAbsDiff(new FlyBrain(model({ nRet: 12 })).forward(x).policy, basePolicy) > 1e-4, 'the retina reaches the heads');
  assert.ok(maxAbsDiff(new FlyBrain(model({ centralDim: 5 })).forward(x).policy, basePolicy) > 1e-4, 'the central summary reaches the heads');
  assert.ok(maxAbsDiff(new FlyBrain(model({ readoutSteps: [1, 3] })).forward(x).policy, basePolicy) > 1e-4, 'the early readout reaches the heads');
});

test('trace mode: sampled activity after every timestep, [t][j] layout, retina drive; nothing without the request', () => {
  const m = model({ nRet: 12, nnzMod: 60, readoutSteps: [1, 3], centralDim: 5, activation: 'satrelu' });
  const js = new FlyBrain(m);
  const x = randomInput(11);
  const ref = reference(m.header, m.arrays, x);
  const idx = Int32Array.from([0, 5, 9, 17, 33, 63, 2]);
  for (const [name, res] of [['FlyBrain', js.forward(x, { trace: idx })], ['mirror', emulateForward(m, x, { trace: idx })]]) {
    assert.equal(res.trace.length, 3 * idx.length, name);
    for (let t = 0; t < 3; t++) for (let j = 0; j < idx.length; j++) {
      assert.ok(Math.abs(res.trace[t * idx.length + j] - ref.states[t][idx[j]]) < 1e-4, `${name} trace[${t}][${j}]`);
    }
    // the last row of the trace is the final activity
    for (let j = 0; j < idx.length; j++) assert.equal(res.trace[2 * idx.length + j], res.activity[idx[j]]);
    assert.ok(maxAbsDiff(res.retinaDrive, ref.retinaDrive) < 1e-5, name);
    assert.ok(maxAbsDiff(res.policy, ref.policy) < 1e-3, `${name} policy with trace on`);
  }
  // an empty trace request and no request both give null
  assert.equal(js.forward(x, { trace: null }).trace, null);
  assert.equal(js.forward(x, {}).trace, null);
  assert.equal(js.forward(x).trace, null);
  // forwardBatched copies the reused views
  const b = js.forwardBatched([x, randomInput(12)], { trace: idx });
  assert.notEqual(b[0].activity, b[1].activity);
  assert.notEqual(b[0].retinaDrive, b[1].retinaDrive);
  assert.equal(b[0].trace.length, 3 * idx.length);
});

test('a photoreceptor that is also an output neuron and duplicate-free maps', () => {
  const m = model({ nRet: 12 });
  const overlap = Array.from(m.arrays.retina_idx).filter((i) => Array.from(m.arrays.output_idx).includes(i));
  // synthModel picks photoreceptors after the outputs, so force one onto an output neuron
  m.arrays.retina_idx[0] = m.arrays.output_idx[0];
  const x = randomInput(5);
  const ref = reference(m.header, m.arrays, x);
  assert.ok(maxAbsDiff(new FlyBrain(m).forward(x).policy, ref.policy) < 1e-3);
  assert.ok(maxAbsDiff(emulateForward(m, x).policy, ref.policy) < 1e-3);
  assert.equal(overlap.length, 0);
  // duplicate photoreceptors are rejected by the GPU path (FlyBrain would sum them)
  m.arrays.retina_idx[1] = m.arrays.retina_idx[0];
  assert.throws(() => emulateForward(m, x), /retina_idx has duplicate/);
});
