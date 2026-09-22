// node --test web/test/parity.test.mjs
// Cross-language parity (SPEC §8): both browser engines evaluated on the exported brain must
// reproduce the Python reference (flychess.export.numpy_forward, every optional feature included)
// in the vectors written by `fly export-web` / `fly test-vectors` from the *same* f16 blob:
//   |Δlogit| <= 1e-2 on the stored top-K legal moves, |Δvalue| <= 1e-2, same argmax over legal moves.
// Engines: FlyBrain (plain JS) and the WebGPU engine through its f32 kernel mirror (emulateForward —
// node has no WebGPU; the real device is checked by web/test/browser/gpu-parity.mjs).
// Model and vectors: FLY_MODEL_DIR (default web/model) and FLY_VECTORS (default
// tests/vectors/model.json), e.g. a scratch export of a fly3-style brain:
//   FLY_MODEL_DIR=/tmp/site3/model FLY_VECTORS=/tmp/site3/model-vectors.json node --test web/test/parity.test.mjs
// FLY_PARITY_EMULATE=0 skips the (slow, ~1 s per position on the full brain) GPU mirror.
// Skips when the model or the vectors are missing, or when they come from different exports.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { gunzipSync } from 'node:zlib';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { Chess } from '../vendor/chess.js';
import { parseArrays } from '../engine/loader.js';
import { FlyBrain } from '../engine/flybrain.js';
import { emulateForward } from '../engine/flybrain-gpu.js';
import { encodeBoard, legalMoveIndices, uciToMove } from '../engine/encoding.js';

const here = dirname(fileURLToPath(import.meta.url));
const modelDir = process.env.FLY_MODEL_DIR || process.env.FLYCHESS_WEB_MODEL || join(here, '..', 'model');
const vectorsPath = process.env.FLY_VECTORS || process.env.FLYCHESS_MODEL_VECTORS || join(here, '..', '..', 'tests', 'vectors', 'model.json');
const emulateGpu = process.env.FLY_PARITY_EMULATE !== '0';
const TOL = 1e-2;

function loadBlob(dir) {
  const raw = join(dir, 'brain.flyb');
  const gz = join(dir, 'brain.flyb.gz');
  let bytes;
  if (existsSync(raw)) bytes = readFileSync(raw);
  else if (existsSync(gz)) bytes = gunzipSync(readFileSync(gz));
  else return null;
  // copy into a fresh, 8-byte aligned ArrayBuffer (Buffer views may share a pool with odd offsets)
  const buffer = new ArrayBuffer(Math.ceil(bytes.byteLength / 8) * 8);
  new Uint8Array(buffer).set(bytes);
  return { buffer, sha256: createHash('sha256').update(bytes).digest('hex'), byteLength: bytes.byteLength };
}

function buildPosition(vec) {
  const chess = new Chess();
  if (vec.moves && vec.moves.length) {
    for (const uci of vec.moves) chess.move(uciToMove(uci));
    assert.equal(chess.fen(), vec.fen);
  } else {
    chess.load(vec.fen);
  }
  return chess;
}

const headerPath = join(modelDir, 'brain.json');
const haveModel = existsSync(headerPath);
const haveVectors = existsSync(vectorsPath);

if (!haveModel || !haveVectors) {
  const why = !haveModel ? `no exported model at ${modelDir} (run \`fly export-web --run <name>\`)`
                         : `no vectors at ${vectorsPath} (run \`fly test-vectors --run <name>\`)`;
  test(`parity: ${why}`, { skip: true }, () => {});
} else {
  const header = JSON.parse(readFileSync(headerPath, 'utf8'));
  const vectors = JSON.parse(readFileSync(vectorsPath, 'utf8'));
  const blob = loadBlob(modelDir);
  const sameExport = blob && (!vectors.blob_sha256 || vectors.blob_sha256 === blob.sha256);

  if (!blob) {
    test('parity: brain.flyb(.gz) missing next to brain.json', { skip: true }, () => {});
  } else if (!sameExport) {
    test(`parity: vectors (${vectors.run_name}@${vectors.exported_at}) were generated from a different export than ` +
         `${modelDir} (${header.run_name}@${header.exported_at}); run \`fly test-vectors\``, { skip: true }, () => {});
  } else {
    assert.equal(blob.byteLength, header.total_bytes, 'blob size matches header');
    const arrays = parseArrays(header, blob.buffer);
    const model = { header, arrays };
    const brain = new FlyBrain(model);
    const F = brain.features;
    const featureNote = [F.vision && 'vision', !F.sensoryInput && 'sensory_input=false', F.readoutSteps.length > 1 && `readout_steps=[${F.readoutSteps}]`,
      F.neuromod && 'neuromod', F.centralDim > 0 && `central_dim=${F.centralDim}`].filter(Boolean).join(', ') || 'none';
    const engines = [['FlyBrain', (x) => brain.forward(x)]];
    if (emulateGpu) engines.push(['WebGPU mirror', (x) => emulateForward(model, x)]);

    test(`parity: header agrees with the vectors (${header.run_name}: features ${featureNote})`, () => {
      assert.equal(header.n, vectors.n);
      assert.equal(header.nnz, vectors.nnz);
      assert.equal(header.steps, vectors.steps);
      assert.ok(vectors.vectors.length >= 12, 'at least 12 curated positions');
      assert.ok(vectors.vectors.some((v) => v.fen.split(' ')[1] === 'b'), 'a black-to-move position');
      assert.ok(vectors.vectors.some((v) => v.moves && v.moves.length), 'a move-list (repetition) position');
    });

    const worst = {};
    for (const [name, forward] of engines) {
      const w = worst[name] = { logit: 0, value: 0 };
      for (const [i, vec] of vectors.vectors.entries()) {
        test(`parity ${name} #${i} ${vec.moves ? vec.moves.join(' ') : vec.fen}`, () => {
          const chess = buildPosition(vec);
          const x = encodeBoard(chess);
          const { policy, value } = forward(x);
          assert.equal(policy.length, header.num_moves);
          assert.ok(Number.isFinite(value));
          const legal = legalMoveIndices(chess);
          assert.equal(legal.length, vec.n_legal, 'legal move count');
          // argmax over the legal moves
          let best = -1, bestV = -Infinity;
          for (const idx of legal) if (policy[idx] > bestV) { bestV = policy[idx]; best = idx; }
          assert.equal(best, vec.argmax, `argmax over legal moves (js ${bestV.toFixed(4)} vs py ${vec.top[0][2].toFixed(4)})`);
          for (const [idx, uci, logit] of vec.top) {
            assert.ok(legal.includes(idx), `${uci} (#${idx}) is legal in JS`);
            const d = Math.abs(policy[idx] - logit);
            w.logit = Math.max(w.logit, d);
            assert.ok(d <= TOL, `logit of ${uci}: js ${policy[idx]} vs py ${logit} (|Δ| ${d} > ${TOL})`);
          }
          const dv = Math.abs(value - vec.value);
          w.value = Math.max(w.value, dv);
          assert.ok(dv <= TOL, `value: js ${value} vs py ${vec.value} (|Δ| ${dv} > ${TOL})`);
        });
      }
    }
    test('parity: summary', () => {
      for (const [name, w] of Object.entries(worst)) {
        console.log(`  parity ok (${name}, ${header.run_name}, features ${featureNote}) on ${vectors.vectors.length} positions: max |Δlogit| ${w.logit.toExponential(2)}, max |Δvalue| ${w.value.toExponential(2)}`);
      }
    });
  }
}
