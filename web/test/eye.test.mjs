// node --test web/test/eye.test.mjs — the pure parts of the "what the fly sees" UI: eye.js (retina
// map layout, board colouring from the fly's side, gaze commentary from the retina drive) and
// brainviz.js (trace normalisation, per-class means for the thought replay). No DOM needed.
import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
  realSquare, squareName, boardFromFen, layoutEyes, seesColor, driveScale, feelsColor,
  squareDrive, gazeShift, gazeLine, eyeFiles, typeCounts, moveSquares,
} from '../eye.js';
import { sampleGroups, traceGroupMeans, normalizeRows, normalizeTrace, traceRow, FLOW_ORDER } from '../brainviz.js';

const START = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';

test('mover-perspective squares mirror for a black fly (ranks flip, files stay)', () => {
  assert.equal(realSquare(12, 'w'), 12);                  // e2 stays e2 for a white fly
  assert.equal(squareName(realSquare(12, 'b')), 'e7');    // a black fly's "e2" is the real e7
  assert.equal(realSquare(realSquare(37, 'b'), 'b'), 37);
  assert.equal(squareName(0), 'a1'); assert.equal(squareName(63), 'h8');
});

test('boardFromFen places pieces at a1 = 0 … h8 = 63', () => {
  const b = boardFromFen(START);
  assert.deepEqual(b[4], { type: 'k', color: 'w' });
  assert.deepEqual(b[60], { type: 'k', color: 'b' });
  assert.deepEqual(b[8], { type: 'p', color: 'w' });
  assert.equal(b[28], null);
  assert.equal(boardFromFen('8/8/8/8/8/8/8/8 w - - 0 1').filter(Boolean).length, 0);
});

test('seesColor: empty squares keep the board tones, own pieces amber, the opponent blue, bigger pieces brighter', () => {
  const b = boardFromFen(START);
  const light = seesColor(28, b, 'w'), dark = seesColor(27, b, 'w');       // e4 lys, d4 mørk
  assert.deepEqual(light.slice(0, 3), [96, 110, 122]); assert.deepEqual(dark.slice(0, 3), [40, 48, 56]);
  const ownK = seesColor(4, b, 'w'), ownP = seesColor(8, b, 'w'), oppQ = seesColor(59, b, 'w');
  assert.deepEqual(ownK.slice(0, 3), [255, 180, 84]); assert.ok(ownK[3] > ownP[3], 'kongen lyser sterkere enn bonden');
  assert.deepEqual(oppQ.slice(0, 3), [108, 198, 234]);
  // en svart flue: «rute 4» i dens eget perspektiv er den ekte e8, altså dens egen konge
  assert.deepEqual(seesColor(4, b, 'b').slice(0, 3), [255, 180, 84]);
  assert.deepEqual(seesColor(60, b, 'b').slice(0, 3), [108, 198, 234]);
});

test('layoutEyes puts the left eye left, dorsal up, one ring per column, and scales each eye to its box', () => {
  // two columns per eye, 8 photoreceptors each (6 × R1-6 + R7 + R8), plus a lone R7 column
  const uv = [], eye = [], type = [];
  const add = (u, v, e, members) => { for (const t of members) { uv.push(u, v); eye.push(e); type.push(t); } };
  const full = [0, 0, 0, 0, 0, 0, 1, 2];
  add(0.1, 0.2, 0, full); add(0.4, 0.9, 0, full); add(0.6, 0.5, 1, full); add(0.95, 0.1, 1, full); add(0.8, 0.8, 1, [1]);
  const L = layoutEyes(Float32Array.from(uv), Uint8Array.from(eye), Uint8Array.from(type), ['R1-6', 'R7', 'R8'], 300, 150);
  assert.equal(L.x.length, 33);
  for (let k = 0; k < 16; k++) assert.ok(L.x[k] < 150, 'left-eye photoreceptors in the left half');
  for (let k = 16; k < 33; k++) assert.ok(L.x[k] >= 150, 'right-eye photoreceptors in the right half');
  assert.ok(L.y[8] < L.y[0], 'the dorsal column (v = 0.9) is drawn above the ventral one');
  // R7 / R8 at (near) the column centre, R1-6 on a ring around it
  const cx = (L.x[6] + L.x[7]) / 2, cy = L.y[6];
  const radii = [0, 1, 2, 3, 4, 5].map((k) => Math.hypot(L.x[k] - cx, L.y[k] - cy));
  assert.ok(radii.every((r) => r > 1 && Math.abs(r - radii[0]) < 1e-3), `ring radii ${radii}`);
  assert.ok(L.r >= 1.1 && L.r <= 4);
  assert.equal(L.boxes[0].count, 16); assert.equal(L.boxes[1].count, 17); assert.equal(L.boxes[1].columns, 3);
  // a single-eye retina leaves the other box empty
  const one = layoutEyes(Float32Array.from([0.5, 0.5, 0.7, 0.7]), Uint8Array.from([1, 1]), Uint8Array.from([1, 2]), ['R1-6', 'R7', 'R8'], 200, 100);
  assert.equal(one.boxes[0], undefined); assert.equal(one.boxes[1].count, 2);
});

test('feels: the scale is the 98th percentile of |drive|, colours split by sign and saturate', () => {
  const d = Float32Array.from({ length: 100 }, (_, i) => (i % 2 ? 1 : -1) * i / 10);
  const s = driveScale(d);
  assert.ok(s >= 9.6 && s <= 9.9, `scale ${s}`);
  assert.equal(driveScale(new Float32Array(0)), 1); assert.equal(driveScale(null), 1);
  assert.equal(driveScale(new Float32Array(5)), 1e-6, 'all-zero drive keeps a positive scale');
  const pos = feelsColor(s, s), neg = feelsColor(-s, s), zero = feelsColor(0, s), huge = feelsColor(100 * s, s);
  assert.ok(pos[1] > pos[0] && pos[1] > pos[2], 'positive = green');
  assert.ok(neg[0] > neg[1] && neg[0] > neg[2], 'negative = red');
  assert.ok(zero[3] < 0.2 && pos[3] === 1);
  assert.deepEqual(huge, pos, 'saturates at the scale');
});

test('squareDrive averages per (eye, square) and gazeShift finds the largest change', () => {
  const square = Uint8Array.from([28, 28, 28, 5, 5, 60]);
  const eye = Uint8Array.from([0, 0, 1, 1, 1, 0]);
  const a = squareDrive(Float32Array.from([1, 3, 10, 0, 0, 2]), square, eye);
  assert.equal(a[28], 2); assert.equal(a[64 + 28], 10); assert.equal(a[64 + 5], 0); assert.equal(a[60], 2);
  assert.ok(Number.isNaN(a[64 + 60]) && Number.isNaN(a[0]), 'unseen (eye, square) pairs are NaN');
  const b = squareDrive(Float32Array.from([1, 3, 10, 4, 4, 2.5]), square, eye);
  assert.deepEqual(gazeShift(a, b), { eye: 1, square: 5, delta: 4 });
  assert.equal(gazeShift(a, a), null, 'nothing changed');
  assert.equal(gazeShift(null, b), null, 'no previous glance');
  assert.equal(gazeShift(a, b, 10), null, 'below the threshold');
});

test('gazeLine navngir øyet, brikken og den ekte ruten sett fra fluas side', () => {
  const b = boardFromFen('rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2');
  const first = () => 0;
  // en hvit flue: rute 45 (f6) holder springeren din
  assert.equal(gazeLine({ eye: 1, square: 45, delta: 1 }, b, 'w', first), 'Fluas høyre øye har festet seg på springeren din på f6.');
  // en svart flue: rute 28 (e4 i dens eget perspektiv) er den ekte e5 — tom; rute 36 er den ekte e4: bonden din
  assert.equal(gazeLine({ eye: 0, square: 28, delta: 1 }, b, 'b', first), 'Fluas venstre øye har festet seg på den tomme ruten e5.');
  assert.equal(gazeLine({ eye: 0, square: 36, delta: 1 }, b, 'b', first), 'Fluas venstre øye har festet seg på bonden din på e4.');
  assert.equal(gazeLine({ eye: 0, square: 4, delta: 1 }, b, 'b', () => 0.5), 'Det venstre øyet lyste opp over sin egen konge på e8.');
  assert.equal(gazeLine(null, b, 'w'), '');
});

test('gazeLine bruker riktig eiendomsord for tårn, som er intetkjønn', () => {
  const b = boardFromFen('4k3/8/8/8/8/8/8/R3K3 w - - 0 1');   // a1: tårnet ditt
  // rute 56 i den svarte fluas perspektiv er den ekte a1
  assert.equal(gazeLine({ eye: 0, square: 56, delta: 1 }, b, 'b', () => 0),
    'Fluas venstre øye har festet seg på tårnet ditt på a1.');
  // og sitt eget tårn står i ubestemt form etter «eget» (rute 0 er a1 for en hvit flue)
  assert.equal(gazeLine({ eye: 1, square: 0, delta: 1 }, b, 'w', () => 0),
    'Fluas høyre øye har festet seg på sitt eget tårn på a1.');
});

test('eyeFiles / typeCounts summarise the header for the landing page', () => {
  const square = Uint8Array.from([0, 9, 18, 27, 36, 45, 54, 63, 4]);   // files a b c d | e f g h, plus e for the left eye
  const eye = Uint8Array.from([0, 0, 0, 0, 1, 1, 1, 1, 1]);
  assert.deepEqual(eyeFiles(square, eye), { left: 'a–d', right: 'e–h', leftCount: 4, rightCount: 5 });
  assert.equal(eyeFiles(Uint8Array.from([0, 2]), Uint8Array.from([0, 0])).left, 'a, c');
  assert.equal(eyeFiles(Uint8Array.from([0]), Uint8Array.from([0])).right, '—');
  assert.deepEqual(typeCounts(Uint8Array.from([0, 0, 1, 2, 2, 2]), ['R1-6', 'R7', 'R8']), [['R1-6', 2], ['R7', 1], ['R8', 3]]);
  assert.deepEqual(moveSquares('e2e4'), [12, 28]); assert.deepEqual(moveSquares(null), []);
});

test('sampleGroups: rows in signal order, photoreceptors in their own retina row', () => {
  const legend = ['unknown', 'optic', 'central', 'sensory', 'descending'];
  const idx = Int32Array.from([10, 11, 12, 13, 14, 15]);
  const cls = Uint8Array.from([1, 3, 2, 4, 3, 1]);
  const g = sampleGroups(idx, cls, legend, Int32Array.from([11, 99]));
  assert.deepEqual(g.map((x) => x.name), ['retina', 'sensory', 'optic', 'central', 'descending']);
  assert.deepEqual([...g[0].idx], [1]); assert.deepEqual([...g[1].idx], [4]); assert.deepEqual([...g[2].idx], [0, 5]);
  const noRet = sampleGroups(idx, cls, legend, null);
  assert.deepEqual(noRet.map((x) => x.name), ['sensory', 'optic', 'central', 'descending']);
  assert.equal(FLOW_ORDER.indexOf('retina'), 0); assert.ok(FLOW_ORDER.indexOf('optic') < FLOW_ORDER.indexOf('central'));
  assert.ok(FLOW_ORDER.indexOf('central') < FLOW_ORDER.indexOf('descending'));
  // an unlisted class name still gets a row (after the known ones)
  const odd = sampleGroups(Int32Array.from([1]), Uint8Array.from([0]), ['glia'], null);
  assert.deepEqual(odd.map((x) => x.name), ['glia']);
});

test('traceGroupMeans / normalizeRows: mean |activity| per row and step, each row scaled to its own max', () => {
  const steps = 3, m = 4;
  // neuron j after step t = (t + 1) * (j + 1), neuron 3 negative
  const trace = Float32Array.from({ length: steps * m }, (_, i) => { const t = Math.floor(i / m), j = i % m; return (j === 3 ? -1 : 1) * (t + 1) * (j + 1); });
  const groups = [{ name: 'a', idx: Int32Array.from([0, 1]) }, { name: 'b', idx: Int32Array.from([3]) }, { name: 'empty', idx: new Int32Array(0) }];
  const means = traceGroupMeans(trace, steps, groups);
  assert.deepEqual([...means], [1.5, 3, 4.5, 4, 8, 12, 0, 0, 0]);
  const norm = normalizeRows(means, steps);
  assert.deepEqual([...norm].map((v) => +v.toFixed(4)), [1 / 3, 2 / 3, 1, 1 / 3, 2 / 3, 1, 0, 0, 0].map((v) => +v.toFixed(4)));
});

test('normalizeTrace / traceRow: log-compressed, p97-scaled over the whole trace, interpolated between steps', () => {
  const trace = Float32Array.from([0, 0, 0, 0, 1, 2, 3, 4, 2, 4, 6, 8]);   // 3 steps × 4 neurons
  const n = normalizeTrace(trace);
  assert.equal(n.length, 12);
  assert.ok(n.slice(0, 4).every((v) => v === 0), 'silent first step stays dark');
  assert.equal(Math.max(...n), 1, 'the maximum reaches 1');
  assert.ok(n[11] >= n[7] && n[7] >= n[5], 'monotone in |activity|');
  const row = traceRow(n, 4, 1.5, new Float32Array(4));
  for (let j = 0; j < 4; j++) assert.ok(Math.abs(row[j] - (n[4 + j] + n[8 + j]) / 2) < 1e-6, 'halfway between steps 2 and 3');
  const last = traceRow(n, 4, 2, new Float32Array(4));
  assert.deepEqual([...last], [...n.slice(8)]);
  assert.ok(normalizeTrace(new Float32Array(4)).every((v) => v === 0), 'an all-zero trace does not divide by zero');
});
