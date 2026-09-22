// node --test web/test/  -- checks web/engine/encoding.js against tests/vectors/encoding.json
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { Chess } from '../vendor/chess.js';
import {
  FLAT_INPUT, NUM_MOVES, encodeBoard, moveToIndex, indexToMove, legalMoveIndices, legalMoveMask,
  uciToMove, positionRepeated, repetitionCount,
} from '../engine/encoding.js';

const here = dirname(fileURLToPath(import.meta.url));
const vectors = JSON.parse(readFileSync(join(here, '..', '..', 'tests', 'vectors', 'encoding.json'), 'utf8'));

function buildPosition(vec) {
  const chess = new Chess();
  if (vec.move_list) {
    for (const uci of vec.move_list) chess.move(uciToMove(uci));
    assert.equal(chess.fen(), vec.fen);
  } else {
    chess.load(vec.fen);
  }
  return chess;
}

function sha256LE(planes) {
  const buf = Buffer.alloc(planes.length * 4);
  for (let i = 0; i < planes.length; i++) buf.writeFloatLE(planes[i], i * 4);
  return createHash('sha256').update(buf).digest('hex');
}

test('vectors file is non-trivial', () => {
  assert.ok(vectors.length >= 30);
  assert.ok(vectors.some((v) => v.move_list));
  assert.ok(vectors.some((v) => v.repetition_plane === 1));
  assert.ok(vectors.some((v) => v.legal_indices.some((i) => i >= 4096)));
});

for (const [i, vec] of vectors.entries()) {
  const label = `#${i} ${vec.move_list ? vec.move_list.join(' ') : vec.fen}`;

  test(`planes ${label}`, () => {
    const chess = buildPosition(vec);
    const planes = encodeBoard(chess);
    assert.equal(planes.length, FLAT_INPUT);
    // sparse planes 0-17
    const expected = new Float32Array(FLAT_INPUT);
    for (const [p, r, f, v] of vec.planes_nonzero) expected[p * 64 + r * 8 + f] = v;
    expected.fill(Math.fround(vec.halfmove_plane), 18 * 64, 19 * 64);
    expected.fill(vec.repetition_plane, 19 * 64, 20 * 64);
    assert.deepEqual(Array.from(planes), Array.from(expected));
    assert.equal(positionRepeated(chess), vec.repetition_plane === 1);
    assert.equal(sha256LE(planes), vec.planes_sha256);
  });

  test(`moves ${label}`, () => {
    const chess = buildPosition(vec);
    assert.deepEqual(legalMoveIndices(chess), vec.legal_indices);
    const mask = legalMoveMask(chess);
    assert.equal(mask.length, NUM_MOVES);
    assert.equal(mask.reduce((a, b) => a + b, 0), vec.legal_indices.length);
    const seen = new Map();
    for (const move of chess.moves({ verbose: true })) {
      const uci = move.from + move.to + (move.promotion || '');
      const idx = moveToIndex(move, chess);
      assert.equal(idx, vec.moves[uci], `index of ${uci}`);
      assert.equal(moveToIndex(uci, chess), idx);
      assert.ok(!seen.has(idx), `duplicate index ${idx}`);
      seen.set(idx, uci);
      const back = indexToMove(idx, chess);
      assert.equal(back.uci, uci, `round trip of ${uci}`);
      assert.equal(chess.move(back).lan.length >= 4, true); // chess.js accepts the decoded move
      chess.undo();
    }
    assert.equal(seen.size, Object.keys(vec.moves).length);
  });
}

// Threefold repetition must follow python-chess `is_repetition(3)` (legal-ep-only position key),
// not chess.js' Zobrist counter, which hashes the ep square even when the capture is pinned/illegal.
test('repetitionCount counts repetitions chess.js misses after a pinned en-passant double push', () => {
  const chess = new Chess('6n1/8/8/8/R3p2k/8/3P4/1N5K w - - 0 1');
  const counts = [];
  for (const u of 'd2d4 g8f6 b1c3 f6g8 c3b1 g8f6 b1c3 f6g8 c3b1'.split(' ')) {
    chess.move(uciToMove(u));
    counts.push(repetitionCount(chess));
  }
  // after d2d4 the position (e4xd3 e.p. is illegal: Ra4 pins the e4 pawn) occurs once; it recurs
  // after every ...Ng8/Nb1 cycle
  assert.deepEqual(counts, [1, 1, 1, 1, 2, 2, 2, 2, 3]);
  assert.equal(chess.isThreefoldRepetition(), false, 'chess.js undercounts here (that is the bug)');
  assert.equal(positionRepeated(chess), true);
  assert.equal(encodeBoard(chess)[19 * 64], 1);
});

test('repetitionCount agrees with chess.js on an ordinary knight shuffle', () => {
  const chess = new Chess();
  const cycle = 'g1f3 g8f6 f3g1 f6g8'.split(' ');
  assert.equal(repetitionCount(chess), 1);
  for (const u of cycle) chess.move(uciToMove(u));
  assert.equal(repetitionCount(chess), 2);
  assert.equal(chess.isThreefoldRepetition(), false);
  for (const u of cycle) chess.move(uciToMove(u));
  assert.equal(repetitionCount(chess), 3);
  assert.equal(chess.isThreefoldRepetition(), true);
  // a different position (side to move differs) is not a repetition
  chess.move(uciToMove('e2e4'));
  assert.equal(repetitionCount(chess), 1);
});
