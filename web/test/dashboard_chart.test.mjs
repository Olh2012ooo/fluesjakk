// node --test web/test/  -- axis math of the training dashboard (flychess/dashboard/static/chartmath.js).
//
// Regression for the tab freeze: bucket-averaging a constant series can leave ymin / ymax 1 ULP apart, the "nice"
// step then drops below the float resolution of the values and an accumulating tick loop never advances. Every
// helper here must terminate (and stay sane) for such inputs.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const { MAX_TICKS, niceStep, expandRange, ticks, tickDecimals, fmtTick, downsample } =
  require(join(here, '..', '..', 'flychess', 'dashboard', 'static', 'chartmath.js'));

const MAX_POINTS = 2000; // dashboard.js downsampling target

function axisTicks(ys) {
  let ymin = Infinity, ymax = -Infinity;
  for (const y of ys) { if (y < ymin) ymin = y; if (y > ymax) ymax = y; }
  [ymin, ymax] = expandRange(ymin, ymax);
  const pad = (ymax - ymin) * 0.08; ymin -= pad; ymax += pad;
  const step = niceStep(ymax - ymin, 4);
  return { ymin, ymax, step, ticks: ticks(ymin, ymax, step) };
}

test('constant series of 4001 points (1-ULP drift after bucket averaging) yields a bounded tick loop', () => {
  for (const [n, c] of [[4001, 0.1], [2001, 2e-4], [8500, 0.1], [3000, 6.1], [4001, 1e-5]]) {
    const xs = Array.from({ length: n }, (_, i) => i * 20), ys = new Array(n).fill(c);
    const d = downsample(xs, ys, MAX_POINTS);
    assert.ok(d.ys.length <= MAX_POINTS && d.ys.length > 0);
    const { ymin, ymax, step, ticks: t } = axisTicks(d.ys);
    assert.ok(t.length >= 2 && t.length <= 12, `n=${n} c=${c}: ${t.length} ticks`);
    assert.ok(step > 0 && ymax - ymin > Math.abs(c) * 0.1, `n=${n} c=${c}: range ${ymin}..${ymax}`);
    for (const v of t) assert.ok(v >= ymin && v <= ymax);
    // the last tick differs from the first: the loop actually advanced instead of spinning in place
    assert.notEqual(t[0], t[t.length - 1]);
  }
});

test('ticks() terminates even when the step is below the float resolution of the range', () => {
  const t = ticks(0.1, 0.10000000000000002, 2e-18); // the exact values observed in the frozen tab
  assert.ok(t.length <= MAX_TICKS);
  assert.equal(ticks(0, 1, 0).length, 0);
  assert.equal(ticks(0, 1, -1).length, 0);
  assert.equal(ticks(NaN, 1, 0.1).length, 0);
  assert.equal(ticks(1, 0, 0.1).length, 0);
  assert.equal(ticks(0, 1e9, 1e-9).length, MAX_TICKS);
  assert.deepEqual(ticks(0, 1, 0.25), [0, 0.25, 0.5, 0.75, 1]);
  assert.deepEqual(ticks(4999, 5001, 1), [4999, 5000, 5001]);
});

test('niceStep is always a positive finite number', () => {
  for (const r of [0, -1, NaN, Infinity, 1e-300, 3e-17, 1, 12345, 1e300]) {
    const s = niceStep(r, 4);
    assert.ok(Number.isFinite(s) && s > 0, `range ${r} -> ${s}`);
  }
  assert.equal(niceStep(10, 4), 2);
  assert.equal(niceStep(100, 4), 20);
  assert.equal(niceStep(1, 4), 0.2);
});

test('expandRange widens zero / ULP-wide ranges and leaves real ranges alone', () => {
  assert.deepEqual(expandRange(1, 2), [1, 2]);
  assert.deepEqual(expandRange(0, 0), [-0.5, 0.5]);
  const [a, b] = expandRange(0.1, 0.10000000000000002);
  assert.ok(a < 0.095 && b > 0.105);
  const [c, d] = expandRange(5, 5);
  assert.ok(c < 5 && d > 5);
  assert.deepEqual(expandRange(NaN, 1), [0, 1]);
  assert.deepEqual(expandRange(2, 1), [1, 2]);
});

test('tick labels share one decimal count per axis', () => {
  assert.equal(tickDecimals(1), 0);
  assert.equal(tickDecimals(50), 0);
  assert.equal(tickDecimals(0.5), 1);
  assert.equal(tickDecimals(0.02), 2);
  assert.deepEqual([150, 100, 50].map((v) => fmtTick(v, 50)), ['150', '100', '50']);
  assert.deepEqual([0.5, 1, 1.5].map((v) => fmtTick(v, 0.5)), ['0.5', '1.0', '1.5']);
  assert.equal(fmtTick(1e-17, 0.5), '0.0'); // noise around zero is snapped, decimals stay consistent
  assert.equal(fmtTick(-0, 1), '0');
  assert.equal(fmtTick(2e-4, 5e-5), '2.0e-4');
  assert.equal(fmtTick(2000000, 500000), '2.0e+6');
  assert.equal(fmtTick(5000, 1000), (5000).toLocaleString(undefined, { maximumFractionDigits: 0 }));
});

test('downsample keeps short series as-is and bucket-averages long ones', () => {
  const xs = [0, 1, 2], ys = [1, 2, 3];
  assert.deepEqual(downsample(xs, ys, 3), { xs, ys });
  const long = Array.from({ length: 10 }, (_, i) => i);
  const d = downsample(long, long.map((v) => v * 2), 5);
  assert.deepEqual(d.xs, [0.5, 2.5, 4.5, 6.5, 8.5]);
  assert.deepEqual(d.ys, [1, 5, 9, 13, 17]);
});
