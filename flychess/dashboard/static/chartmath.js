/* fly-chess dashboard — pure chart helpers (axis ranges, ticks, downsampling).
 *
 * No DOM access: loaded as a classic script by index.html (window.FlyChartMath) and required by the node tests in
 * web/test/dashboard_chart.test.mjs. Everything here must terminate for *any* finite input — these functions run
 * inside requestAnimationFrame and a non-terminating loop freezes the whole tab.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.FlyChartMath = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  /** Hard bound on the number of grid lines per axis, whatever the float arithmetic does. */
  const MAX_TICKS = 1000;

  /** A "nice" (1 / 2 / 5 × 10^k) step that splits `range` into roughly `targetTicks` intervals. */
  function niceStep(range, targetTicks) {
    if (!Number.isFinite(range) || range <= 0) return 1;
    const raw = range / Math.max(1, targetTicks);
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const norm = raw / mag;
    const nice = norm < 1.5 ? 1 : norm < 3.5 ? 2 : norm < 7.5 ? 5 : 10;
    const step = nice * mag;
    return Number.isFinite(step) && step > 0 ? step : 1;
  }

  /**
   * Expand a degenerate [min, max] so that `max - min` is a usable range. A range is degenerate when it is zero
   * *or* only a few ULPs wide (bucket-averaging a constant series yields values 1 ULP apart); `rel` is the
   * relative width below which the range is treated as degenerate.
   */
  function expandRange(min, max, rel) {
    rel = rel === undefined ? 1e-9 : rel;
    if (!Number.isFinite(min) || !Number.isFinite(max)) return [0, 1];
    if (max < min) { const t = min; min = max; max = t; }
    const scale = Math.max(Math.abs(min), Math.abs(max), 1e-300);
    if (max - min <= scale * rel) {
      const e = Math.abs(min) * 0.1 || 0.5;
      min -= e; max += e;
    }
    return [min, max];
  }

  /**
   * Tick values in [min, max] at multiples of `step`, at most MAX_TICKS of them. Each value is computed as
   * `first + i * step` (never accumulated) so a step below the float resolution of `min` cannot stall the loop.
   */
  function ticks(min, max, step) {
    const out = [];
    if (!Number.isFinite(min) || !Number.isFinite(max) || !Number.isFinite(step) || step <= 0 || max < min) return out;
    const first = Math.ceil(min / step) * step;
    for (let i = 0, v = first; i < MAX_TICKS && v <= max; i++, v = first + i * step) out.push(v);
    return out;
  }

  /** Number of decimals needed to print ticks spaced by `step` (0 for step >= 1). */
  function tickDecimals(step) {
    if (!Number.isFinite(step) || step <= 0) return 0;
    return Math.max(0, Math.min(12, -Math.floor(Math.log10(step) + 1e-9)));
  }

  /** Format one tick of a linear axis with a decimal count that is consistent across the axis. */
  function fmtTick(v, step) {
    if (!Number.isFinite(v)) return "–";
    if (Math.abs(v) < step / 2) v = 0; // -0 / 1e-17 noise around zero
    const a = Math.abs(v);
    if (step < 1e-3 || a >= 1e6) return v.toExponential(1);
    const d = tickDecimals(step);
    if (a >= 1000 && d === 0) return v.toLocaleString(undefined, { maximumFractionDigits: 0 });
    return v.toFixed(d);
  }

  /** Equal-count bucket means of parallel `xs` / `ys` arrays so that at most `maxPoints` points remain. */
  function downsample(xs, ys, maxPoints) {
    const n = xs.length;
    if (n <= maxPoints) return { xs, ys };
    const per = Math.ceil(n / maxPoints);
    const oxs = [], oys = [];
    for (let i = 0; i < n; i += per) {
      let sx = 0, sy = 0, c = 0;
      for (let j = i; j < Math.min(n, i + per); j++) { sx += xs[j]; sy += ys[j]; c++; }
      oxs.push(sx / c); oys.push(sy / c);
    }
    return { xs: oxs, ys: oys };
  }

  return { MAX_TICKS, niceStep, expandRange, ticks, tickDecimals, fmtTick, downsample };
});
