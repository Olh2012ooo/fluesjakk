/* fly-chess training dashboard — vanilla JS, no build step, no CDN.
 *
 * Sections: utils · Chart (canvas, hover, log toggle, downsampling) · chess replay (SAN resolver + SVG board)
 *           · neuron activity canvas · log tail · store / websocket / run selector.
 */
(function () {
  "use strict";

  // ------------------------------------------------------------------ utils
  const $ = (sel, root) => (root || document).querySelector(sel);
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const SERIES = ["--s1", "--s2", "--s3", "--s4", "--s5", "--s6", "--s7", "--s8"].map(css);

  function fmtNum(v, digits) {
    if (v === null || v === undefined || Number.isNaN(v)) return "–";
    const a = Math.abs(v);
    if (a === 0) return "0";
    if (a < 1e-2 || a >= 1e6) return v.toExponential(digits === undefined ? 2 : digits);
    if (a >= 1000) return v.toLocaleString(undefined, { maximumFractionDigits: 0 });
    if (a >= 100) return v.toFixed(1);
    if (a >= 1) return v.toFixed(digits === undefined ? 3 : digits);
    return v.toFixed(digits === undefined ? 4 : digits);
  }
  function fmtInt(v) { return v === null || v === undefined ? "–" : Math.round(v).toLocaleString(); }
  function fmtDuration(s) {
    if (s === null || s === undefined || !Number.isFinite(s)) return "–";
    s = Math.max(0, Math.round(s));
    const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
    if (h > 48) return `${(h / 24).toFixed(1)} d`;
    if (h) return `${h}h ${String(m).padStart(2, "0")}m`;
    if (m) return `${m}m ${String(sec).padStart(2, "0")}s`;
    return `${sec}s`;
  }
  function fmtTime(t) {
    if (!t) return "";
    const d = new Date(t * 1000);
    return d.toLocaleTimeString(undefined, { hour12: false });
  }
  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }
  function lowerBound(arr, x) { // first index with arr[i] >= x
    let lo = 0, hi = arr.length;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (arr[mid] < x) lo = mid + 1; else hi = mid; }
    return lo;
  }
  // axis math (nice steps, degenerate-range expansion, bounded tick generation, downsampling) — see chartmath.js
  const { niceStep, expandRange, ticks, fmtTick, downsample } = window.FlyChartMath;

  // ------------------------------------------------------------------ Chart
  const MAX_POINTS = 2000; // bucketed downsampling target
  const tooltipEl = $("#tooltip");

  class Chart {
    constructor(container, opts) {
      this.el = container;
      this.opts = opts;
      this.log = !!opts.log;
      this.series = new Map(); // key -> {label, color, dash, points, xs:[], ys:[], ds:{n, xs, ys}}
      this.hoverX = null;
      this.dirty = true;
      this._raf = 0;

      this.el.innerHTML = `
        <div class="chart-head"><h3>${escapeHtml(opts.title)}</h3>
          <div class="controls"><button class="btn log-toggle" aria-pressed="${this.log}" title="log scale y-axis">log</button></div></div>
        <canvas></canvas>
        <div class="legend"></div>
        <div class="empty">no data yet</div>`;
      this.canvas = $("canvas", this.el);
      this.ctx = this.canvas.getContext("2d");
      this.legendEl = $(".legend", this.el);
      this.emptyEl = $(".empty", this.el);
      $(".log-toggle", this.el).addEventListener("click", (e) => {
        this.log = !this.log; e.currentTarget.setAttribute("aria-pressed", String(this.log)); this.requestRender();
      });
      this.canvas.addEventListener("mousemove", (e) => this.onHover(e));
      this.canvas.addEventListener("mouseleave", () => { this.hoverX = null; tooltipEl.hidden = true; this.requestRender(); });
      new ResizeObserver(() => this.requestRender()).observe(this.canvas);
      (opts.series || []).forEach((s) => this.addSeries(s.key, s));
    }

    addSeries(key, spec) {
      if (this.series.has(key)) return this.series.get(key);
      const s = {
        key, label: spec.label || key, color: spec.color || SERIES[this.series.size % SERIES.length],
        dash: !!spec.dash, points: !!spec.points, xs: [], ys: [], ds: null,
      };
      this.series.set(key, s);
      this.renderLegend();
      return s;
    }
    clear() {
      const fixed = new Set((this.opts.series || []).map((s) => s.key));
      for (const [key, s] of this.series) {
        if (fixed.has(key)) { s.xs = []; s.ys = []; s.ds = null; } else this.series.delete(key); // dynamic (e.g. Elo opponents)
      }
      this.renderLegend();
      this.requestRender();
    }
    append(key, x, y, spec) {
      if (typeof y !== "number" || !Number.isFinite(y) || typeof x !== "number") return;
      const s = this.series.get(key) || this.addSeries(key, spec || {});
      s.xs.push(x); s.ys.push(y); s.ds = null;
      this.requestRender();
    }

    /** Bucketed downsampling: equal-count buckets, mean x / mean y (cached until the series changes). */
    downsampled(s) {
      if (s.ds && s.ds.n === s.xs.length) return s.ds;
      const { xs, ys } = downsample(s.xs, s.ys, MAX_POINTS);
      s.ds = { n: s.xs.length, xs, ys };
      return s.ds;
    }

    requestRender() {
      if (this._raf) return;
      this._raf = requestAnimationFrame(() => { this._raf = 0; this.render(); });
    }

    renderLegend() {
      const parts = [];
      for (const s of this.series.values()) {
        const last = s.ys.length ? fmtNum(s.ys[s.ys.length - 1]) : "";
        const cls = s.points ? "sw pt" : s.dash ? "sw dash" : "sw";
        const style = s.points ? `background:${s.color}` : `border-color:${s.color}`;
        parts.push(`<span class="item"><span class="${cls}" style="${style}"></span>${escapeHtml(s.label)} <span class="last">${last}</span></span>`);
      }
      this.legendEl.innerHTML = parts.join("");
    }

    yTransform(v) { return this.log ? Math.log10(v) : v; }

    render() {
      const dpr = window.devicePixelRatio || 1;
      const W = this.canvas.clientWidth, H = this.canvas.clientHeight;
      if (!W || !H) return;
      if (this.canvas.width !== Math.round(W * dpr) || this.canvas.height !== Math.round(H * dpr)) {
        this.canvas.width = Math.round(W * dpr); this.canvas.height = Math.round(H * dpr);
      }
      const ctx = this.ctx;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);
      this.renderLegend();

      // ranges
      let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity, any = false;
      const views = [];
      for (const s of this.series.values()) {
        const d = this.downsampled(s);
        if (!d.xs.length) continue;
        const vx = [], vy = [];
        for (let i = 0; i < d.xs.length; i++) {
          const y = d.ys[i];
          if (this.log && y <= 0) continue;
          vx.push(d.xs[i]); vy.push(this.yTransform(y));
        }
        if (!vx.length) continue;
        any = true;
        xmin = Math.min(xmin, vx[0]); xmax = Math.max(xmax, vx[vx.length - 1]);
        for (const y of vy) { if (y < ymin) ymin = y; if (y > ymax) ymax = y; }
        views.push({ s, vx, vy });
      }
      this.emptyEl.hidden = any;
      if (!any) return;
      // Degenerate ranges: a constant series (or one that is constant up to float noise after bucket averaging)
      // must still yield a step that advances the tick loops below.
      if (xmax - xmin <= Math.max(Math.abs(xmin), Math.abs(xmax), 1) * 1e-9) {
        const ext = this.opts.xExtent ? this.opts.xExtent() : null; // e.g. the run's full step range
        if (ext && Number.isFinite(ext[0]) && Number.isFinite(ext[1]) && ext[0] <= xmin && ext[1] > ext[0]) {
          xmin = ext[0]; xmax = Math.max(ext[1], xmax);
        } else { const e = Math.max(1, Math.abs(xmin) * 0.05); xmin -= e; xmax += e; }
      }
      [ymin, ymax] = expandRange(ymin, ymax);
      const ypad = (ymax - ymin) * 0.08; ymin -= ypad; ymax += ypad;
      if (!this.log && this.opts.zeroBased && ymin > 0) ymin = 0;

      const pad = { l: 48, r: 10, t: 8, b: 20 };
      const pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;
      const X = (x) => pad.l + ((x - xmin) / (xmax - xmin)) * pw;
      const Y = (y) => pad.t + ph - ((y - ymin) / (ymax - ymin)) * ph;

      // grid + axes
      ctx.font = "10px " + css("--mono");
      ctx.fillStyle = css("--muted");
      ctx.strokeStyle = css("--line");
      ctx.lineWidth = 1;
      const ystep = niceStep(ymax - ymin, 4);
      for (const v of ticks(ymin, ymax, ystep)) {
        const y = Math.round(Y(v)) + 0.5;
        ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
        const label = this.log ? fmtNum(Math.pow(10, v), 1) : fmtTick(v, ystep);
        ctx.textAlign = "right"; ctx.textBaseline = "middle"; ctx.fillText(label, pad.l - 5, y);
      }
      const xstep = Math.max(1, niceStep(xmax - xmin, Math.max(2, Math.floor(pw / 90))));
      ctx.textAlign = "center"; ctx.textBaseline = "top";
      for (const v of ticks(xmin, xmax, xstep)) {
        const x = Math.round(X(v)) + 0.5;
        ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, pad.t + ph); ctx.stroke();
        const label = fmtInt(v), half = ctx.measureText(label).width / 2;
        ctx.fillText(label, Math.min(Math.max(x, pad.l + half), W - pad.r - half), pad.t + ph + 4); // keep inside the canvas
      }

      // series
      for (const { s, vx, vy } of views) {
        ctx.strokeStyle = s.color; ctx.fillStyle = s.color;
        if (s.points) {
          for (let i = 0; i < vx.length; i++) {
            ctx.beginPath(); ctx.arc(X(vx[i]), Y(vy[i]), 4, 0, Math.PI * 2); ctx.fill();
          }
          if (vx.length > 1) { ctx.globalAlpha = 0.35; ctx.setLineDash([2, 4]); ctx.lineWidth = 1; }
          else continue;
        } else {
          ctx.lineWidth = s.dash ? 1.5 : 2;
          ctx.setLineDash(s.dash ? [5, 4] : []);
          ctx.globalAlpha = s.dash ? 0.9 : 1;
        }
        ctx.beginPath();
        for (let i = 0; i < vx.length; i++) { const x = X(vx[i]), y = Y(vy[i]); if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y); }
        ctx.stroke();
        ctx.setLineDash([]); ctx.globalAlpha = 1;
      }

      // hover crosshair
      if (this.hoverX !== null) {
        const xv = xmin + ((this.hoverX - pad.l) / pw) * (xmax - xmin);
        if (xv >= xmin && xv <= xmax) {
          const rows = [];
          let xStep = null;
          for (const { s, vx, vy } of views) {
            let i = lowerBound(vx, xv);
            if (i > 0 && (i >= vx.length || xv - vx[i - 1] < vx[i] - xv)) i--;
            if (i >= vx.length) continue;
            const yv = this.log ? Math.pow(10, vy[i]) : vy[i];
            rows.push({ s, x: vx[i], y: yv, px: X(vx[i]), py: Y(vy[i]) });
            if (xStep === null || Math.abs(vx[i] - xv) < Math.abs(xStep - xv)) xStep = vx[i];
          }
          if (rows.length) {
            const cx = Math.round(X(xStep)) + 0.5;
            ctx.strokeStyle = css("--amber"); ctx.globalAlpha = 0.5; ctx.setLineDash([3, 3]);
            ctx.beginPath(); ctx.moveTo(cx, pad.t); ctx.lineTo(cx, pad.t + ph); ctx.stroke();
            ctx.setLineDash([]); ctx.globalAlpha = 1;
            for (const r of rows) {
              ctx.fillStyle = r.s.color; ctx.beginPath(); ctx.arc(r.px, r.py, 4, 0, Math.PI * 2); ctx.fill();
              ctx.strokeStyle = css("--bg"); ctx.lineWidth = 2; ctx.stroke();
            }
            this.showTooltip(rows, xStep);
          }
        }
      }
    }

    onHover(e) {
      const rect = this.canvas.getBoundingClientRect();
      this.hoverX = e.clientX - rect.left;
      this._lastMouse = { x: e.clientX, y: e.clientY };
      this.requestRender();
    }
    showTooltip(rows, xStep) {
      const m = this._lastMouse || { x: 0, y: 0 };
      tooltipEl.innerHTML = `<div class="tt-x">step ${fmtInt(xStep)}</div>` + rows.map((r) =>
        `<div class="tt-row"><span><span class="sw" style="background:${r.s.color}"></span>${escapeHtml(r.s.label)}</span><span class="val">${fmtNum(r.y)}</span></div>`).join("");
      tooltipEl.hidden = false;
      const tw = tooltipEl.offsetWidth, th = tooltipEl.offsetHeight;
      let left = m.x + 14, top = m.y + 14;
      if (left + tw > window.innerWidth - 8) left = m.x - tw - 14;
      if (top + th > window.innerHeight - 8) top = m.y - th - 14;
      tooltipEl.style.left = left + "px"; tooltipEl.style.top = top + "px";
    }
  }

  // ------------------------------------------------------------------ chess replay (minimal SAN resolver)
  // board: Array(64) of piece chars (upper = white) or null; square = rank*8 + file (a1 = 0)
  const FILES = "abcdefgh";
  const GLYPH = { K: "♔", Q: "♕", R: "♖", B: "♗", N: "♘", P: "♙", k: "♚", q: "♛", r: "♜", b: "♝", n: "♞", p: "♟" };
  const KNIGHT_D = [[1, 2], [2, 1], [2, -1], [1, -2], [-1, -2], [-2, -1], [-2, 1], [-1, 2]];
  const KING_D = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]];
  const BISHOP_D = [[1, 1], [1, -1], [-1, 1], [-1, -1]];
  const ROOK_D = [[1, 0], [-1, 0], [0, 1], [0, -1]];

  function startState() {
    const board = new Array(64).fill(null);
    const back = "RNBQKBNR";
    for (let f = 0; f < 8; f++) {
      board[f] = back[f]; board[8 + f] = "P"; board[48 + f] = "p"; board[56 + f] = back[f].toLowerCase();
    }
    return { board, turn: "w", castling: { K: true, Q: true, k: true, q: true }, ep: null };
  }
  const colorOf = (p) => (p === p.toUpperCase() ? "w" : "b");
  const sq = (f, r) => r * 8 + f;
  const inside = (f, r) => f >= 0 && f < 8 && r >= 0 && r < 8;

  function attacked(board, target, byColor) {
    const tf = target % 8, tr = Math.floor(target / 8);
    const enemy = (p, t) => p && colorOf(p) === byColor && p.toUpperCase() === t;
    for (const [df, dr] of KNIGHT_D) { const f = tf + df, r = tr + dr; if (inside(f, r) && enemy(board[sq(f, r)], "N")) return true; }
    for (const [df, dr] of KING_D) { const f = tf + df, r = tr + dr; if (inside(f, r) && enemy(board[sq(f, r)], "K")) return true; }
    const pr = byColor === "w" ? tr - 1 : tr + 1;
    for (const df of [-1, 1]) { const f = tf + df; if (inside(f, pr) && enemy(board[sq(f, pr)], "P")) return true; }
    const slide = (dirs, types) => {
      for (const [df, dr] of dirs) {
        let f = tf + df, r = tr + dr;
        while (inside(f, r)) {
          const p = board[sq(f, r)];
          if (p) { if (colorOf(p) === byColor && types.includes(p.toUpperCase())) return true; break; }
          f += df; r += dr;
        }
      }
      return false;
    };
    return slide(BISHOP_D, ["B", "Q"]) || slide(ROOK_D, ["R", "Q"]);
  }

  /** Pseudo-legal destination squares for the piece on `from` (castling & en passant included). */
  function targets(state, from) {
    const { board, turn } = state;
    const p = board[from], t = p.toUpperCase();
    const f0 = from % 8, r0 = Math.floor(from / 8);
    const out = [];
    const push = (f, r) => { if (!inside(f, r)) return false; const q = board[sq(f, r)]; if (q && colorOf(q) === turn) return false; out.push(sq(f, r)); return !q; };
    if (t === "P") {
      const dir = turn === "w" ? 1 : -1, start = turn === "w" ? 1 : 6;
      if (inside(f0, r0 + dir) && !board[sq(f0, r0 + dir)]) {
        out.push(sq(f0, r0 + dir));
        if (r0 === start && !board[sq(f0, r0 + 2 * dir)]) out.push(sq(f0, r0 + 2 * dir));
      }
      for (const df of [-1, 1]) {
        const f = f0 + df, r = r0 + dir;
        if (!inside(f, r)) continue;
        const q = board[sq(f, r)];
        if ((q && colorOf(q) !== turn) || sq(f, r) === state.ep) out.push(sq(f, r));
      }
    } else if (t === "N") { for (const [df, dr] of KNIGHT_D) push(f0 + df, r0 + dr); }
    else if (t === "K") {
      for (const [df, dr] of KING_D) push(f0 + df, r0 + dr);
      const rank = turn === "w" ? 0 : 7, enemy = turn === "w" ? "b" : "w";
      if (r0 === rank && f0 === 4 && !attacked(board, from, enemy)) {
        if (state.castling[turn === "w" ? "K" : "k"] && !board[sq(5, rank)] && !board[sq(6, rank)] &&
            !attacked(board, sq(5, rank), enemy)) out.push(sq(6, rank));
        if (state.castling[turn === "w" ? "Q" : "q"] && !board[sq(3, rank)] && !board[sq(2, rank)] && !board[sq(1, rank)] &&
            !attacked(board, sq(3, rank), enemy)) out.push(sq(2, rank));
      }
    } else {
      const dirs = t === "B" ? BISHOP_D : t === "R" ? ROOK_D : BISHOP_D.concat(ROOK_D);
      for (const [df, dr] of dirs) { let f = f0 + df, r = r0 + dr; while (push(f, r)) { f += df; r += dr; } }
    }
    return out;
  }

  function makeMove(state, from, to, promo) {
    const board = state.board.slice();
    const p = board[from], t = p.toUpperCase(), turn = state.turn;
    const castling = Object.assign({}, state.castling);
    let ep = null;
    board[to] = p; board[from] = null;
    if (t === "P") {
      const dir = turn === "w" ? 1 : -1;
      if (to === state.ep) board[to - 8 * dir] = null;
      if (Math.abs(to - from) === 16) ep = from + 8 * dir;
      if (promo) board[to] = turn === "w" ? promo.toUpperCase() : promo.toLowerCase();
    }
    if (t === "K") {
      castling[turn === "w" ? "K" : "k"] = false; castling[turn === "w" ? "Q" : "q"] = false;
      if (to - from === 2) { board[to - 1] = board[to + 1]; board[to + 1] = null; }
      if (from - to === 2) { board[to + 1] = board[to - 2]; board[to - 2] = null; }
    }
    for (const [s, k] of [[0, "Q"], [7, "K"], [56, "q"], [63, "k"]]) if (from === s || to === s) castling[k] = false;
    return { board, turn: turn === "w" ? "b" : "w", castling, ep };
  }

  function kingSquare(board, color) { return board.indexOf(color === "w" ? "K" : "k"); }
  function legalAfter(state, from, to, promo) {
    const next = makeMove(state, from, to, promo);
    const k = kingSquare(next.board, state.turn);
    return k < 0 || !attacked(next.board, k, next.turn);
  }

  /** Resolve a SAN string against a state -> {from, to, promo} or null. */
  function resolveSAN(state, san) {
    let s = san.replace(/[+#!?]+$/g, "").replace(/^\d+\.+/, "");
    if (!s) return null;
    const rank = state.turn === "w" ? 0 : 7;
    if (/^O-O(-O)?$|^0-0(-0)?$/.test(s)) {
      const from = sq(4, rank), to = sq(s.length > 3 ? 2 : 6, rank);
      return targets(state, from).includes(to) && legalAfter(state, from, to) ? { from, to } : null;
    }
    const m = /^([KQRBN])?([a-h])?([1-8])?x?([a-h][1-8])(?:=?([QRBN]))?$/.exec(s);
    if (!m) return null;
    const piece = m[1] || "P", disF = m[2], disR = m[3], to = sq(FILES.indexOf(m[4][0]), +m[4][1] - 1), promo = m[5];
    const cands = [];
    for (let from = 0; from < 64; from++) {
      const p = state.board[from];
      if (!p || colorOf(p) !== state.turn || p.toUpperCase() !== piece) continue;
      if (disF && from % 8 !== FILES.indexOf(disF)) continue;
      if (disR && Math.floor(from / 8) !== +disR - 1) continue;
      if (targets(state, from).includes(to) && legalAfter(state, from, to, promo)) cands.push(from);
    }
    if (cands.length !== 1) return cands.length ? { from: cands[0], to, promo } : null;
    return { from: cands[0], to, promo };
  }

  /** PGN -> {sans: [...], states: [...], last: [[from,to]...], error} */
  function parsePGN(pgn) {
    const headers = {};
    let body = pgn.replace(/\[(\w+)\s+"([^"]*)"\]\s*/g, (_, k, v) => { headers[k] = v; return ""; });
    body = body.replace(/\{[^}]*\}/g, " ").replace(/;[^\n]*/g, " ").replace(/\$\d+/g, " ");
    let depth = 0, flat = "";
    for (const ch of body) { if (ch === "(") depth++; else if (ch === ")") depth = Math.max(0, depth - 1); else if (!depth) flat += ch; }
    const tokens = flat.split(/\s+/).filter(Boolean);
    const sans = [], states = [startState()], last = [null];
    let error = null;
    for (const tok0 of tokens) {
      const tok = tok0.replace(/^\d+\.+/, "");
      if (!tok || /^(1-0|0-1|1\/2-1\/2|\*)$/.test(tok)) continue;
      const st = states[states.length - 1];
      const mv = resolveSAN(st, tok);
      if (!mv) { error = `cannot replay move ${sans.length + 1}: ${tok0}`; break; }
      sans.push(tok); states.push(makeMove(st, mv.from, mv.to, mv.promo)); last.push([mv.from, mv.to]);
    }
    return { headers, sans, states, last, error };
  }

  const boardEl = $("#board");
  function renderBoard(state, lastMove) {
    const S = 45, parts = [];
    for (let r = 7; r >= 0; r--) for (let f = 0; f < 8; f++) {
      const i = sq(f, r), x = f * S, y = (7 - r) * S;
      const hl = lastMove && (lastMove[0] === i || lastMove[1] === i);
      parts.push(`<rect class="sq ${(f + r) % 2 ? "light" : "dark"}" x="${x}" y="${y}" width="${S}" height="${S}"/>`);
      if (hl) parts.push(`<rect class="sq hl" x="${x}" y="${y}" width="${S}" height="${S}"/>`);
      if (f === 0) parts.push(`<text class="coord" x="${x + 2}" y="${y + 10}">${r + 1}</text>`);
      if (r === 0) parts.push(`<text class="coord" x="${x + S - 7}" y="${y + S - 3}">${FILES[f]}</text>`);
      const p = state.board[i];
      if (p) parts.push(`<text class="piece ${colorOf(p)}" x="${x + S / 2}" y="${y + S / 2 + 1}">${GLYPH[p]}</text>`);
    }
    boardEl.innerHTML = parts.join("");
  }

  const game = { parsed: null, ply: 0, playing: false, timer: 0, key: null };
  const gameSlider = $("#game-slider"), gamePlay = $("#game-play"), gamePly = $("#game-ply"), pgnEl = $("#pgn"), gameMeta = $("#game-meta");
  function showPly(ply) {
    const g = game.parsed; if (!g) return;
    game.ply = Math.max(0, Math.min(ply, g.states.length - 1));
    renderBoard(g.states[game.ply], g.last[game.ply]);
    gameSlider.value = game.ply; gamePly.textContent = `${game.ply} / ${g.states.length - 1}`;
    const html = g.sans.map((s, i) => `${i % 2 === 0 ? `${i / 2 + 1}. ` : ""}<span class="${i + 1 === game.ply ? "cur" : ""}">${escapeHtml(s)}</span>`).join(" ");
    pgnEl.innerHTML = (g.headerText ? escapeHtml(g.headerText) + "\n\n" : "") + html + (g.error ? `\n<span style="color:var(--red)">${escapeHtml(g.error)}</span>` : "");
    const cur = $(".cur", pgnEl); if (cur && game.playing) cur.scrollIntoView({ block: "nearest" });
  }
  function setPlaying(on) {
    game.playing = on; gamePlay.textContent = on ? "❚❚" : "▶";
    clearInterval(game.timer);
    if (on) game.timer = setInterval(() => {
      if (!game.parsed) return;
      if (game.ply >= game.parsed.states.length - 1) { setPlaying(false); return; }
      showPly(game.ply + 1);
    }, 650);
  }
  function loadGame(rec) {
    const key = `${rec.step}:${rec.t}`;
    if (game.key === key) return;
    game.key = key;
    const parsed = parsePGN(rec.pgn || "");
    parsed.headerText = Object.entries(parsed.headers).filter(([k]) => ["White", "Black", "Result", "Event"].includes(k)).map(([k, v]) => `${k}: ${v}`).join("  ");
    game.parsed = parsed;
    gameSlider.max = parsed.states.length - 1;
    gameMeta.textContent = `· step ${fmtInt(rec.step)} · ${rec.source || ""} · ${rec.result || ""} · ${rec.moves ?? parsed.sans.length} moves`;
    showPly(0);
    setPlaying(parsed.sans.length > 0);
  }
  gameSlider.addEventListener("input", () => { setPlaying(false); showPly(+gameSlider.value); });
  gamePlay.addEventListener("click", () => {
    if (game.parsed && game.ply >= game.parsed.states.length - 1) showPly(0);
    setPlaying(!game.playing);
  });
  renderBoard(startState(), null);

  // ------------------------------------------------------------------ neuron activity
  const actCanvas = $("#activity"), actCtx = actCanvas.getContext("2d");
  const actMeta = $("#activity-meta"), actNote = $("#activity-note");
  const activity = { positions: null, posByIdx: null, layoutKey: null, rec: null, sprites: null, lastFetch: 0 };
  const RAMP = [[42, 36, 18], [122, 90, 18], [240, 178, 50], [255, 231, 163]]; // sequential amber: dim → bright

  function rampColor(t) {
    t = Math.max(0, Math.min(1, t)) * (RAMP.length - 1);
    const i = Math.min(RAMP.length - 2, Math.floor(t)), u = t - i;
    const c = RAMP[i].map((v, k) => Math.round(v + (RAMP[i + 1][k] - v) * u));
    return `rgb(${c[0]},${c[1]},${c[2]})`;
  }
  function sprites() {
    if (activity.sprites) return activity.sprites;
    const out = [];
    for (let b = 0; b < 16; b++) {
      const size = 24, c = document.createElement("canvas"); c.width = c.height = size;
      const g = c.getContext("2d"), t = b / 15;
      const grad = g.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
      grad.addColorStop(0, rampColor(t)); grad.addColorStop(0.25, rampColor(t));
      grad.addColorStop(1, "rgba(240,178,50,0)");
      g.fillStyle = grad; g.fillRect(0, 0, size, size);
      out.push(c);
    }
    activity.sprites = out;
    return out;
  }

  function drawActivity() {
    const dpr = window.devicePixelRatio || 1;
    const W = actCanvas.clientWidth, H = actCanvas.clientHeight;
    if (!W || !H) return;
    if (actCanvas.width !== Math.round(W * dpr) || actCanvas.height !== Math.round(H * dpr)) {
      actCanvas.width = Math.round(W * dpr); actCanvas.height = Math.round(H * dpr);
    }
    const ctx = actCtx;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const rec = activity.rec;
    if (!rec || !rec.values || !rec.values.length) return;
    const idx = rec.neuron_idx || rec.values.map((_, i) => i);
    const n = rec.values.length;

    // layout: connectome positions (front view: x, y) when available, else a grid
    const usePos = activity.posByIdx && idx.every((i) => activity.posByIdx.has(i));
    const cols = Math.ceil(Math.sqrt(n)), rows = Math.ceil(n / cols);
    const pad = 12;
    const pts = new Array(n);
    for (let k = 0; k < n; k++) {
      let x, y;
      const p = usePos ? activity.posByIdx.get(idx[k]) : null;
      if (p) { x = pad + p[0] * (W - 2 * pad); y = pad + p[1] * (H - 2 * pad); } // FlyWire y grows ventrally: no flip
      else { x = pad + ((k % cols) + 0.5) / cols * (W - 2 * pad); y = pad + (Math.floor(k / cols) + 0.5) / rows * (H - 2 * pad); }
      pts[k] = [x, y];
    }
    // colour by robust-normalised value (5th–95th percentile)
    const sorted = rec.values.filter(Number.isFinite).slice().sort((a, b) => a - b);
    const lo = sorted[Math.floor(sorted.length * 0.05)] ?? 0, hi = sorted[Math.floor(sorted.length * 0.95)] ?? 1;
    const span = hi - lo || 1;
    const sp = sprites();
    const size = Math.max(4, Math.min(14, Math.sqrt((W * H) / n) * 0.9));
    ctx.globalCompositeOperation = "lighter";
    for (let k = 0; k < n; k++) {
      const v = rec.values[k];
      if (!Number.isFinite(v)) continue;
      const t = Math.max(0, Math.min(1, (v - lo) / span));
      ctx.globalAlpha = 0.25 + 0.75 * t;
      const s = size * (0.7 + 0.8 * t);
      ctx.drawImage(sp[Math.round(t * 15)], pts[k][0] - s / 2, pts[k][1] - s / 2, s, s);
    }
    ctx.globalCompositeOperation = "source-over"; ctx.globalAlpha = 1;
    const active = sorted.length ? sorted.filter((v) => v > 1e-6).length / sorted.length : 0;
    actMeta.textContent = `· step ${fmtInt(rec.step)} · ${n} neurons`;
    actNote.textContent = `${usePos ? "connectome positions (frontal view)" : "grid layout (no positions)"} · ${(active * 100).toFixed(0)}% active · max ${fmtNum(sorted[sorted.length - 1])} · mean ${fmtNum(sorted.reduce((a, b) => a + b, 0) / (sorted.length || 1))}`;
  }
  new ResizeObserver(() => drawActivity()).observe(actCanvas);

  async function fetchPositions(run) {
    activity.lastFetch = Date.now();
    const gen = state.gen;
    try {
      const r = await fetch(`/api/run/${encodeURIComponent(run)}/positions`);
      const j = await r.json();
      if (gen !== state.gen || run !== state.run) return; // the run changed while the request was in flight
      if (!j.available) { activity.posByIdx = null; drawActivity(); return; }
      const map = new Map();
      j.neuron_idx.forEach((i, k) => { const p = j.positions[k]; if (p) map.set(i, p); });
      activity.posByIdx = map;
      drawActivity();
    } catch (e) { activity.posByIdx = null; }
  }

  // ------------------------------------------------------------------ log tail
  const logEl = $("#log"), logFollow = $("#log-follow");
  const LOG_MAX = 300;
  function logLine(rec) {
    let msg;
    switch (rec.kind) {
      case "status": msg = `${rec.message || ""}${rec.stage ? `  [${rec.stage}]` : ""}${rec.eta_s != null ? `  eta ${fmtDuration(rec.eta_s)}` : ""}`; break;
      case "eval": msg = `val_loss ${fmtNum(rec.val_loss)}  top1 ${fmtNum(rec.val_top1)}  top3 ${fmtNum(rec.val_top3)}  value_mse ${fmtNum(rec.val_value_mse)}`; break;
      case "elo": msg = `vs ${rec.opponent}: ${rec.elo_estimate != null ? Math.round(rec.elo_estimate) : "–"} Elo  (${rec.wins}W ${rec.draws}D ${rec.losses}L / ${rec.games})`; break;
      case "game": msg = `${rec.source || ""} game ${rec.result || ""} in ${rec.moves} moves`; break;
      case "error": msg = rec.message || JSON.stringify(rec); break;
      default: return;
    }
    const div = document.createElement("div");
    div.className = `ln ${rec.kind}`;
    div.innerHTML = `<span class="ts">${fmtTime(rec.t)}</span> <span class="kind">${rec.kind.padEnd(6)}</span> <span class="step">${fmtInt(rec.step)}</span>  ${escapeHtml(msg)}`;
    logEl.appendChild(div);
    while (logEl.childElementCount > LOG_MAX) logEl.removeChild(logEl.firstChild);
    if (logFollow.checked) logEl.scrollTop = logEl.scrollHeight;
  }

  // ------------------------------------------------------------------ charts
  const chartEls = {};
  document.querySelectorAll(".chart[data-chart]").forEach((el) => { chartEls[el.dataset.chart] = el; });
  const charts = {
    loss: new Chart(chartEls.loss, { title: "loss", series: [
      { key: "loss", label: "total", color: SERIES[0] }, { key: "policy_loss", label: "policy", color: SERIES[1] },
      { key: "value_loss", label: "value", color: SERIES[2] }, { key: "val_loss", label: "eval total", color: SERIES[0], dash: true }] }),
    acc: new Chart(chartEls.acc, { title: "move accuracy", zeroBased: true, series: [
      { key: "top1", label: "top-1", color: SERIES[0] }, { key: "top3", label: "top-3", color: SERIES[1] },
      { key: "val_top1", label: "eval top-1", color: SERIES[0], dash: true }, { key: "val_top3", label: "eval top-3", color: SERIES[1], dash: true }] }),
    lr: new Chart(chartEls.lr, { title: "learning rate", series: [{ key: "lr", label: "lr", color: SERIES[3] }] }),
    throughput: new Chart(chartEls.throughput, { title: "throughput", zeroBased: true, series: [{ key: "pos_per_sec", label: "positions / s", color: SERIES[2] }] }),
    elo: new Chart(chartEls.elo, { title: "Elo estimate", series: [], xExtent: () => [0, totalSteps()] }),
    value: new Chart(chartEls.value, { title: "value head", series: [
      { key: "value_loss", label: "train mse", color: SERIES[2] }, { key: "val_value_mse", label: "eval mse", color: SERIES[2], dash: true }] }),
  };

  // ------------------------------------------------------------------ store
  // `offset` is the metrics.jsonl byte cursor after the last record received (from the server's init / cursor
  // frames); a reconnect to the same run resumes from it instead of re-fetching (and resetting) everything.
  const state = { run: null, runInfo: null, lastTrain: null, lastStatus: null, gen: 0, ws: null, retry: 1000, lastGame: null, lastActivity: null, offset: null };
  const st = {
    stage: $("#st-stage"), step: $("#st-step"), eta: $("#st-eta"), gpu: $("#st-gpu"), pps: $("#st-pps"), epoch: $("#st-epoch"),
    conn: $("#st-conn"), connText: $("#st-conn-text"), bar: $("#progress-bar"), runInfo: $("#run-info"),
  };

  function setConn(stateName, text) { st.conn.dataset.state = stateName; st.connText.textContent = text || stateName; }

  function resetAll() {
    for (const c of Object.values(charts)) c.clear();
    logEl.innerHTML = "";
    state.lastTrain = null; state.lastStatus = null; state.lastGame = null; state.lastActivity = null; state.offset = null;
    activity.rec = null; activity.posByIdx = null; drawActivity();
    game.key = null; game.parsed = null; setPlaying(false); renderBoard(startState(), null);
    gameSlider.max = 0; gamePly.textContent = "0 / 0"; pgnEl.textContent = "no game yet — the fly has not played a sample game"; gameMeta.textContent = "";
    actMeta.textContent = ""; actNote.textContent = "waiting for an activity sample…";
    updateStatus();
  }

  function ingest(rec, live) {
    const step = rec.step;
    switch (rec.kind) {
      case "train":
        for (const k of ["loss", "policy_loss", "value_loss"]) charts.loss.append(k, step, rec[k]);
        charts.acc.append("top1", step, rec.top1); charts.acc.append("top3", step, rec.top3);
        charts.lr.append("lr", step, rec.lr);
        charts.throughput.append("pos_per_sec", step, rec.pos_per_sec);
        charts.value.append("value_loss", step, rec.value_loss);
        state.lastTrain = rec;
        break;
      case "eval":
        charts.loss.append("val_loss", step, rec.val_loss);
        charts.acc.append("val_top1", step, rec.val_top1); charts.acc.append("val_top3", step, rec.val_top3);
        charts.value.append("val_value_mse", step, rec.val_value_mse);
        logLine(rec);
        break;
      case "elo": {
        const key = `elo:${rec.opponent}`;
        charts.elo.append(key, step, rec.elo_estimate, { label: `vs ${rec.opponent}`, points: true });
        logLine(rec);
        break;
      }
      case "game": state.lastGame = rec; logLine(rec); break;
      case "activity": state.lastActivity = rec; break;
      case "status": state.lastStatus = rec; logLine(rec); break;
      default: if (rec.kind === "error") logLine(rec);
    }
    if (live) applyLatest();
  }

  function applyLatest() {
    if (state.lastGame) loadGame(state.lastGame);
    if (state.lastActivity && state.lastActivity !== activity.rec) {
      activity.rec = state.lastActivity;
      const idx = activity.rec.neuron_idx;
      const covered = !idx || (activity.posByIdx && idx.every((i) => activity.posByIdx.has(i)));
      if (!covered && Date.now() - activity.lastFetch > 15000) fetchPositions(state.run);
      drawActivity();
    }
    updateStatus();
  }

  /** Total steps of the current run (latest status record, else run.json config), or null. */
  function totalSteps() {
    const s = state.lastStatus, info = state.runInfo || {};
    return s && s.total_steps ? s.total_steps : (info.config && (info.config.steps || info.config.total_steps)) || null;
  }

  function updateStatus() {
    const t = state.lastTrain, s = state.lastStatus, info = state.runInfo || {};
    const stage = (s && s.stage) || (t && t.stage) || (info.config && info.config.stage) || "–";
    const step = t ? t.step : s ? s.step : null;
    const total = totalSteps();
    st.stage.textContent = stage;
    st.step.textContent = step === null ? "–" : `${fmtInt(step)}${total ? ` / ${fmtInt(total)}` : ""}`;
    st.eta.textContent = s && s.eta_s != null ? fmtDuration(s.eta_s) : "–";
    st.gpu.textContent = t && t.gpu_mem_gb != null ? `${t.gpu_mem_gb.toFixed(2)} GB` : "–";
    st.pps.textContent = t && t.pos_per_sec != null ? fmtInt(t.pos_per_sec) : "–";
    st.epoch.textContent = t && t.epoch != null ? (Number.isInteger(t.epoch) ? t.epoch : t.epoch.toFixed(2)) : "–";
    st.bar.style.width = total && step !== null ? `${Math.min(100, (100 * step) / total).toFixed(1)}%` : "0%";
    const gm = info.graph_meta || {};
    const parts = [];
    if (state.run) parts.push(`run ${state.run}`);
    if (info.started_at) parts.push(`started ${new Date(info.started_at * 1000).toLocaleString()}`);
    if (gm.n || gm.neurons) parts.push(`${fmtInt(gm.n || gm.neurons)} neurons`);
    if (gm.nnz || gm.synapses) parts.push(`${fmtInt(gm.nnz || gm.synapses)} connections`);
    if (info.graph_path) parts.push(String(info.graph_path).split("/").pop());
    st.runInfo.textContent = parts.join(" · ") || "–";
  }

  // ------------------------------------------------------------------ websocket
  /**
   * Open the websocket for `run`. `after` (a byte cursor from a previous connection to the *same* run) asks the
   * server to resume the tail from there: the server answers `{"kind":"resume"}` and the accumulated history is
   * kept; without it (or if the cursor is stale) the server sends a full `init` and the UI is rebuilt from it.
   * Frames: init · resume · cursor (`{"offset"}` after each batch of live records) · raw metrics records.
   */
  function connect(run, after) {
    const gen = ++state.gen;
    if (state.ws) { try { state.ws.close(); } catch (e) { /* ignore */ } state.ws = null; }
    if (!run) { setConn("idle", "idle"); return; }
    setConn("connecting", "connecting");
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const query = after !== null && after !== undefined ? `?after=${encodeURIComponent(after)}` : "";
    const ws = new WebSocket(`${proto}://${location.host}/ws/${encodeURIComponent(run)}${query}`);
    state.ws = ws;
    ws.onopen = () => { if (gen !== state.gen) return; state.retry = 1000; setConn("waiting", "waiting for metrics"); };
    ws.onmessage = (ev) => {
      if (gen !== state.gen) return;
      let rec; try { rec = JSON.parse(ev.data); } catch (e) { return; }
      switch (rec.kind) {
        case "init":
          resetAll();
          state.runInfo = rec.run || {};
          for (const m of rec.metrics || []) ingest(m, false);
          if (typeof rec.offset === "number") state.offset = rec.offset;
          applyLatest();
          setConn("live", `live · ${fmtInt(Object.values(rec.counts || {}).reduce((a, b) => a + b, 0) || (rec.metrics || []).length)} records`);
          fetchPositions(run);
          break;
        case "resume":
          if (typeof rec.offset === "number") state.offset = rec.offset;
          setConn("live", "live · resumed");
          break;
        case "cursor":
          if (typeof rec.offset === "number") state.offset = rec.offset;
          break;
        default:
          ingest(rec, true);
          setConn("live", `live · ${fmtTime(rec.t)}`);
      }
    };
    ws.onclose = () => {
      if (gen !== state.gen) return;
      setConn("offline", `reconnecting in ${Math.round(state.retry / 1000)}s`);
      setTimeout(() => { if (gen === state.gen) connect(run, run === state.run ? state.offset : null); }, state.retry);
      state.retry = Math.min(10000, state.retry * 2);
    };
    ws.onerror = () => { /* onclose follows */ };
  }

  function openRun(run) {
    if (!run) return;
    state.run = run;
    if (![...runSelect.options].some((o) => o.value === run)) {
      runSelect.insertAdjacentHTML("afterbegin", `<option value="${escapeHtml(run)}">${escapeHtml(run)} · (waiting)</option>`);
    }
    runSelect.value = run;
    location.hash = run;
    document.title = `fly-chess · ${run}`;
    resetAll();
    connect(run);
  }

  // ------------------------------------------------------------------ run selector
  const runSelect = $("#run-select");
  async function refreshRuns() {
    let runs = [];
    try { runs = await (await fetch("/api/runs")).json(); } catch (e) { return runs; }
    const current = state.run;
    runSelect.innerHTML = runs.length ? runs.map((r) =>
      `<option value="${escapeHtml(r.name)}">${escapeHtml(r.name)}${r.stage ? ` · ${escapeHtml(r.stage)}` : ""}${r.last_step != null ? ` · ${fmtInt(r.last_step)}` : ""}</option>`).join("")
      : `<option value="">— no runs yet —</option>`;
    if (current && ![...runSelect.options].some((o) => o.value === current)) {
      runSelect.insertAdjacentHTML("afterbegin", `<option value="${escapeHtml(current)}">${escapeHtml(current)} · (waiting)</option>`);
    }
    if (current) runSelect.value = current;
    return runs;
  }
  runSelect.addEventListener("change", () => openRun(runSelect.value));

  async function boot() {
    const runs = await refreshRuns();
    let def = null;
    try { def = (await (await fetch("/api/default")).json()).run; } catch (e) { /* ignore */ }
    const fromHash = decodeURIComponent(location.hash.slice(1));
    const pick = fromHash || def || (runs[0] && runs[0].name);
    if (pick) openRun(pick); else setConn("idle", "no runs — start `fly train`");
    setInterval(async () => {
      const rs = await refreshRuns();
      if (!state.run && rs.length) openRun(rs[0].name);
    }, 20000);
  }
  window.addEventListener("hashchange", () => {
    const h = decodeURIComponent(location.hash.slice(1));
    if (h && h !== state.run) openRun(h);
  });
  boot();
})();
