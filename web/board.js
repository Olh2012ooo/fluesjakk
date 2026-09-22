// board.js — avhengighetsfritt sjakkbrett i SVG.
//
//   const board = new Board(el, { onMove(from, to, promotion), orientation: 'white' });
//   board.setPosition(fen);                 // animerer brikkene som flytter seg
//   board.setLegal(chess.moves({verbose:true}));
//   board.highlight({ lastMove: ['e2', 'e4'], check: 'e8' });
//   board.setMovable('white' | 'black' | null);
//   board.cancelPromotion();                // lukk forvandlingsvelgeren
//   board.setVisning({ lovlige: true, siste: true, koordinater: true });
//
// Tastatur: brettet kan fokuseres; piltastene flytter en markør, Enter/Mellomrom velger en brikke
// eller slipper den på markøren, Escape avbryter valget eller forvandlingsvelgeren.
//
// Brikkene er egne, kantete SVG-former (se BRIKKER). Fargene kommer fra CSS-variabler på verten
// (--sq-light, --sq-dark, --piece-w, --piece-w-ink, --piece-b, --piece-b-ink, --hl-*).

const FILES = 'abcdefgh';
const S = 100;                        // rutebredde i viewBox-enheter

// Egen tegning: kantete, tydelige silhuetter, 100×100 ruter med grunnlinje på y=90.
// `.ink`-delene tar omrissfargen. Alle har samme sokkel nederst, så de står støtt ved siden av hverandre.
const SOKKEL = '<path d="M26 78l4 10h40l4-10Z"/>';
export const BRIKKER = {
  // bonde: kule på skråstilt krage
  p: `<circle cx="50" cy="25" r="11.5"/><path d="M41 38h18l3 10H38Z"/><path d="M38 48h24l6 30H32Z"/>${SOKKEL}`,

  // tårn: tindet krone over en trappet body
  r: `<path d="M26 14h11v9h9v-9h8v9h9v-9h11v25H26Z"/><path d="M30 39h40l-3 13H33Z"/><path d="M33 52h34l-2 26H35Z"/>${SOKKEL}`,

  // springer: kantet hestehode med øye og man
  n: `<path d="M33 78V68c0-14 6-23 16-28l-10-3c-9-3-10-13-3-16l11-3c3-6 6-9 10-11l3-8 3 8 7-6 1 16c9 9 10 30 6 61Z"/><circle cx="56" cy="30" r="2.8" class="ink"/><path d="M64 34c3 8 4 16 3 24" class="ink" fill="none"/>${SOKKEL}`,

  // løper: dråpeformet mitra med skråsnitt
  b: `<path d="M50 11c5 9 14 16 14 27 0 10-6 16-14 16s-14-6-14-16c0-11 9-18 14-27Z"/><path d="M49 20 42 36" class="ink" fill="none"/><path d="M39 52h22l2 9H37Z"/><path d="M37 61h26l4 17H33Z"/>${SOKKEL}`,

  // dronning: takket krone med brei skjørt
  q: `<path d="M22 21l5 10 7-12 8 14 8-16 8 16 8-14 7 12 5-10 5 27H17Z"/><path d="M19 48h62l-5 13H24Z"/><path d="M24 61h52l-3 17H27Z"/>${SOKKEL}`,

  // konge: kors over krone og kappe
  k: `<path d="M46 5h8v7h7v8h-7v7h-8v-7h-7v-8h7Z"/><path d="M31 31c7-10 31-10 38 0l3 33H28Z"/><path d="M30 64h40l-2 14H32Z"/>${SOKKEL}`,
};

// Brikkenavn på norsk, til skjermlesere.
const NAVN = { p: 'bonde', n: 'springer', b: 'løper', r: 'tårn', q: 'dronning', k: 'konge' };

const NS = 'http://www.w3.org/2000/svg';
function svgEl(tag, attrs = {}, parent) {
  const el = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  if (parent) parent.appendChild(el);
  return el;
}

/** Les brikkeplasseringen i en FEN inn i et Map fra rute til {type, color}. */
export function parseFenPieces(fen) {
  const map = new Map();
  const rows = fen.split(' ')[0].split('/');
  for (let r = 0; r < 8; r++) {
    let file = 0;
    for (const ch of rows[r]) {
      if (ch >= '1' && ch <= '8') { file += +ch; continue; }
      const color = ch === ch.toUpperCase() ? 'w' : 'b';
      map.set(FILES[file] + (8 - r), { type: ch.toLowerCase(), color });
      file++;
    }
  }
  return map;
}

export class Board {
  /**
   * @param {HTMLElement} el vertselement (styres av CSS; brettet fyller det og holder kvadratisk form)
   * @param {{onMove?:(from:string,to:string,promotion?:string)=>void, orientation?:'white'|'black', coordinates?:boolean, onSelect?:(sq:string|null)=>void}} opts
   */
  constructor(el, opts = {}) {
    this.el = el;
    this.onMove = opts.onMove || (() => {});
    this.onSelect = opts.onSelect || (() => {});
    this.orientation = opts.orientation || 'white';
    this.coordinates = opts.coordinates !== false;
    this.legal = [];
    this.movable = null;           // 'w' | 'b' | null
    this.selected = null;
    this.pieces = new Map();       // rute -> {type, color, el}
    this.marks = { lastMove: null, check: null };
    this.drag = null;
    this.cursor = null;            // tastaturmarkør (vises bare når tastaturet styrer)
    this._kb = false;
    this.visning = { lovlige: true, siste: true };
    this._build();
    this._bind();
  }

  // ------------------------------------------------------------------ oppbygging
  _build() {
    this.el.classList.add('cb-host');
    const svg = svgEl('svg', {
      viewBox: `0 0 ${8 * S} ${8 * S}`, class: 'cb', role: 'application', tabindex: 0,
      'aria-label': 'Sjakkbrett. Piltastene flytter markøren, Enter velger eller slipper en brikke, Escape avbryter.',
    });
    this.svg = svg;
    this.gSquares = svgEl('g', { class: 'cb-squares' }, svg);
    this.gMarks = svgEl('g', { class: 'cb-marks' }, svg);
    this.gCoords = svgEl('g', { class: 'cb-coords' }, svg);
    this.gPieces = svgEl('g', { class: 'cb-pieces' }, svg);
    this.gDots = svgEl('g', { class: 'cb-dots' }, svg);
    this.gPromo = svgEl('g', { class: 'cb-promo' }, svg);
    this.squareEls = {};
    for (let r = 0; r < 8; r++) for (let f = 0; f < 8; f++) {
      const sq = FILES[f] + (r + 1);
      const rect = svgEl('rect', { width: S, height: S, class: (f + r) % 2 ? 'cb-light' : 'cb-dark', 'data-sq': sq }, this.gSquares);
      this.squareEls[sq] = rect;
    }
    this.el.replaceChildren(svg);
    this._layoutSquares();
  }

  _xy(sq) {
    const f = FILES.indexOf(sq[0]), r = +sq[1] - 1;
    const white = this.orientation === 'white';
    return { x: (white ? f : 7 - f) * S, y: (white ? 7 - r : r) * S };
  }

  _sqAt(px, py) {
    const pt = this.svg.createSVGPoint();
    pt.x = px; pt.y = py;
    const p = pt.matrixTransform(this.svg.getScreenCTM().inverse());
    const col = Math.floor(p.x / S), row = Math.floor(p.y / S);
    if (col < 0 || col > 7 || row < 0 || row > 7) return null;
    const white = this.orientation === 'white';
    const f = white ? col : 7 - col, r = white ? 7 - row : row;
    return { sq: FILES[f] + (r + 1), x: p.x, y: p.y };
  }

  _layoutSquares() {
    for (const [sq, rect] of Object.entries(this.squareEls)) {
      const { x, y } = this._xy(sq);
      rect.setAttribute('x', x); rect.setAttribute('y', y);
    }
    this.gCoords.replaceChildren();
    if (this.coordinates) {
      const white = this.orientation === 'white';
      for (let i = 0; i < 8; i++) {
        const fileIdx = white ? i : 7 - i, rankIdx = white ? 7 - i : i;         // kolonne i / rad i
        const bottomRank = white ? 0 : 7, leftFile = white ? 0 : 7;
        const light = (f, r) => ((f + r) % 2 ? 'on-light' : 'on-dark');
        svgEl('text', { x: i * S + S - 6, y: 8 * S - 6, class: 'cb-coord ' + light(fileIdx, bottomRank), 'text-anchor': 'end' }, this.gCoords).textContent = FILES[fileIdx];
        svgEl('text', { x: 6, y: i * S + 18, class: 'cb-coord ' + light(leftFile, rankIdx) }, this.gCoords).textContent = rankIdx + 1;
      }
    }
    for (const [sq, p] of this.pieces) this._place(p.el, sq, false);
    this._renderMarks();
    this._renderDots();
  }

  _makePiece(type, color) {
    const g = svgEl('g', { class: `cb-piece cb-${color}`, 'data-type': type, 'data-color': color });
    g.innerHTML = BRIKKER[type];
    this.gPieces.appendChild(g);
    return g;
  }

  _place(el, sq, animate = true) {
    const { x, y } = this._xy(sq);
    el.style.transition = animate ? '' : 'none';
    el.style.transform = `translate(${x}px, ${y}px)`;
    if (!animate) { void el.getBBox?.(); el.style.transition = ''; }
  }

  // ------------------------------------------------------------------ offentlig API
  /** Sett orientering ('white' | 'black') og legg ut på nytt. */
  setOrientation(color) {
    this._closePromotion();
    this.orientation = color;
    this._layoutSquares();
  }

  flip() { this.setOrientation(this.orientation === 'white' ? 'black' : 'white'); }

  /** Hvilken side brukeren kan flytte; null låser brettet. */
  setMovable(color) {
    this._closePromotion();          // velgeren gir bare mening for stillingen den ble åpnet i
    this.movable = color === 'white' ? 'w' : color === 'black' ? 'b' : color || null;
    if (!this.movable) this._select(null);
    this.el.classList.toggle('cb-locked', !this.movable);
  }

  /** Lovlige trekk som chess.js-objekter ({from, to, promotion?}). */
  setLegal(list) {
    this.legal = list || [];
    this._renderDots();
  }

  /** Slå hjelpemidler av og på (innstillinger). */
  setVisning({ lovlige, siste, koordinater } = {}) {
    if (lovlige !== undefined) { this.visning.lovlige = !!lovlige; this._renderDots(); }
    if (siste !== undefined) { this.visning.siste = !!siste; this._renderMarks(); }
    if (koordinater !== undefined && !!koordinater !== this.coordinates) {
      this.coordinates = !!koordinater;
      this._layoutSquares();
    }
  }

  /** Oppdater stillingen fra en FEN og animer brikkene som flyttet seg. */
  setPosition(fen, { animate = true } = {}) {
    this._closePromotion();
    const next = parseFenPieces(fen);
    const removed = [];
    for (const [sq, p] of this.pieces) {
      const n = next.get(sq);
      if (n && n.type === p.type && n.color === p.color) next.delete(sq);
      else removed.push([sq, p]);
    }
    for (const [sq, p] of removed) this.pieces.delete(sq);
    for (const [sq, n] of next) {
      // gjenbruk en forsvunnet brikke av samme slag (den nærmeste) så den glir i stedet for å poppe
      let best = -1, bestD = Infinity;
      removed.forEach(([osq, op], i) => {
        if (op.type !== n.type || op.color !== n.color) return;
        const d = Math.abs(FILES.indexOf(osq[0]) - FILES.indexOf(sq[0])) + Math.abs(+osq[1] - +sq[1]);
        if (d < bestD) { bestD = d; best = i; }
      });
      let el;
      if (best >= 0) { el = removed[best][1].el; removed.splice(best, 1); this.gPieces.appendChild(el); }
      else { el = this._makePiece(n.type, n.color); if (animate) el.classList.add('cb-appear'); }
      this.pieces.set(sq, { type: n.type, color: n.color, el });
      this._place(el, sq, animate && best >= 0);
    }
    for (const [, p] of removed) {
      p.el.classList.add('cb-vanish');
      const el = p.el;
      setTimeout(() => el.remove(), animate ? 160 : 0);
    }
    this._select(null);
  }

  /** @param {{lastMove?: [string,string]|null, check?: string|null}} marks */
  highlight(marks) {
    this.marks = { ...this.marks, ...marks };
    this._renderMarks();
  }

  /** Lukk forvandlingsvelgeren hvis den er åpen (trekket blir forkastet). */
  cancelPromotion() { this._closePromotion(); }

  /** Rutene brikkene står på (til tester og delebildet). */
  position() {
    const out = {};
    for (const [sq, p] of this.pieces) out[sq] = p.color + p.type;
    return out;
  }

  // ------------------------------------------------------------------ tegning
  _renderMarks() {
    this.gMarks.replaceChildren();
    const { lastMove, check } = this.marks;
    if (lastMove && this.visning.siste) for (const sq of lastMove) { const { x, y } = this._xy(sq); svgEl('rect', { x, y, width: S, height: S, class: 'cb-last' }, this.gMarks); }
    if (check) { const { x, y } = this._xy(check); svgEl('rect', { x, y, width: S, height: S, class: 'cb-check' }, this.gMarks); }
    if (this.selected) { const { x, y } = this._xy(this.selected); svgEl('rect', { x, y, width: S, height: S, class: 'cb-selected' }, this.gMarks); }
    if (this.cursor && this._kb) { const { x, y } = this._xy(this.cursor); svgEl('rect', { x, y, width: S, height: S, class: 'cb-cursor' }, this.gMarks); }
  }

  _renderDots() {
    this.gDots.replaceChildren();
    if (!this.selected || !this.visning.lovlige) return;
    const seen = new Set();
    for (const m of this.legal) {
      if (m.from !== this.selected || seen.has(m.to)) continue;
      seen.add(m.to);
      const { x, y } = this._xy(m.to);
      const capture = this.pieces.has(m.to) || /e/.test(m.flags || '');
      if (capture) svgEl('circle', { cx: x + S / 2, cy: y + S / 2, r: S * 0.44, class: 'cb-dot cb-capture' }, this.gDots);
      else svgEl('circle', { cx: x + S / 2, cy: y + S / 2, r: S * 0.15, class: 'cb-dot' }, this.gDots);
    }
  }

  _select(sq) {
    this.selected = sq;
    this._renderMarks();
    this._renderDots();
    this.onSelect(sq);
  }

  _tryMove(from, to) {
    const options = this.legal.filter((m) => m.from === from && m.to === to);
    if (options.length === 0) return false;
    if (options.some((m) => m.promotion)) {
      this._askPromotion(from, to, options[0].color || this.pieces.get(from)?.color);
      return true;
    }
    this._select(null);
    this.onMove(from, to, undefined);
    return true;
  }

  _askPromotion(from, to, color) {
    this.gPromo.replaceChildren();
    const { x, y } = this._xy(to);
    const down = y === 0 ? 1 : -1;               // legg brikkene vekk fra kanten
    const choices = ['q', 'n', 'r', 'b'];
    svgEl('rect', { x: 0, y: 0, width: 8 * S, height: 8 * S, class: 'cb-promo-veil' }, this.gPromo)
      .addEventListener('pointerdown', (e) => { e.stopPropagation(); this._closePromotion(); });
    choices.forEach((pc, i) => {
      const yy = y + down * i * S;
      const g = svgEl('g', { class: `cb-promo-choice cb-${color}`, tabindex: 0, role: 'button', 'aria-label': `forvandle til ${NAVN[pc]}` }, this.gPromo);
      svgEl('rect', { x, y: yy, width: S, height: S, rx: 10 }, g);
      const p = svgEl('g', { class: `cb-piece cb-${color} cb-static` }, g);
      p.innerHTML = BRIKKER[pc];
      p.style.transform = `translate(${x}px, ${yy}px)`;
      const pick = (e) => { e.stopPropagation(); e.preventDefault(); this._closePromotion(); this._select(null); this.onMove(from, to, pc); };
      g.addEventListener('pointerdown', pick);
      g.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') pick(e); });
    });
    if (this._kb) this.gPromo.querySelector('.cb-promo-choice')?.focus?.();
  }

  _closePromotion() {
    if (!this.gPromo.childElementCount) return;
    const hadFocus = typeof document !== 'undefined' && this.gPromo.contains(document.activeElement);
    this.gPromo.replaceChildren();
    if (hadFocus) this.svg.focus?.();          // hold tastaturbrukere på brettet
  }

  // ------------------------------------------------------------------ interaksjon
  _bind() {
    const svg = this.svg;
    svg.addEventListener('pointerdown', (e) => this._down(e));
    svg.addEventListener('pointermove', (e) => this._move(e));
    svg.addEventListener('pointerup', (e) => this._up(e));
    svg.addEventListener('pointercancel', () => this._cancelDrag());
    svg.addEventListener('contextmenu', (e) => e.preventDefault());
    svg.addEventListener('keydown', (e) => this._key(e));
    svg.addEventListener('blur', () => this._setKb(false));
  }

  _setKb(on) {
    if (this._kb === on) return;
    this._kb = on;
    this._renderMarks();
  }

  /** Flytt tastaturmarkøren (dx, dy) ruter sett fra tilskueren. */
  _moveCursor(dx, dy) {
    const white = this.orientation === 'white';
    let f, r;
    if (this.cursor) { f = FILES.indexOf(this.cursor[0]); r = +this.cursor[1] - 1; }
    else { f = white ? 4 : 3; r = white ? 0 : 7; }                 // start på kongen nærmest spilleren
    f += white ? dx : -dx; r += white ? -dy : dy;
    if (f < 0 || f > 7 || r < 0 || r > 7) return;
    this.cursor = FILES[f] + (r + 1);
  }

  _key(e) {
    if (this.gPromo.contains(e.target)) {                           // inne i forvandlingsvelgeren
      if (e.key === 'Escape') { e.preventDefault(); this._closePromotion(); }
      return;
    }
    if (e.key === 'Escape') {
      e.preventDefault();
      if (this.gPromo.childElementCount) this._closePromotion();
      else if (this.selected) this._select(null);
      return;
    }
    const ARROWS = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };
    if (ARROWS[e.key]) {
      e.preventDefault();
      if (this.gPromo.childElementCount) return;
      this._setKb(true);
      this._moveCursor(...ARROWS[e.key]);
      this._renderMarks();
      return;
    }
    if (e.key !== 'Enter' && e.key !== ' ') return;
    e.preventDefault();
    if (!this.movable || this.gPromo.childElementCount) return;
    this._setKb(true);
    if (!this.cursor) { this._moveCursor(0, 0); this._renderMarks(); return; }
    const sq = this.cursor;
    if (this.selected && this.selected !== sq && this._tryMove(this.selected, sq)) return;
    const piece = this.pieces.get(sq);
    if (piece && piece.color === this.movable && this.selected !== sq) this._select(sq);
    else this._select(null);
  }

  _down(e) {
    if (e.button !== 0 && e.pointerType === 'mouse') return;
    this._setKb(false);
    if (!this.movable || this.gPromo.childElementCount) return;
    const hit = this._sqAt(e.clientX, e.clientY);
    if (!hit) return;
    const piece = this.pieces.get(hit.sq);
    if (this.selected && this.selected !== hit.sq && this._tryMove(this.selected, hit.sq)) { e.preventDefault(); return; }
    if (!piece || piece.color !== this.movable) { this._select(null); return; }
    e.preventDefault();
    this._select(hit.sq);
    this.drag = { from: hit.sq, el: piece.el, moved: false, x0: hit.x, y0: hit.y, pointerId: e.pointerId };
    piece.el.classList.add('cb-dragging');
    this.gPieces.appendChild(piece.el);
    try { this.svg.setPointerCapture(e.pointerId); } catch { /* bryr oss ikke */ }
  }

  _move(e) {
    const d = this.drag;
    if (!d) return;
    const hit = this._sqAt(e.clientX, e.clientY);
    if (!hit) return;
    if (!d.moved && Math.hypot(hit.x - d.x0, hit.y - d.y0) < S * 0.08) return;
    d.moved = true;
    d.el.style.transition = 'none';
    d.el.style.transform = `translate(${hit.x - S / 2}px, ${hit.y - S / 2}px)`;
  }

  _up(e) {
    const d = this.drag;
    if (!d) return;
    this.drag = null;
    d.el.classList.remove('cb-dragging');
    try { this.svg.releasePointerCapture(d.pointerId); } catch { /* bryr oss ikke */ }
    if (!d.moved) { this._place(d.el, d.from, false); return; }  // klikk-valg: behold valget, vent på målruten
    const hit = this._sqAt(e.clientX, e.clientY);
    d.el.style.transition = '';
    if (hit && hit.sq !== d.from && this._tryMove(d.from, hit.sq)) {
      const still = this.pieces.get(d.from);
      if (this.gPromo.childElementCount) this._place(d.el, d.from, false);        // velgeren er åpen: snapp tilbake
      else if (still && still.el === d.el) this._place(d.el, hit.sq, false);     // appen legger inn trekket etterpå
      // ellers har setPosition() allerede flyttet brikken
      return;
    }
    this._place(d.el, d.from, true);
    if (hit && hit.sq !== d.from) this._select(null);
  }

  _cancelDrag() {
    const d = this.drag;
    if (!d) return;
    this.drag = null;
    d.el.classList.remove('cb-dragging');
    this._place(d.el, d.from, true);
  }
}
