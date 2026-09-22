// node --test web/test/board.test.mjs -- board UI state machine (promotion picker lifecycle, keyboard mode).
// Runs without a browser: a tiny fake DOM implements just what board.js touches.
import { test, beforeEach } from 'node:test';
import assert from 'node:assert/strict';

import { Chess } from '../vendor/chess.js';

// ------------------------------------------------------------------ fake DOM
class FakeClassList {
  constructor(el) { this.el = el; this.set = new Set(); }
  add(...c) { c.forEach((x) => this.set.add(x)); this._sync(); }
  remove(...c) { c.forEach((x) => this.set.delete(x)); this._sync(); }
  toggle(c, force) { const on = force === undefined ? !this.set.has(c) : !!force; on ? this.set.add(c) : this.set.delete(c); this._sync(); return on; }
  contains(c) { return this.set.has(c); }
  _sync() { this.el.attrs.class = [...this.set].join(' '); }
}

class FakeEl {
  constructor(tag) {
    this.tag = tag; this.attrs = {}; this.children = []; this.parent = null;
    this.style = {}; this.listeners = {}; this.classList = new FakeClassList(this); this._html = '';
  }
  setAttribute(k, v) { this.attrs[k] = String(v); if (k === 'class') { this.classList.set = new Set(String(v).split(/\s+/).filter(Boolean)); } }
  getAttribute(k) { return this.attrs[k] ?? null; }
  appendChild(c) { c.remove(); c.parent = this; this.children.push(c); return c; }
  replaceChildren(...cs) { this.children.forEach((c) => { c.parent = null; }); this.children = []; cs.forEach((c) => this.appendChild(c)); }
  remove() { if (this.parent) { this.parent.children = this.parent.children.filter((c) => c !== this); this.parent = null; } }
  get childElementCount() { return this.children.length; }
  set innerHTML(v) { this._html = v; this.children = []; }
  get innerHTML() { return this._html; }
  get textContent() { return this._text || ''; }
  set textContent(v) { this._text = v; }
  contains(el) { for (let e = el; e; e = e.parent) if (e === this) return true; return false; }
  querySelector(sel) { return this.querySelectorAll(sel)[0] || null; }
  querySelectorAll(sel) {
    const cls = sel.replace(/^\./, ''); const out = [];
    const walk = (e) => { for (const c of e.children) { if (c.classList.contains(cls)) out.push(c); walk(c); } };
    walk(this); return out;
  }
  addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); }
  dispatch(type, ev = {}) {
    const e = { type, target: this, preventDefault() { e.defaulted = true; }, stopPropagation() { e.stopped = true; }, ...ev };
    for (let el = this; el && !e.stopped; el = el.parent) for (const fn of el.listeners[type] || []) fn(e);
    return e;
  }
  focus() { globalThis.document.activeElement = this; }
  getBBox() { return { x: 0, y: 0, width: 0, height: 0 }; }
  matches() { return false; }
  createSVGPoint() { return { x: 0, y: 0, matrixTransform() { return { x: this.x, y: this.y }; } }; }
  getScreenCTM() { return { inverse() { return {}; } }; }
  setPointerCapture() {} releasePointerCapture() {}
}

globalThis.document = {
  activeElement: null,
  createElementNS: (_ns, tag) => new FakeEl(tag),
  createElement: (tag) => new FakeEl(tag),
};

const { Board } = await import('../board.js');

// ------------------------------------------------------------------ helpers
const PROMO_FEN = '4k3/1P6/8/8/8/8/8/4K3 w - - 0 1';
let host, board, moves, chess;

function setup(fen = PROMO_FEN, movable = 'white') {
  host = new FakeEl('div'); moves = [];
  board = new Board(host, { onMove: (f, t, p) => moves.push([f, t, p]) });
  chess = new Chess(fen);
  board.setPosition(fen, { animate: false });
  board.setMovable(movable);
  board.setLegal(chess.moves({ verbose: true }));
}
const key = (k, target = board.svg) => target.dispatch('keydown', { key: k });
const promoChoices = () => board.gPromo.querySelectorAll('.cb-promo-choice');
const veil = () => board.gPromo.querySelectorAll('.cb-promo-veil');

beforeEach(() => setup());

// ------------------------------------------------------------------ promotion picker lifecycle (website-2)
test('a pawn reaching the last rank opens the promotion picker instead of moving', () => {
  assert.equal(board._tryMove('b7', 'b8'), true);
  assert.equal(promoChoices().length, 4);
  assert.equal(veil().length, 1);
  assert.deepEqual(moves, []);
});

test('picking a piece closes the picker and reports the promotion', () => {
  board._tryMove('b7', 'b8');
  const knight = promoChoices().find((g) => g.attrs['aria-label'] === 'forvandle til springer');
  knight.dispatch('pointerdown');
  assert.equal(board.gPromo.childElementCount, 0);
  assert.deepEqual(moves, [['b7', 'b8', 'n']]);
});

test('setPosition() closes a dangling picker (new game / opponent move / resign)', () => {
  board._tryMove('b7', 'b8');
  assert.equal(promoChoices().length, 4);
  board.setPosition(new Chess().fen(), { animate: false });
  assert.equal(board.gPromo.childElementCount, 0, 'no veil or choices left over the new position');
  assert.deepEqual(moves, []);
});

test('setMovable() closes the picker whether the board is locked or handed to either side', () => {
  for (const arg of [null, 'white', 'black']) {
    setup();
    board._tryMove('b7', 'b8');
    board.setMovable(arg);
    assert.equal(board.gPromo.childElementCount, 0, `setMovable(${arg})`);
  }
});

test('setOrientation()/flip() close the picker', () => {
  board._tryMove('b7', 'b8');
  board.flip();
  assert.equal(board.gPromo.childElementCount, 0);
});

test('cancelPromotion() is public and idempotent; the veil click still works', () => {
  board.cancelPromotion();                        // nothing open: no throw
  board._tryMove('b7', 'b8');
  board.cancelPromotion();
  assert.equal(board.gPromo.childElementCount, 0);
  board._tryMove('b7', 'b8');
  veil()[0].dispatch('pointerdown');
  assert.equal(board.gPromo.childElementCount, 0);
  assert.deepEqual(moves, []);
});

test('after the picker is closed by a new position, the board accepts input again', () => {
  board._tryMove('b7', 'b8');
  const start = new Chess();
  board.setPosition(start.fen(), { animate: false });
  board.setMovable('white');
  board.setLegal(start.moves({ verbose: true }));
  assert.equal(board._tryMove('e2', 'e4'), true);
  assert.deepEqual(moves, [['e2', 'e4', undefined]]);
});

// ------------------------------------------------------------------ keyboard mode (website-7)
test('the board is focusable and not an inert image', () => {
  assert.equal(board.svg.attrs.tabindex, '0');
  assert.notEqual(board.svg.attrs.role, 'img');
});

test('arrows move a cursor, Enter selects a piece and drops it', () => {
  const start = new Chess();
  setup(start.fen());
  key('Enter');                                   // places the cursor on e1 (near king square)
  assert.equal(board.cursor, 'e1');
  assert.ok(board.gMarks.querySelector('.cb-cursor'), 'cursor is drawn while keyboard-driven');
  key('ArrowUp');
  assert.equal(board.cursor, 'e2');
  key('Enter');
  assert.equal(board.selected, 'e2');
  key('ArrowUp'); key('ArrowUp');
  assert.equal(board.cursor, 'e4');
  key('Enter');
  assert.deepEqual(moves, [['e2', 'e4', undefined]]);
  assert.equal(board.selected, null);
});

test('arrows follow the viewer perspective when the board is flipped, and stop at the edge', () => {
  board.setOrientation('black');
  key('ArrowUp');                                 // from the near king square for black (d8)
  assert.equal(board.cursor, 'd7');
  key('ArrowLeft');
  assert.equal(board.cursor, 'e7');               // left on screen = towards the h-file for black
  for (let i = 0; i < 10; i++) key('ArrowLeft');
  assert.equal(board.cursor, 'h7');
});

test('Escape clears the selection, then the promotion picker', () => {
  const start = new Chess();
  setup(start.fen());
  key('Enter'); key('ArrowUp'); key('Enter');
  assert.equal(board.selected, 'e2');
  key('Escape');
  assert.equal(board.selected, null);

  setup();
  board._tryMove('b7', 'b8');
  key('Escape');
  assert.equal(board.gPromo.childElementCount, 0);
  assert.deepEqual(moves, []);
});

test('a keyboard-driven promotion focuses the picker; Enter on a choice moves and returns focus to the board', () => {
  board.cursor = 'b7'; board._kb = true;
  key('Enter');                                   // select b7
  assert.equal(board.selected, 'b7');
  key('ArrowUp'); key('Enter');                   // drop on b8 -> picker
  assert.equal(promoChoices().length, 4);
  assert.equal(document.activeElement, promoChoices()[0], 'first choice focused');
  key('Enter', document.activeElement);
  assert.deepEqual(moves, [['b7', 'b8', 'q']]);
  assert.equal(board.gPromo.childElementCount, 0);
  assert.equal(document.activeElement, board.svg);
});

// ------------------------------------------------------------------ visningsvalg (innstillinger)
test('setVisning slår hjelpemidlene av og på uten å rote til stillingen', () => {
  const start = new Chess();
  setup(start.fen());
  board._select('e2');
  assert.equal(board.gDots.childElementCount, 2, 'e2 kan gå til e3 og e4');
  board.setVisning({ lovlige: false });
  assert.equal(board.gDots.childElementCount, 0, 'prikkene forsvinner når lovlige trekk er slått av');
  board.setVisning({ lovlige: true });
  assert.equal(board.gDots.childElementCount, 2, 'og kommer tilbake igjen');

  board.highlight({ lastMove: ['e2', 'e4'] });
  assert.ok(board.gMarks.querySelector('.cb-last'), 'siste trekk tegnes');
  board.setVisning({ siste: false });
  assert.equal(board.gMarks.querySelector('.cb-last'), null, 'og skjules når innstillingen er av');

  board.setVisning({ koordinater: false });
  assert.equal(board.gCoords.childElementCount, 0);
  board.setVisning({ koordinater: true });
  assert.equal(board.gCoords.childElementCount, 16, 'åtte bokstaver og åtte tall');
});

test('promotion-velgeren navngir brikkene på norsk for skjermlesere', () => {
  board._tryMove('b7', 'b8');
  const navn = promoChoices().map((g) => g.attrs['aria-label']).sort();
  assert.deepEqual(navn, ['forvandle til dronning', 'forvandle til løper', 'forvandle til springer', 'forvandle til tårn']);
});

test('keys do nothing on a locked board and the cursor hides when the pointer takes over', () => {
  board.setMovable(null);
  key('Enter');
  assert.equal(board.selected, null);
  assert.deepEqual(moves, []);
  board.setMovable('white');
  key('ArrowUp');
  assert.ok(board.gMarks.querySelector('.cb-cursor'));
  board.svg.dispatch('blur');
  assert.equal(board.gMarks.querySelector('.cb-cursor'), null);
});
