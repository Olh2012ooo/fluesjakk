// node --test web/test/style.test.mjs -- statiske kontroller på web/style.css:
// lesbarhet (WCAG AA) og at brettet kommer først i den smale layouten.
//
// Testene beskriver designkrav, ikke en bestemt palett. De leser fargetokensene ut av CSS-en og
// regner ut kontrasten, så et nytt tema kan innføres uten at testene må skrives om — men et tema
// med for svak kontrast blir stoppet.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const css = readFileSync(join(here, '..', 'style.css'), 'utf8').replace(/\/\*[\s\S]*?\*\//g, '');

// ---------------------------------------------------------------- helpers
const tokens = Object.fromEntries([...css.matchAll(/--([\w-]+):\s*(#[0-9a-fA-F]{6})\s*;/g)].map((m) => [m[1], m[2].toLowerCase()]));
const lin = (c) => { c /= 255; return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4; };
const lum = (hex) => { const n = parseInt(hex.slice(1), 16); return 0.2126 * lin(n >> 16) + 0.7152 * lin((n >> 8) & 255) + 0.0722 * lin(n & 255); };
const contrast = (a, b) => { const [hi, lo] = [lum(a), lum(b)].sort((p, q) => q - p); return (hi + 0.05) / (lo + 0.05); };
const tok = (name) => { assert.ok(tokens[name], `token --${name} mangler eller er ikke en 6-sifret hex-farge`); return tokens[name]; };

/** Fjerner alle @-regler (media, supports, keyframes …) med balanserte klammer, så et oppslag på
 *  toppnivå ikke ved et uhell plukker opp en regel som bare gjelder inne i en medieforespørsel. */
function utenAtRegler(tekst) {
  let ut = '', i = 0;
  for (;;) {
    const at = tekst.indexOf('@', i);
    const klamme = at < 0 ? -1 : tekst.indexOf('{', at);
    if (at < 0 || klamme < 0) return ut + tekst.slice(i);
    ut += tekst.slice(i, at);
    let dybde = 0, j = klamme;
    for (; j < tekst.length; j++) {
      if (tekst[j] === '{') dybde++;
      else if (tekst[j] === '}' && --dybde === 0) { j++; break; }
    }
    i = j;
  }
}

/** Deklarasjonene til `selector` (eksakt selektortekst) inne i @media-blokken (eller på toppnivå når media er null). */
function decls(selector, media = null) {
  let scope = media ? css : utenAtRegler(css);
  if (media) {
    scope = [...css.matchAll(/@media\s*\(([^)]*)\)\s*\{([\s\S]*?)\n\}/g)].filter((m) => m[1].replace(/\s/g, '') === media.replace(/\s/g, '')).map((m) => m[2]).join('\n');
    assert.ok(scope, `fant ingen @media (${media})-blokk`);
  }
  const out = {};
  for (const m of scope.matchAll(/(^|\n)\s*([^{}\n]+?)\s*\{([^{}]*)\}/g)) {
    if (m[2].split(',').map((s) => s.trim()).includes(selector)) {
      for (const d of m[3].split(';')) { const i = d.indexOf(':'); if (i > 0) out[d.slice(0, i).trim()] = d.slice(i + 1).trim(); }
    }
  }
  return out;
}

/** Går gjennom hver @media-blokk og returnerer kroppene til de som matcher. */
const mediaBlocks = (query) => [...css.matchAll(/@media\s*\(([^)]*)\)\s*\{([\s\S]*?)\n\}/g)]
  .filter((m) => m[1].replace(/\s/g, '') === query.replace(/\s/g, '')).map((m) => m[2]).join('\n');

// ---------------------------------------------------------------- kontrast (tilgjengelighet)
test('--dim brødtekst når WCAG AA (4,5:1) på alle flater den brukes på', () => {
  const dim = tok('dim');
  const brainvizBg = decls('.brainviz').background;
  assert.match(brainvizBg, /^#[0-9a-fA-F]{6}$/, '.brainviz-bakgrunnen bør være en flat hex-farge');
  for (const [navn, bg] of [['--bg', tok('bg')], ['--surface', tok('surface')], ['--surface-2', tok('surface-2')], ['.brainviz', brainvizBg.toLowerCase()]]) {
    const r = contrast(dim, bg);
    assert.ok(r >= 4.5, `--dim ${dim} på ${navn} ${bg}: ${r.toFixed(2)}:1 < 4,5:1`);
  }
});

test('--muted sekundærtekst når WCAG AA på flatene', () => {
  for (const g of ['bg', 'surface', 'surface-2']) {
    const r = contrast(tok('muted'), tok(g));
    assert.ok(r >= 4.5, `--muted på --${g}: ${r.toFixed(2)}:1`);
  }
});

test('--text når WCAG AAA (7:1) på alle flater', () => {
  for (const g of ['bg', 'surface', 'surface-2', 'surface-3']) {
    const r = contrast(tok('text'), tok(g));
    assert.ok(r >= 7, `--text på --${g}: ${r.toFixed(2)}:1 < 7:1`);
  }
});

test('aksentfargene er lesbare som tekst på bakgrunnen', () => {
  for (const navn of ['lime', 'amber', 'blue', 'danger', 'info']) {
    const r = contrast(tok(navn), tok('bg'));
    assert.ok(r >= 4.5, `--${navn} på --bg: ${r.toFixed(2)}:1 < 4,5:1`);
  }
});

test('knappeteksten på den primære knappen har reell kontrast', () => {
  const bg = decls('.btn-primary').background;
  assert.match(bg, /^var\(--lime\)$/, '.btn-primary bør bruke --lime som bakgrunn');
  const r = contrast(tok('text'), tok('lime'));
  assert.ok(r < 4.5, 'primærknappen trenger sin egen mørke tekstfarge, ikke --text');
  // fargen står i regelen som eier .btn-primary sammen med bakgrunnen
  const fg = decls('.btn-primary').color;
  assert.ok(fg && fg !== 'var(--text)', '.btn-primary må sette en eksplisitt mørk tekstfarge');
});

test('koordinatene på brettet har egne farger med reell kontrast mot ruten de ligger på', () => {
  assert.equal(decls('.cb-coord.on-light').fill, 'var(--coord-on-light)');
  assert.equal(decls('.cb-coord.on-dark').fill, 'var(--coord-on-dark)');
  const onLight = contrast(tok('coord-on-light'), tok('sq-light'));
  const onDark = contrast(tok('coord-on-dark'), tok('sq-dark'));
  assert.ok(onLight >= 4.5, `koordinatfarge på lys rute: ${onLight.toFixed(2)}:1`);
  assert.ok(onDark >= 4.5, `koordinatfarge på mørk rute: ${onDark.toFixed(2)}:1`);
});

test('brikkene skiller seg fra rutene de står på', () => {
  // hvit brikke mot lys rute og svart brikke mot mørk rute er de vanskeligste tilfellene
  assert.ok(contrast(tok('piece-w'), tok('sq-light')) >= 3, 'hvit brikke på lys rute');
  assert.ok(contrast(tok('piece-b-ink'), tok('sq-dark')) >= 3, 'omrisset av svart brikke på mørk rute');
  assert.ok(contrast(tok('sq-light'), tok('sq-dark')) >= 1.8, 'lys og mørk rute må kunne skilles');
});

test('de lovlige trekkene og markeringene er synlige men diskré', () => {
  assert.match(decls('.cb-dot').fill, /^var\(--lime\)$/, 'prikkene for lovlige trekk skal bruke aksentfargen');
  assert.ok(Number(decls('.cb-dot').opacity) > 0.2, 'prikkene må være synlige');
  assert.match(decls('.cb-last').fill, /^var\(--hl-last\)$/, 'siste trekk skal ha sin egen markering');
  assert.match(decls('.cb-check').fill, /^var\(--hl-check\)$/, 'sjakk skal ha sin egen markering');
});

// ---------------------------------------------------------------- smal layout
test('under 1080px blir spillet én kolonne med brettet først, så panelet, så visualiseringen', () => {
  const M = 'max-width: 1080px';
  const game = decls('.game', M)['grid-template-columns'];
  assert.ok(game && /1fr/.test(game) && !/340px/.test(game), 'spillet skal kollapse til én kolonne');
  const order = (sel) => { const v = decls(sel, M).order; assert.ok(v !== undefined, `${sel} trenger en eksplisitt order i den smale layouten`); return Number(v); };
  const brett = order('.board-col'), panel = order('.side'), viz = order('.strip');
  assert.ok(brett < panel, 'brettet kommer før fluepanelet og trekklisten');
  assert.ok(panel < viz, 'hjernevisningen kommer til slutt');
  assert.deepEqual([brett, panel, viz], [0, 1, 2], 'rekkefølgen skal være eksplisitt og sammenhengende');
});

test('brettet får alltid plass på skjermen, også på lave vinduer', () => {
  const w = decls('.board-frame').width;
  assert.ok(w, '.board-frame trenger en breddebegrensning');
  assert.match(w, /100dvh|100vh/, 'bredden må ta hensyn til vinduets høyde, ellers renner brettet ut av skjermen');
  assert.match(w, /min\(/, 'bredden må være den minste av kolonnebredden og vindushøyden');
  assert.equal(decls('.cb-host')['aspect-ratio'], '1', 'brettet skal være kvadratisk');
});

test('telefonens hjernecanvas overstyrer grunnregelen (og står etter den)', () => {
  const base = css.indexOf('#brain-canvas {');
  const mobil = css.search(/@media\s*\(max-width:\s*720px\)\s*\{[\s\S]*?#brain-canvas\s*\{\s*height:\s*\d+px/);
  assert.ok(base >= 0, 'grunnregelen for #brain-canvas finnes');
  assert.ok(mobil >= 0, 'det finnes en egen høyde for #brain-canvas på telefon');
  assert.ok(mobil > base, 'telefonregelen må stå etter grunnregelen, ellers vinner ikke kaskaden');
});

test('redusert bevegelse respekteres både fra systemet og fra innstillingene', () => {
  const system = mediaBlocks('prefers-reduced-motion: reduce');
  assert.match(system, /animation-duration/, 'systeminnstillingen må slå av animasjoner');
  assert.match(css, /:root\[data-bevegelse="av"\]/, 'innstillingen i spillet må også slå av animasjoner');
});

test('fokusmarkeringen er synlig og bruker aksentfargen', () => {
  const f = decls(':focus-visible');
  assert.equal(f.outline?.includes('var(--lime)'), true, ':focus-visible må bruke aksentfargen');
  assert.ok(f['outline-offset'], 'fokusringen må ikke ligge oppå elementet');
});
