// app.js — Fluesjakk.
//
// Grensesnittet: forsiden, modellnedlasting, brettet, fluepanelet, hjernevisningen,
// resultatkortet, topplisten og vennelaget. Alle sjakkregler kommer fra chess.js;
// hvert trekk flua gjør kommer fra nettverket som kjører i engine/worker.js.
//
// Ingenting i denne fila regner på modellen. Den sender stillinger til arbeideren og viser
// det den får tilbake. Det er med vilje: grensesnittet kan endres fritt uten at flua endrer
// hvordan den spiller.

import { Chess } from './vendor/chess.js';
import { Board } from './board.js';
import { repetitionCount } from './engine/encoding.js';
import { BrainCanvas, ClassStrip, sampleGroups, KLASSE_NAVN } from './brainviz.js';
import { EyePanel, squareDrive, gazeShift, gazeLine, typeCounts, eyeFiles } from './eye.js';
import { parseArrays, blobProblem } from './engine/loader.js';
import { T, VANSKELIGHETER, VANSKELIGHET_MOTOR, humorFor, kommentar, humorTekst, resultatTekst, delTekst, tall, ms, klokke, mb } from './i18n.js';
import { SLIK_INNHOLD, FEILSOK_INNHOLD } from './slik.js';

const $ = (id) => document.getElementById(id);
const velg = (a) => a[Math.floor(Math.random() * a.length)];
const escapeHtml = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

const VANSKELIGHET_NAVN = Object.fromEntries(VANSKELIGHETER.map((v) => [v.id, v.navn]));

/** Fluas ansiktsuttrykk. `.mouth` byttes ut etter humøret. */
const MUNNER = {
  tenker: 'M92 94 H108',
  trygg: 'M86 92 Q100 99 114 92',
  selvgod: 'M85 93 Q100 98 116 86',
  nervos: 'M86 96 Q100 89 114 96',
  panikk: 'M88 90 Q100 104 112 90 Z',
  nysgjerrig: 'M87 93 Q100 101 113 93',
};

// ============================================================================ arbeiderklient
// En forespørsel som ikke får svar på denne tiden regnes som død arbeider: terminate() og
// minnefeil utløser ingen onerror, så uten dette ville et trekk henge for alltid.
const TIMEOUT_MS = 60_000;

class Hjerne {
  constructor() {
    this.pending = new Map();
    this.nextId = 1;
    this.ready = null;
    this.info = null;
    this.dead = false;
    this.onProgress = () => {};
    this.onThinking = () => {};
    this.onBackend = () => {};
    this._spawn();
  }

  _spawn() {
    this.worker = new Worker('./engine/worker.js', { type: 'module' });
    this.worker.onmessage = (ev) => this._onMessage(ev.data);
    this.worker.onerror = (ev) => { this.dead = true; this._fail(new Error(ev.message || 'arbeideren krasjet')); };
  }

  /** Last fra en URL (vanlig nettside) eller fra ferdig innleste data (nedlastet modell). */
  load({ baseUrl, header, buffer, fromCache } = {}) {
    this._loadArg = header && buffer ? { header, buffer, fromCache } : { baseUrl };
    this.dead = false;
    this.ready = new Promise((resolve, reject) => { this._resolveReady = resolve; this._rejectReady = reject; });
    this.worker.postMessage({ type: 'load', ...this._loadArg });
    return this.ready;
  }

  /** Fersk arbeider + ny lasting. Forespørsler er tilstandsløse (arbeideren spiller opp
   *  msg.moves fra startstillingen), så ingenting annet trenger å gjenopprettes. */
  restart() {
    try { this.worker.terminate(); } catch { /* allerede borte */ }
    this._fail(new Error('starter på nytt'));
    this._spawn();
    if (this._loadArg?.header) this._loadArg = { ...this._loadArg, fromCache: true };
    return this.load(this._loadArg || { baseUrl: 'model/' });
  }

  _fail(err) {
    this._rejectReady?.(err);
    for (const [, p] of this.pending) { clearTimeout(p.timer); p.reject(err); }
    this.pending.clear();
  }

  _settle(id, fn) {
    const p = this.pending.get(id);
    if (!p) return;
    this.pending.delete(id); clearTimeout(p.timer); fn(p);
  }

  _onMessage(msg) {
    if (msg.type === 'progress') { this.onProgress(msg); return; }
    if (msg.type === 'ready') { this.info = msg; this._resolveReady?.(msg); return; }
    if (msg.type === 'backend') { if (this.info) this.info.backend = msg.backend; this.onBackend(msg); return; }
    if (msg.type === 'thinking') {
      const p = this.pending.get(msg.id);
      if (!p || p.cancelled) return;      // framdrift fra et forlatt søk skal ikke lekke inn i grensesnittet
      p.touch(); this.onThinking(msg); return;
    }
    if (msg.type === 'error') {
      if (msg.id && this.pending.has(msg.id)) this._settle(msg.id, (p) => p.reject(new Error(msg.message)));
      else this._fail(new Error(msg.message));
      return;
    }
    this._settle(msg.id, (p) => p.resolve(msg));
  }

  _request(payload) {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      const p = { resolve, reject, timer: 0, type: payload.type, cancelled: false };
      p.touch = () => {
        clearTimeout(p.timer);
        p.timer = setTimeout(() => {
          this.dead = true;
          this._settle(id, () => reject(new Error(T.feil.tidsavbrudd)));
        }, TIMEOUT_MS);
      };
      this.pending.set(id, p);
      p.touch();
      this.worker.postMessage({ ...payload, id });
    });
  }

  move(fen, moves, difficulty) { return this._request({ type: 'move', fen, moves, difficulty, trace: true }); }
  eval(fen, moves) { return this._request({ type: 'eval', fen, moves }); }

  /** Forlat alle trekkforespørsler som er i gang: arbeideren stopper et pågående superflue-søk
   *  i løpet av noen simuleringer i stedet for å fullføre før neste partis første trekk. */
  cancelMoves() {
    for (const [id, p] of this.pending) {
      if (p.type !== 'move' || p.cancelled) continue;
      p.cancelled = true;
      this.worker.postMessage({ type: 'cancel', id });
    }
  }
}

// ============================================================================ lyd (syntetisert)
class Lyder {
  constructor() { this.pa = false; this.ctx = null; this.summing = null; }
  slaaAvPaa() { this.pa = !this.pa; if (!this.pa) this.stoppSumming(); return this.pa; }
  _ctx() {
    if (!this.ctx) this.ctx = new (window.AudioContext || window.webkitAudioContext)();
    if (this.ctx.state === 'suspended') this.ctx.resume();
    return this.ctx;
  }
  klikk(slag = 'trekk') {
    if (!this.pa) return;
    try {
      const ctx = this._ctx(), t = ctx.currentTime;
      const len = slag === 'slag' ? 0.09 : 0.045;
      const buf = ctx.createBuffer(1, Math.ceil(ctx.sampleRate * len), ctx.sampleRate);
      const d = buf.getChannelData(0);
      for (let i = 0; i < d.length; i++) d[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / d.length, 3);
      const src = ctx.createBufferSource(); src.buffer = buf;
      const f = ctx.createBiquadFilter(); f.type = 'lowpass'; f.frequency.value = slag === 'slag' ? 500 : 1800; f.Q.value = 1.2;
      const g = ctx.createGain(); g.gain.value = slag === 'slag' ? 0.5 : 0.32;
      src.connect(f).connect(g).connect(ctx.destination);
      src.start(t);
      if (slag === 'slag') {
        const o = ctx.createOscillator(); o.type = 'sine';
        o.frequency.setValueAtTime(160, t); o.frequency.exponentialRampToValueAtTime(70, t + 0.09);
        const og = ctx.createGain(); og.gain.setValueAtTime(0.22, t); og.gain.exponentialRampToValueAtTime(0.001, t + 0.12);
        o.connect(og).connect(ctx.destination); o.start(t); o.stop(t + 0.13);
      }
    } catch { /* lyd er pynt */ }
  }
  akkord(seier) {
    if (!this.pa) return;
    try {
      const ctx = this._ctx(), t = ctx.currentTime;
      const toner = seier ? [440, 554, 659] : [330, 311, 262];
      toner.forEach((fq, i) => {
        const o = ctx.createOscillator(); o.type = 'triangle'; o.frequency.value = fq;
        const g = ctx.createGain();
        g.gain.setValueAtTime(0.0001, t + i * 0.12);
        g.gain.exponentialRampToValueAtTime(0.12, t + i * 0.12 + 0.02);
        g.gain.exponentialRampToValueAtTime(0.0001, t + i * 0.12 + 0.6);
        o.connect(g).connect(ctx.destination); o.start(t + i * 0.12); o.stop(t + i * 0.12 + 0.65);
      });
    } catch { /* lyd er pynt */ }
  }
  startSumming() {
    if (!this.pa || this.summing) return;
    try {
      const ctx = this._ctx(), t = ctx.currentTime;
      const o = ctx.createOscillator(); o.type = 'sawtooth'; o.frequency.value = 175;
      const o2 = ctx.createOscillator(); o2.type = 'square'; o2.frequency.value = 176.5;
      const lfo = ctx.createOscillator(); lfo.frequency.value = 24;
      const lg = ctx.createGain(); lg.gain.value = 0.012;
      const f = ctx.createBiquadFilter(); f.type = 'lowpass'; f.frequency.value = 700;
      const g = ctx.createGain();
      g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(0.028, t + 0.25);
      lfo.connect(lg).connect(g.gain);
      o.connect(f); o2.connect(f); f.connect(g).connect(ctx.destination);
      o.start(); o2.start(); lfo.start();
      this.summing = { o, o2, lfo, g };
    } catch { /* lyd er pynt */ }
  }
  stoppSumming() {
    const b = this.summing; if (!b) return; this.summing = null;
    try {
      const t = this.ctx.currentTime;
      b.g.gain.cancelScheduledValues(t); b.g.gain.setValueAtTime(b.g.gain.value, t);
      b.g.gain.exponentialRampToValueAtTime(0.0001, t + 0.2);
      setTimeout(() => { b.o.stop(); b.o2.stop(); b.lfo.stop(); }, 260);
    } catch { /* lyd er pynt */ }
  }
}

// ============================================================================ innstillinger
const INNSTILLINGER_NOKKEL = 'fluesjakk.innstillinger.v1';
const STANDARD = {
  lyd: false,
  bevegelse: true,
  lovligeTrekk: true,
  sisteTrekk: true,
  koordinater: true,
  kommentarer: true,
  flueaktivitet: true,
  fluasOyne: true,
  teknisk: false,
  tema: 'kveld',
};

/** Innstillingene beskrives som data, så skjemaet og lagringen alltid holdes i takt. */
const INNSTILLINGER_SKJEMA = [
  { nokkel: 'lyd', navn: 'Lyd', hjelp: 'Klikk når brikkene flyttes, og summing mens flua tenker.' },
  { nokkel: 'kommentarer', navn: 'Fluas kommentarer', hjelp: 'Flua kommenterer trekkene sine underveis.' },
  { nokkel: 'flueaktivitet', navn: 'Vis flueaktivitet', hjelp: 'Nevronene tegnes opp der de faktisk sitter i fluehjernen.' },
  { nokkel: 'fluasOyne', navn: 'Vis fluas øyne', hjelp: 'Fasettøynene og det de ser på brettet.' },
  { nokkel: 'lovligeTrekk', navn: 'Vis lovlige trekk', hjelp: 'Prikker på rutene brikken kan flyttes til.' },
  { nokkel: 'sisteTrekk', navn: 'Vis siste trekk', hjelp: 'Uthever rutene trekket gikk fra og til.' },
  { nokkel: 'koordinater', navn: 'Vis koordinater', hjelp: 'Bokstaver og tall langs brettkanten.' },
  { nokkel: 'bevegelse', navn: 'Animasjoner', hjelp: 'Glidende brikker og levende visualiseringer.' },
  { nokkel: 'teknisk', navn: 'Teknisk modus', hjelp: 'Vis modellinformasjon, verdier og beregningstid.' },
];

const Innstillinger = {
  data: { ...STANDARD },
  last() {
    try { Object.assign(this.data, JSON.parse(localStorage.getItem(INNSTILLINGER_NOKKEL) || '{}')); } catch { /* bruk standard */ }
    return this.data;
  },
  sett(nokkel, verdi) {
    this.data[nokkel] = verdi;
    try { localStorage.setItem(INNSTILLINGER_NOKKEL, JSON.stringify(this.data)); } catch { /* privat modus */ }
  },
  nullstill() {
    this.data = { ...STANDARD };
    try { localStorage.removeItem(INNSTILLINGER_NOKKEL); } catch { /* privat modus */ }
    return this.data;
  },
};

// ============================================================================ modell
// Modellen er 60 MB rå / 50 MB komprimert og ligger ikke i kodelageret. Den lastes ned én gang
// fra Hugging Face og legges i nettleserens cache, slik at spillet virker uten internett etterpå.
// Er modellen allerede lagt ved siden av spillet (mappe `model/`), brukes den og ingenting lastes ned.
const MODELL_CACHE = 'fluesjakk-modell-v1';
const MODELL_KILDE = 'https://huggingface.co/cesp99/fly-chess/resolve/main/web/';
const MODELL_NOKLER = { header: 'modell/brain.json', blob: 'modell/brain.flyb' };

async function cacheAapne() {
  if (typeof caches === 'undefined') return null;
  try { return await caches.open(MODELL_CACHE); } catch { return null; }
}

/** Ligger modellen i den lokale cachen fra en tidligere nedlasting? */
async function hentLagretModell() {
  const c = await cacheAapne();
  if (!c) return null;
  try {
    const [h, b] = await Promise.all([c.match(MODELL_NOKLER.header), c.match(MODELL_NOKLER.blob)]);
    if (!h || !b) return null;
    const header = await h.json();
    const buffer = await b.arrayBuffer();
    if (await blobProblem(header, buffer, { padded: true })) { await slettLagretModell(); return null; }
    return { header, buffer, fromCache: true };
  } catch { return null; }
}

async function slettLagretModell() {
  const c = await cacheAapne();
  if (!c) return;
  try { await Promise.all([c.delete(MODELL_NOKLER.header), c.delete(MODELL_NOKLER.blob)]); } catch { /* ignorer */ }
}

/** Er modellen lagt ved siden av spillet (eller eksportert lokalt)? Da slippes ingen nedlasting. */
async function finnesLokalModell() {
  const base = document.querySelector('meta[name="flue-modell-base"]')?.content?.trim() || 'model/';
  const url = new URL(base.endsWith('/') ? base : base + '/', location.href).href;
  try {
    const r = await fetch(url + 'brain.json', { method: 'HEAD', cache: 'no-cache' });
    return { ok: r.ok, baseUrl: url };
  } catch { return { ok: false, baseUrl: url }; }
}

/**
 * Laster ned modellen med framdrift, pakker ut gzip og kontrollerer sjekksummen før den lagres.
 * @param {(lastet:number, total:number, fase:'header'|'nedlasting'|'pakker'|'lagrer')=>void} onFremdrift
 * @param {AbortSignal} signal
 */
async function lastNedModell(onFremdrift, signal) {
  const kilde = MODELL_KILDE.endsWith('/') ? MODELL_KILDE : MODELL_KILDE + '/';
  const hr = await fetch(kilde + 'brain.json', { cache: 'no-cache', signal });
  if (!hr.ok) throw new Error(T.feil.nettverk);
  const header = await hr.json();
  onFremdrift?.(0, header.gzip_bytes || header.total_bytes || 0, 'header');

  const gr = await fetch(kilde + 'brain.flyb.gz', { cache: 'no-cache', signal });
  if (!gr.ok) throw new Error(T.feil.nettverk);
  const total = Number(gr.headers.get('content-length')) || header.gzip_bytes || 0;

  const buffer = await pakkUt(gr, total, (l) => onFremdrift?.(l, total, 'nedlasting'), signal);
  onFremdrift?.(total, total, 'pakker');

  const problem = await blobProblem(header, buffer);
  if (problem) throw new Error(T.feil.modellKorrupt);

  const padde = buffer.byteLength % 8 === 0
    ? buffer
    : (() => { const p = new ArrayBuffer(Math.ceil(buffer.byteLength / 8) * 8); new Uint8Array(p).set(new Uint8Array(buffer)); return p; })();

  onFremdrift?.(total, total, 'lagrer');
  const c = await cacheAapne();
  if (c) {
    try {
      await Promise.all([
        c.put(MODELL_NOKLER.header, new Response(JSON.stringify(header), { headers: { 'content-type': 'application/json' } })),
        c.put(MODELL_NOKLER.blob, new Response(padde, { headers: { 'content-type': 'application/octet-stream' } })),
      ]);
    } catch { /* full disk eller privat modus: spillet virker denne økten likevel */ }
  }
  return { header, buffer: padde, fromCache: false };
}

/** Strømmevis gunzip med framdrift på de komprimerte bytene. */
async function pakkUt(svar, total, onFremdrift, signal) {
  if (!svar.body || typeof DecompressionStream === 'undefined') {
    const buf = await svar.arrayBuffer();
    onFremdrift?.(total, total);
    return buf;
  }
  const leser = svar.body.getReader();
  const forste = await leser.read();
  if (forste.done) throw new Error(T.feil.modellKorrupt);
  const erGzip = forste.value.byteLength >= 2 && forste.value[0] === 0x1f && forste.value[1] === 0x8b;

  let lastet = 0;
  const opp = new ReadableStream({
    start(c) { c.enqueue(forste.value); },
    async pull(c) {
      if (signal?.aborted) { c.error(new DOMException('avbrutt', 'AbortError')); return; }
      const { done, value } = await leser.read();
      if (done) c.close(); else c.enqueue(value);
    },
    cancel() { return leser.cancel(); },
  });
  const tellende = new TransformStream({
    transform(chunk, c) { lastet += chunk.byteLength; onFremdrift?.(lastet, total); c.enqueue(chunk); },
  });
  let strom = opp.pipeThrough(tellende);
  if (erGzip) strom = strom.pipeThrough(new DecompressionStream('gzip'));
  return new Response(strom).arrayBuffer();
}

// ============================================================================ appen
class App {
  constructor() {
    this.hjerne = new Hjerne();
    this.lyder = new Lyder();
    this.sjakk = new Chess();
    this.brett = null;
    this.viz = new BrainCanvas($('brain-canvas'));
    this.stripe = new ClassStrip($('strip-canvas'));
    this.oye = new EyePanel($('oye-panel'));
    this.grupper = [];
    this.flueSvg = '';
    this.state = null;
    this.vennelag = null;
    this.modell = null;
    this.moteId = 0;
    this.innst = Innstillinger.last();

    this._byggVanskelighetskort();
    this._byggInnstillinger();
    this._bindUI();
    this._brukInnstillinger();
    this._fyllStatiskTekst();
    this._lastGrafikk();
    this._start();
  }

  // ---------------------------------------------------------------- oppstart
  async _lastGrafikk() {
    try {
      this.flueSvg = await (await fetch('assets/flue.svg')).text();
      $('hero-flue').innerHTML = this.flueSvg;
      $('flue-avatar').innerHTML = this.flueSvg;
      $('merke-logo').innerHTML = await (await fetch('assets/logo.svg')).text();
      this._settMunn('nysgjerrig');
    } catch { /* bare pynt */ }
  }

  _fyllStatiskTekst() {
    $('slik-innhold').innerHTML = SLIK_INNHOLD;
    $('feilsok-innhold').innerHTML = FEILSOK_INNHOLD;
    $('nedlasting-forklaring').innerHTML = T.nedlasting.forklaring;
    $('nedlasting-lisens').textContent = T.nedlasting.lisens;
    $('om-grunnlag').innerHTML = OM_GRUNNLAG;
    $('om-personvern').textContent = T.om.personvernTekst;
    $('om-krav').innerHTML = OM_KRAV;
    $('om-lisens').innerHTML = OM_LISENS;
    $('om-snarveier').innerHTML = Object.entries(T.om.hurtigtaster)
      .map(([, v], i) => `<dt>${['F', 'Ctrl+Z', 'N', 'Esc', '↑ ↓ ← →', 'Enter', 'Esc'][i]}</dt><dd>${v}</dd>`).join('');
    document.title = `${T.navn} — ${T.slagord.toLowerCase()}`;
  }

  _byggVanskelighetskort() {
    $('vanskelighetskort').innerHTML = VANSKELIGHETER.map((v, i) => `
      <label class="kort">
        <input type="radio" name="vanskelighet" value="${v.id}"${i === 1 ? ' checked' : ''}>
        <span class="kort-ikon" aria-hidden="true">${v.ikon}</span>
        <span class="kort-navn">${v.navn}</span>
        <span class="kort-kort">${v.kort}</span>
        <span class="kort-tekst">${v.beskrivelse}</span>
      </label>`).join('');
  }

  _byggInnstillinger() {
    $('innstillinger-liste').innerHTML = INNSTILLINGER_SKJEMA.map((s) => `
      <label class="innstilling-rad">
        <input type="checkbox" data-sett="${s.nokkel}">
        <span><span class="innstilling-navn">${s.navn}</span><span class="innstilling-hjelp">${s.hjelp}</span></span>
      </label>`).join('');
    $('innstillinger-liste').addEventListener('change', (e) => {
      const n = e.target.dataset.sett;
      if (!n) return;
      Innstillinger.sett(n, e.target.checked);
      this.innst = Innstillinger.data;
      this._brukInnstillinger();
      this.toast(T.innstillingerLagret);
    });
    $('sett-tema').addEventListener('change', (e) => {
      Innstillinger.sett('tema', e.target.value);
      this.innst = Innstillinger.data;
      this._brukInnstillinger();
    });
    $('btn-nullstill').addEventListener('click', () => {
      this.innst = Innstillinger.nullstill();
      this._brukInnstillinger();
      this.toast(T.innstillingerLagret);
    });
  }

  /** Innstillingene styrer både data-attributter på <html> og synligheten til panelene. */
  _brukInnstillinger() {
    const s = this.innst;
    document.documentElement.dataset.bevegelse = s.bevegelse ? 'på' : 'av';
    document.documentElement.dataset.tema = s.tema;
    for (const el of document.querySelectorAll('[data-sett]')) el.checked = !!s[el.dataset.sett];
    $('sett-tema').value = s.tema;
    $('teknisk-panel').hidden = !s.teknisk;
    $('oye-panel').hidden = !s.fluasOyne;
    $('strip').hidden = !s.flueaktivitet;
    if (this.brett) {
      this.brett.setVisning({ lovlige: s.lovligeTrekk, siste: s.sisteTrekk, koordinater: s.koordinater });
    }
    if (!s.flueaktivitet) this._skjulTrace();
    if (s.teknisk) this._tegnTeknisk();
  }

  async _start() {
    const lokal = await finnesLokalModell();
    if (lokal.ok) { this._startLasting({ baseUrl: lokal.baseUrl }); return; }
    await this._provLagretModell();
  }

  async _provLagretModell() {
    const lagret = await hentLagretModell();
    if (lagret) { this._startLasting({ header: lagret.header, buffer: lagret.buffer, fromCache: true }); return; }
    this._visNedlasting();
  }

  // ---------------------------------------------------------------- nedlasting av modell
  _visNedlasting() {
    $('panel-nedlasting').hidden = false;
    $('panel-oppsett').hidden = true;
    $('nedlasting-feil').hidden = true;
    $('btn-nedlasting').disabled = false;
    $('btn-nedlasting').textContent = T.nedlasting.knapp;
    $('nedlasting-fremdrift').hidden = true;
    $('nedlasting-lisens').textContent = T.nedlasting.lisens;
  }

  async _lastNed() {
    const knapp = $('btn-nedlasting');
    const fremdrift = $('nedlasting-fremdrift');
    const fyll = $('nedlasting-fyll');
    const tekst = $('nedlasting-tekst');
    const pros = $('nedlasting-prosent');
    knapp.disabled = true;
    fremdrift.hidden = false;
    fyll.style.width = '0%';
    tekst.textContent = T.henterModell;
    pros.textContent = '0 %';

    let avbryt = null;
    this._avbrytNedlasting = () => { avbryt?.abort(); };
    knapp.textContent = T.nedlasting.avbryt;

    try {
      avbryt = new AbortController();
      const { header, buffer } = await lastNedModell((lastet, total, fase) => {
        const andel = total ? lastet / total : 0;
        fyll.style.width = `${Math.round(andel * 100)}%`;
        pros.textContent = T.nedlasting.fremdrift(Math.round(andel * 100));
        tekst.textContent = fase === 'header' ? T.henterModell
          : fase === 'pakker' ? T.pakkerUt
            : fase === 'lagrer' ? T.lagrer
              : `${T.lasterNed} — ${mb(lastet)} / ${mb(total)} MB`;
      }, avbryt.signal);
      tekst.textContent = T.nedlasting.starter;
      pros.textContent = '100 %';
      $('panel-nedlasting').hidden = true;
      this.modell = { header, buffer, fromCache: false };
      this._startLasting({ header, buffer, fromCache: false });
    } catch (err) {
      const avbrutt = err?.name === 'AbortError';
      $('panel-nedlasting').hidden = false;
      fremdrift.hidden = true;
      knapp.disabled = false;
      knapp.textContent = T.nedlasting.knapp;
      this._feil(T.feil.nettverkTittel, avbrutt ? T.feil.avbrutt : (err?.message || T.feil.ukjent), () => this._lastNed());
      this._logg('nedlasting feilet', err);
    } finally {
      this._avbrytNedlasting = null;
    }
  }

  // ---------------------------------------------------------------- lasting av modellen
  _startLasting(lastArg) {
    const fyll = $('last-fyll');
    const tekst = $('last-tekst');
    const nevroner = $('last-nevroner').querySelector('b');
    let n = 0, vist = 0, ramme = 0;
    let andel = 0;

    const tell = () => {
      const maal = n * Math.min(1, andel || 0);
      if (Math.abs(maal - vist) > 1) {
        vist += (maal - vist) * 0.2;
        nevroner.textContent = tall(Math.round(vist));
        ramme = requestAnimationFrame(tell);
      } else ramme = 0;
    };

    this.hjerne.onProgress = (m) => {
      if (m.n) n = m.n;
      if (m.phase === 'header') { $('spes-navn').textContent = m.runName || 'flue3'; tekst.textContent = T.henterModell; return; }
      andel = m.total ? m.loaded / m.total : (m.phase === 'decode' ? 1 : 0);
      fyll.style.width = `${Math.round(andel * 100)}%`;
      tekst.textContent = m.phase === 'decode' ? T.pakkerUt
        : m.total ? `${mb(m.loaded)} MB / ${mb(m.total)} MB` : `${mb(m.loaded)} MB`;
      if (!ramme) ramme = requestAnimationFrame(tell);
    };

    $('panel-nedlasting').hidden = true;
    $('panel-oppsett').hidden = false;
    $('lastefeil').hidden = true;
    $('btn-start').disabled = true;
    $('btn-start').textContent = T.vokserHjerne;

    this.hjerne.load(lastArg).then((info) => {
      const h = info.header;
      andel = 1; fyll.style.width = '100%';
      nevroner.textContent = tall(h.n);
      tekst.textContent = info.fromCache ? T.fraHurtiglager : `${mb(info.bytes)} MB`;
      this._oppdaterModellinfo(info);
      this._beskrivOyne(info);
      $('btn-start').disabled = false;
      $('btn-start').textContent = T.klarTilAAspille;
    }).catch((err) => {
      if (ramme) { cancelAnimationFrame(ramme); ramme = 0; }
      this._logg('modelllasting feilet', err);
      const tittel = /brain\.json|404|failed|fetch/i.test(String(err?.message))
        ? T.feil.ingenModellTittel : T.feil.arbeider;
      this._feil(tittel, String(err?.message || T.feil.ukjent), () => {
        this.hjerne.restart().then(() => {
          $('btn-start').disabled = false; $('btn-start').textContent = T.klarTilAAspille;
        }).catch(() => {});
      });
      $('btn-start').textContent = T.ingenModell;
    });
  }

  /** Menneskelig feilmelding med en vei videre — aldri en stakksporing. */
  _feil(tittel, detalj, provIgjen) {
    // meldingen havner i panelet brukeren faktisk ser på
    const iNedlasting = !$('panel-nedlasting').hidden;
    const el = iNedlasting ? $('nedlasting-feil') : $('lastefeil');
    el.hidden = false;
    el.innerHTML = `<b>${escapeHtml(tittel)}</b><p class="dim" style="margin:.4em 0">${escapeHtml(detalj || '')}</p>
      <p class="muted" style="font-size:.82rem">${T.feil.hjelp}</p>
      <div class="dialog-rad" style="margin-top:12px">
        ${provIgjen ? '<button class="btn btn-small" id="btn-laste-prov" type="button">' + T.provIgjen + '</button>' : ''}
        <button class="btn btn-small btn-ghost" id="btn-laste-ned" type="button">${T.nedlasting.knapp}</button>
        <button class="btn btn-small btn-ghost" id="btn-laste-logg" type="button">${T.feil.apneLogg}</button>
      </div>`;
    $('btn-laste-prov')?.addEventListener('click', () => { el.hidden = true; provIgjen(); });
    $('btn-laste-ned')?.addEventListener('click', () => { el.hidden = true; this._visNedlasting(); });
    $('btn-laste-logg')?.addEventListener('click', () => this._visLogg());
  }

  _oppdaterModellinfo(info) {
    const h = info.header;
    $('spes-navn').textContent = h.run_name || 'flue3';
    $('spes-nevroner').textContent = tall(h.n);
    $('spes-forbindelser').textContent = tall(h.nnz_total || h.nnz);
    $('spes-motor').textContent = info.backend === 'webgpu' ? 'WebGPU' : 'JS';
    $('spes-motor').title = info.backend === 'webgpu'
      ? 'framoverpasset kjører på skjermkortet'
      : 'framoverpasset kjører i ren JavaScript';
    this.hjerne.onBackend = (m) => {
      $('spes-motor').textContent = m.backend === 'webgpu' ? 'WebGPU' : 'JS';
      this._logg('motor byttet', m.reason || m.backend);
    };
    $('lede-nevroner').textContent = tall(h.n);
    $('lede-koblinger').textContent = `${(h.nnz_total / 1e6).toFixed(2).replace('.', ',')} millioner`;

    this.viz.setData(info.sample, info.silhouette, info.legend);
    this._tegnForklaring(info);
    this.grupper = sampleGroups(info.sample.idx, info.sample.cls, info.legend, info.retina?.idx);
    this.oye.setRetina(info.retina);
    $('hjerne-under').textContent = `${tall(info.sample.idx.length)} av ${tall(h.n)} nevroner`;
    $('hjerne-tittel').textContent = 'Nevronaktivitet';
    this._teknisk = { info };
    this._tegnTeknisk();
  }

  _tegnForklaring(info) {
    const antall = {};
    for (const c of info.silhouette.cls) antall[c] = (antall[c] || 0) + 1;
    $('legend').innerHTML = '';
    for (const [cls] of Object.entries(antall).sort((a, b) => b[1] - a[1])) {
      const navn = info.legend[cls] || `klasse ${cls}`;
      const [r, g, b] = this.viz.colorOf(+cls);
      const span = document.createElement('span');
      span.style.setProperty('--c', `rgb(${r},${g},${b})`);
      span.textContent = KLASSE_NAVN[navn] || navn.replace(/_/g, ' ');
      $('legend').appendChild(span);
    }
  }

  _beskrivOyne(info) {
    const R = info.retina;
    $('spes-oye').hidden = !R;
    $('lede-oye').hidden = !R; $('fakta-oye').hidden = !R;
    if (!R) return;
    $('spes-oye-n').textContent = tall(R.n);
    const filer = eyeFiles(R.square, R.eye);
    const typer = typeCounts(R.type, R.legend).filter(([, c]) => c > 0).map(([t, c]) => `${tall(c)} ${t}`).join(', ');
    $('lede-oye').innerHTML = `Den har også øyne: <b>${tall(R.n)}</b> fotoreseptorer i de to fasettøynene
      (${escapeHtml(typer)}) ser hver sin rute på brettet — venstre øye ser linjene ${filer.left},
      høyre øye ${filer.right} — og bildet må gjennom den ekte lamina, medulla og lobula før
      sentralhjernen får høre om det.`;
  }

  // ---------------------------------------------------------------- skjermer
  visStart() {
    this.vennelag = null;
    this._tegnVennelag();          // fjerner vennelagsstillingen fra forrige runde
    $('skjerm-start').hidden = false;
    $('skjerm-parti').hidden = true;
    $('skjerm-slik').hidden = true;
    this.viz.setThinking(false);
    this.lyder.stoppSumming();
    window.scrollTo({ top: 0 });
  }

  visSlik() {
    $('skjerm-start').hidden = true;
    $('skjerm-parti').hidden = true;
    $('skjerm-slik').hidden = false;
    window.scrollTo({ top: 0 });
  }

  toast(tekst) {
    const t = $('varsel');
    t.textContent = tekst; t.hidden = false;
    clearTimeout(this._toastT);
    this._toastT = setTimeout(() => { t.hidden = true; }, 2400);
  }

  _logg(hendelse, detalj) {
    this._loggbok = this._loggbok || [];
    this._loggbok.push(`[${new Date().toISOString()}] ${hendelse}: ${detalj?.message || detalj || ''}`);
    if (this._loggbok.length > 200) this._loggbok.shift();
  }

  _visLogg() {
    const h = this.hjerne.info?.header || {};
    const linjer = [
      `${T.navn} — teknisk logg`,
      `tid: ${new Date().toISOString()}`,
      `nettleser: ${navigator.userAgent}`,
      `modell: ${h.run_name || '—'} · ${h.n || '—'} nevroner · ${h.nnz_total || h.nnz || '—'} koblinger`,
      `motor: ${this.hjerne.info?.backend || '—'}`,
      `WebGPU i nettleseren: ${'gpu' in navigator ? 'ja' : 'nei'}`,
      '',
      ...(this._loggbok || []),
    ];
    this._kopier(linjer.join('\n'), 'Teknisk logg kopiert');
  }

  async _kopier(tekst, melding) {
    try { await navigator.clipboard.writeText(tekst); this.toast(melding); }
    catch { this.toast('Utklippstavlen er blokkert'); window.prompt('Kopier:', tekst); }
  }

  // ---------------------------------------------------------------- UI-binding
  _bindUI() {
    $('btn-start').addEventListener('click', () => {
      this.vennelag = null;
      this.spill({
        difficulty: document.querySelector('input[name=vanskelighet]:checked').value,
        color: document.querySelector('input[name=farge]:checked').value,
        name: $('spiller-navn').value.trim() || T.spiller,
      });
    });
    $('btn-nedlasting').addEventListener('click', () => {
      if (this._avbrytNedlasting) this._avbrytNedlasting(); else this._lastNed();
    });
    $('btn-merke').addEventListener('click', () => this.visStart());
    $('btn-slik').addEventListener('click', () => this.visSlik());
    $('btn-nytt').addEventListener('click', () => {
      if (this.vennelag) this.spill({ ...this.state.opts, name: this.vennelag.navn[this.vennelag.nå] });
      else this.visStart();
    });
    $('btn-angre').addEventListener('click', () => this.angre());
    $('btn-snu').addEventListener('click', () => this.snu());
    $('btn-prov').addEventListener('click', () => this.provFluetrekk());
    $('btn-gi-opp').addEventListener('click', () => this.giOpp());
    $('btn-pgn').addEventListener('click', () => this._kopier(this.pgn(), 'PGN kopiert'));
    $('btn-innstillinger').addEventListener('click', () => $('innstillinger-dialog').showModal());
    $('btn-om').addEventListener('click', () => $('om-dialog').showModal());
    $('lenke-om').addEventListener('click', (e) => { e.preventDefault(); $('om-dialog').showModal(); });
    $('btn-feilsoking').addEventListener('click', () => $('feilsok-dialog').showModal());
    $('btn-kopier-logg').addEventListener('click', () => this._visLogg());
    $('btn-toppliste').addEventListener('click', () => { this._tegnToppliste(); $('toppliste-dialog').showModal(); });
    $('btn-tom-liste').addEventListener('click', () => { this._lagreListe([]); this._tegnToppliste(); });
    $('btn-vennelag').addEventListener('click', () => $('vennelag-dialog').showModal());
    $('btn-vennelag-start').addEventListener('click', () => this._startVennelag());
    $('btn-omkamp').addEventListener('click', () => { $('resultat-dialog').close(); this.spill({ ...this.state.opts }); });
    $('btn-neste-spiller').addEventListener('click', () => { $('resultat-dialog').close(); this._nesteVennelagSpiller(); });
    $('btn-kopier-bilde').addEventListener('click', () => this._kopierBilde());
    $('btn-last-ned-bilde').addEventListener('click', () => this._lagreBilde());
    $('btn-kopier-tekst').addEventListener('click', () => this._kopier(this._delTekst(), 'Kopiert'));
    $('spiller-navn').addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !$('btn-start').disabled) { e.preventDefault(); $('btn-start').click(); }
    });

    this.oye.onModeChange = () => { /* teksten oppdateres av EyePanel selv */ };

    document.addEventListener('keydown', (e) => {
      if (e.target.matches('input, textarea, select')) return;
      if (!this.state) return;
      if (e.key === 'Escape') this.brett?.cancelPromotion();
      if (document.querySelector('dialog[open]')) return;
      if (e.key === 'f' && !e.ctrlKey && !e.metaKey && !e.altKey) this.snu();
      if (e.key === 'z' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); this.angre(); }
      if (e.key === 'n' && !e.ctrlKey && !e.metaKey && !e.altKey) this.visStart();
    });

    this.hjerne.onThinking = (m) => { $('kl-sok').textContent = `${m.done}/${m.total}`; };

    const skyv = $('replay-scrub'), steg = $('replay-steg'), spill = $('replay-play');
    this.viz.onReplay = (pos, steps, playing) => {
      skyv.max = String(steps - 1);
      if (document.activeElement !== skyv || !playing) skyv.value = String(pos);
      steg.textContent = `steg ${Math.min(steps, Math.floor(pos + 1e-6) + 1)} / ${steps}`;
      spill.textContent = playing ? '❚❚' : '▶';
      spill.setAttribute('aria-label', playing ? 'Sett tanken på pause' : 'Spill av fluas tanke');
      this.stripe.setPos(pos);
    };
    skyv.addEventListener('input', () => this.viz.seek(+skyv.value));
    spill.addEventListener('click', () => this.viz.toggle());
    $('replay-se').addEventListener('click', () => this.sePaaNytt());

    // «Slik fungerer det»-siden bygges fra slik.js, så knappen fanges opp med delegering
    document.addEventListener('click', (e) => {
      if (e.target?.id === 'btn-slik-spill') this.visStart();
    });
  }

  // ---------------------------------------------------------------- spillflyt
  spill(opts) {
    const farge = opts.color === 'random' ? (Math.random() < 0.5 ? 'white' : 'black') : opts.color;
    this.sjakk = new Chess();
    this.state = {
      opts, menneske: farge === 'white' ? 'w' : 'b', difficulty: opts.difficulty, name: opts.name,
      moves: [], tenktTotalt: 0, sims: 0, resultat: null, start: performance.now(), tenker: false,
    };
    this.moteId++;
    this.hjerne.cancelMoves();

    if (!this.brett) {
      this.brett = new Board($('brett'), { onMove: (f, t, p) => this.menneskeligTrekk(f, t, p), orientation: farge });
      this.brett.setVisning({ lovlige: this.innst.lovligeTrekk, siste: this.innst.sisteTrekk, koordinater: this.innst.koordinater });
    } else {
      this.brett.setOrientation(farge);
    }
    this.brett.setPosition(this.sjakk.fen(), { animate: false });
    this.brett.highlight({ lastMove: null, check: null });
    this.oye.setGame(this.state.menneske === 'w' ? 'b' : 'w');
    this.oye.setBoard(this.sjakk.fen(), null);
    this.blikk = null;
    this._skjulTrace();

    $('skjerm-start').hidden = true;
    $('skjerm-slik').hidden = true;
    $('skjerm-parti').hidden = false;
    $('flue-navn').textContent = `${T.flua} (${VANSKELIGHET_NAVN[opts.difficulty]})`;
    $('kl-siste').textContent = '—';
    $('kl-total').textContent = '0,0 s';
    $('kl-sok').textContent = opts.difficulty === 'superflue' ? '0 sim' : '—';
    $('btn-gi-opp').disabled = false;
    this._tegnTrekk();
    this._tegnVennelag();
    this._settHumor('nysgjerrig', humorTekst('nysgjerrig'));
    this._settVerdi(0);
    this._oppdaterTur();
    if (this.sjakk.turn() !== this.state.menneske) this.flueTrekk();
    else {
      this.hjerne.eval(this.sjakk.fen(), []).then((r) => {
        // fluas syn på startstillingen: verdien er den som skal trekke (din), så snu den for flua
        if (this.state && this.state.moves.length === 0 && !this.state.tenker) {
          if (r.trace) this._visTrace(r.trace, r.traceSteps); else this.viz.setActivity(r.activitySample);
          this._settVerdi(-r.value);
          this._settHumor(humorFor(-r.value), kommentar({ policyTop: r.policyTop, value: -r.value, vanskelighet: this.state.difficulty }));
          this._tegnTeknisk({ policyTop: r.policyTop, verdi: -r.value, trekk: null, tid: null });
        }
      }).catch(() => {});
    }
    window.scrollTo({ top: 0, behavior: this.innst.bevegelse ? 'smooth' : 'auto' });
  }

  _oppdaterTur() {
    const s = this.state;
    const mennesketsTur = this.sjakk.turn() === s.menneske && !s.resultat;
    this.brett.setMovable(mennesketsTur ? s.menneske : null);
    this.brett.setLegal(mennesketsTur ? this.sjakk.moves({ verbose: true }) : []);

    const fluenavn = `${T.flua} · ${VANSKELIGHET_NAVN[s.difficulty]}`;
    const topp = this.brett.orientation === 'white' ? 'b' : 'w';
    const fyll = (el, farge) => {
      const meg = farge === s.menneske;
      el.dataset.side = farge;
      el.querySelector('.navn').innerHTML = escapeHtml(meg ? s.name : fluenavn);
      el.classList.toggle('aktiv', this.sjakk.turn() === farge && !s.resultat);
    };
    fyll($('spiller-topp'), topp);
    fyll($('spiller-bunn'), topp === 'w' ? 'b' : 'w');
    $('tur-merke').textContent = s.resultat ? '' : (mennesketsTur ? T.dinTur : T.fluasTur);

    const st = $('status');
    st.classList.toggle('over', !!s.resultat);
    st.classList.remove('feil');
    $('btn-prov').hidden = true;
    if (s.resultat) st.textContent = s.resultat.tekst;
    else if (this.sjakk.isCheck()) st.textContent = mennesketsTur ? `${T.sjakk}! Din tur` : `${T.sjakk}! Flua er i sjakk`;
    else st.textContent = mennesketsTur ? T.dinTur : T.fluaTenker;
    $('btn-angre').disabled = !mennesketsTur || s.moves.length < 2;
    this.brett.highlight({ check: this.sjakk.isCheck() ? this._kongeRute(this.sjakk.turn()) : null });
  }

  _kongeRute(farge) {
    const b = this.sjakk.board();
    for (let r = 0; r < 8; r++) for (let f = 0; f < 8; f++) {
      const p = b[r][f];
      if (p && p.type === 'k' && p.color === farge) return 'abcdefgh'[f] + (8 - r);
    }
    return null;
  }

  _brukTrekk(mv) {
    const s = this.state;
    s.moves.push(mv.from + mv.to + (mv.promotion || ''));
    this.brett.setPosition(this.sjakk.fen());
    this.brett.highlight({ lastMove: [mv.from, mv.to] });
    this.oye.setBoard(this.sjakk.fen(), mv.from + mv.to);
    if (this.innst.lyd) this.lyder.klikk(mv.captured ? 'slag' : 'trekk');
    this._tegnTrekk();
    const ferdig = this._sjekkSlutt();
    this._oppdaterTur();
    return ferdig;
  }

  menneskeligTrekk(fra, til, forfremmelse) {
    const s = this.state;
    if (!s || s.tenker || s.resultat || this.sjakk.turn() !== s.menneske) return;
    let mv;
    try { mv = this.sjakk.move({ from: fra, to: til, promotion: forfremmelse }); } catch { return; }
    if (!mv) return;
    if (!this._brukTrekk(mv)) this.flueTrekk();
  }

  async flueTrekk() {
    const s = this.state;
    if (!s || s.resultat) return;
    s.tenker = true;
    const id = ++this.moteId;
    const settHumor = (h, tekst) => this._settHumor(h, tekst);
    settHumor('tenker', this.innst.kommentarer ? velg([T.fluaTenker, 'Nevronene jobber.', 'Flua regner på stillingen.']) : null);
    this.viz.setThinking(true);
    if (this.innst.lyd) this.lyder.startSumming();
    $('kommentar').classList.add('fade');

    const fen = this.sjakk.fen();
    let r;
    try {
      // s.difficulty er det norske navnet grensesnittet viser; motoren vil ha protokollnøkkelen
      r = await this.hjerne.move(fen, s.moves.slice(), VANSKELIGHET_MOTOR[s.difficulty] || 'fly');
    } catch (err) {
      if (id !== this.moteId || this.state !== s) return;
      s.tenker = false; this.viz.setThinking(false); this.lyder.stoppSumming();
      $('kommentar').classList.remove('fade');
      this._logg('trekk feilet', err);
      this._settHumor('nervos', T.feil.arbeider);
      this._visProv();
      return;
    }
    if (id !== this.moteId || this.state !== s) return;
    s.tenker = false;
    this.viz.setThinking(false);
    this.lyder.stoppSumming();
    $('kommentar').classList.remove('fade');
    if (!r.move) { this._sjekkSlutt(); this._oppdaterTur(); return; }

    // en liten pause så flua synlig «bestemmer seg» selv når nettverket er raskt
    const vent = Math.max(0, 450 - r.thinkMs);
    if (vent) await new Promise((res) => setTimeout(res, vent));
    if (id !== this.moteId) return;

    const mv = this.sjakk.move({ from: r.move.slice(0, 2), to: r.move.slice(2, 4), promotion: r.move[4] || undefined });
    s.tenktTotalt += r.thinkMs; s.sims = r.sims;
    $('kl-siste').textContent = ms(r.thinkMs);
    $('kl-total').textContent = ms(s.tenktTotalt);
    $('kl-sok').textContent = s.difficulty === 'superflue' ? `${r.sims} sim`
      : (r.sims ? `1 trekk × ${r.sims}` : 'bare policy');
    if (r.backend) $('spes-motor').textContent = r.backend === 'webgpu' ? 'WebGPU' : 'JS';
    if (r.trace) this._visTrace(r.trace, r.traceSteps); else this.viz.setActivity(r.activitySample);
    this._settVerdi(r.value);
    this._tegnTeknisk({ policyTop: r.policyTop, verdi: r.value, trekk: mv.san, tid: r.thinkMs, sims: r.sims });

    let tekst = kommentar({ policyTop: r.policyTop, value: r.value, san: mv.san, vanskelighet: s.difficulty, sims: r.sims });
    const blikk = this._seBlikk(r.retinaDrive, s.moves.length);
    if (blikk && Math.random() < 0.45) tekst += ` ${blikk}`;
    this._settHumor(humorFor(r.value), this.innst.kommentarer ? tekst : null);
    this._brukTrekk(mv);
  }

  /**
   * Retina-strømmen fra stillingen flua nettopp så på (den var selv den som skulle trekke, så
   * fotoreseptorenes ruter er i dens eget perspektiv): gir øyepanelet data og finner, ved å
   * sammenligne med forrige blikk, hvilket (øye, rute) som endret seg mest — en kommentarlinje
   * drevet utelukkende av nettverkets innganger.
   */
  _seBlikk(drive, ply) {
    if (!drive || !this.oye.hasRetina) return '';
    const R = this.oye.retina;
    const trekkNr = Math.floor(ply / 2) + 1;
    this.oye.setDrive(drive, `ved forrige blikk (trekk ${trekkNr})`);
    const nå = squareDrive(drive, R.square, R.eye);
    const skift = gazeShift(this.blikk, nå);
    this.blikk = nå;
    return gazeLine(skift, this.oye.board, this.oye.flyColor);
  }

  _visProv() {
    const st = $('status');
    st.textContent = T.feil.arbeider;
    st.classList.add('feil');
    $('btn-prov').hidden = false;
  }

  async provFluetrekk() {
    const s = this.state;
    if (!s || s.resultat || s.tenker) return;
    $('btn-prov').hidden = true; $('status').classList.remove('feil');
    $('status').textContent = T.fluaTenker;
    if (this.hjerne.dead) {
      try { await this.hjerne.restart(); } catch (err) {
        if (this.state !== s || s.resultat) return;
        this.toast(T.feil.arbeider);
        this._visProv();
        return;
      }
    }
    if (this.state === s && !s.resultat && !s.tenker) this.flueTrekk();
  }

  angre() {
    const s = this.state;
    if (!s || s.tenker || s.resultat || s.moves.length < 2 || this.sjakk.turn() !== s.menneske) return;
    this.sjakk.undo(); this.sjakk.undo();
    s.moves.length -= 2;
    this.brett.setPosition(this.sjakk.fen());
    const siste = this.sjakk.history({ verbose: true }).at(-1);
    this.brett.highlight({ lastMove: siste ? [siste.from, siste.to] : null });
    this.oye.setBoard(this.sjakk.fen(), siste ? siste.from + siste.to : null);
    this.oye.setDrive(null); this.blikk = null;
    this._tegnTrekk();
    this._oppdaterTur();
    this._settHumor('nysgjerrig', 'Flua later som ingenting.');
  }

  giOpp() {
    const s = this.state;
    if (!s || s.resultat) return;
    this.moteId++; this.hjerne.cancelMoves(); s.tenker = false;
    this.viz.setThinking(false); this.lyder.stoppSumming();
    this._avslutt({ utfall: 'tap', grunn: 'resignation', tekst: 'Du ga opp. Flua vinner.' });
  }

  _sjekkSlutt() {
    const c = this.sjakk;
    // chess.js sin hash-baserte trefoldighetssjekk overser gjentakelser der første forekomst
    // fulgte etter et dobbeltbondetrekk med en pinnet (ulovlig) en passant-fangst; FEN-tellingen gjør det ikke
    const trefoldig = repetitionCount(c) >= 3;
    if (!c.isGameOver() && !trefoldig) return false;
    const s = this.state;
    if (c.isCheckmate()) {
      const vinner = c.turn() === 'w' ? 'b' : 'w';
      const jegVant = vinner === s.menneske;
      this._avslutt({
        utfall: jegVant ? 'vinner' : 'tap', grunn: 'checkmate',
        tekst: jegVant ? `${T.sjakkMatt}! Du slo flua.` : `${T.sjakkMatt}. Flua vinner.`,
      });
    } else {
      const grunn = c.isStalemate() ? 'stalemate' : trefoldig ? 'repetition' : c.isInsufficientMaterial() ? 'material' : 'fifty';
      this._avslutt({ utfall: 'remis', grunn, tekst: resultatTekst('remis', s.name, grunn).under });
    }
    return true;
  }

  _avslutt(resultat) {
    const s = this.state;
    s.resultat = resultat;
    s.varighet = performance.now() - s.start;
    s.antallTrekk = Math.ceil(s.moves.length / 2);
    this.brett.setMovable(null);
    this._oppdaterTur();
    $('btn-gi-opp').disabled = true;
    if (this.innst.lyd) this.lyder.akkord(resultat.utfall === 'vinner');
    const humor = resultat.utfall === 'vinner' ? 'panikk' : resultat.utfall === 'tap' ? 'selvgod' : 'nysgjerrig';
    this._settHumor(humor, resultat.utfall === 'vinner' ? 'Flua er knust.'
      : resultat.utfall === 'tap' ? 'Flua pusser vingene, seierrik.' : 'Flua godtar remisen. Motvillig.');
    const rad = {
      navn: s.name, resultat: resultat.utfall, grunn: resultat.grunn, vanskelighet: s.difficulty,
      farge: s.menneske, trekk: s.antallTrekk, varighet: Math.round(s.varighet),
      tenkt: Math.round(s.tenktTotalt), dato: new Date().toISOString(),
    };
    this._lagreListe([rad, ...this._lesListe()]);
    if (this.vennelag) { this.vennelag.resultater[this.vennelag.nå] = rad; this._tegnVennelag(); }
    this._visResultat(rad);
  }

  // ---------------------------------------------------------------- fluepanelet
  _settMunn(humor) {
    const munn = $('flue-avatar').querySelector('.mouth');
    if (munn) munn.setAttribute('d', MUNNER[humor] || MUNNER.nysgjerrig);
  }

  _settHumor(humor, tekst) {
    $('flue-avatar').dataset.humor = humor;
    $('flue-humor').textContent = T.humor[humor] || humor;
    this._settMunn(humor);
    if (tekst !== null && tekst !== undefined) $('kommentar').textContent = tekst;
  }

  _settVerdi(v) {
    $('verdi-prikk').style.left = `${50 + 50 * Math.max(-1, Math.min(1, v))}%`;
    $('verdi-prikk').title = `Verdihodet: ${v.toFixed(2)}`;
  }

  // ---------------------------------------------------------------- hjernevisning
  _visTrace(trace, steps, { autoplay = true } = {}) {
    if (!this.innst.flueaktivitet) return;
    if (!trace || !steps || !this.viz.sample || trace.length !== steps * this.viz.sample.idx.length) { this._skjulTrace(); return; }
    $('replay').hidden = false;
    $('strip-canvas').hidden = false;
    $('strip').hidden = false;
    this.stripe.setTrace(trace, steps, this.grupper);
    this.viz.setTrace(trace, steps, { autoplay });
    $('hjerne-under').textContent = `${tall(this.viz.sample.idx.length)} nevroner · ${steps} tidssteg`;
  }

  _skjulTrace() {
    $('replay').hidden = true;
    $('strip-canvas').hidden = true;
  }

  /** «Se igjen»: be nettverket vurdere stillingen på nytt og spill av den tanken (uten trekk). */
  async sePaaNytt() {
    const s = this.state;
    if (!s || s.tenker) return;
    const knapp = $('replay-se'); knapp.disabled = true;
    try {
      const r = await this.hjerne.eval(this.sjakk.fen(), s.moves.slice());
      if (this.state !== s || s.tenker) return;
      this._visTrace(r.trace, r.traceSteps);
      if (r.backend) $('spes-motor').textContent = r.backend === 'webgpu' ? 'WebGPU' : 'JS';
      this._tegnTeknisk({ policyTop: r.policyTop, verdi: r.value, trekk: null, tid: null });
    } catch (err) {
      this._logg('ny vurdering feilet', err);
      this.toast(T.feil.arbeider);
    } finally { knapp.disabled = false; }
  }

  snu() {
    if (!this.brett) return;
    this.brett.flip();
    if (this.state) this._oppdaterTur();
  }

  // ---------------------------------------------------------------- teknisk modus
  _tegnTeknisk(ekstra = {}) {
    const info = this.hjerne.info;
    if (!info) return;
    // Tallene samles inn uansett, så panelet viser noe fornuftig også når det slås på midt i et parti.
    this._teknisk = { ...(this._teknisk || {}), ...ekstra, info };
    if (!this.innst.teknisk) return;
    const h = info.header;
    const v = this._teknisk;
    const rader = [
      ['Modell', h.run_name || 'flue3'],
      ['Nevroner', tall(h.n)],
      ['Forbindelser', tall(h.nnz_total || h.nnz)],
      ['Synapser', tall(h.total_synapses)],
      ['Tidssteg', h.steps],
      ['Kjøremotor', info.backend === 'webgpu' ? 'WebGPU' : 'JavaScript'],
      ['Fotoreseptorer', h.n_ret ? tall(h.n_ret) : '—'],
      ['Vanskelighetsgrad', VANSKELIGHET_NAVN[this.state?.difficulty] || '—'],
      ['Simuleringer', v.sims ?? '—'],
      ['Verdi', typeof v.verdi === 'number' ? v.verdi.toFixed(3) : '—'],
      ['Valgt trekk', v.trekk || '—'],
      ['Policy-topp', v.policyTop?.length
        ? v.policyTop.slice(0, 3).map((p) => `${p.san} ${Math.round(p.p * 100)}%`).join(' · ') : '—'],
      ['Beregningstid', typeof v.tid === 'number' ? ms(v.tid) : '—'],
    ];
    $('teknisk-rutenett').innerHTML = rader
      .map(([n, x]) => `<dt>${n}</dt><dd><b>${escapeHtml(String(x))}</b></dd>`).join('');
    $('teknisk-motor').textContent = info.backend === 'webgpu' ? 'WebGPU' : 'JS';
  }

  // ---------------------------------------------------------------- trekkliste og PGN
  _tegnTrekk() {
    const ol = $('trekk-liste');
    const hist = this.sjakk.history();
    ol.innerHTML = '';
    if (!hist.length) { ol.innerHTML = `<li class="tom">${T.ingenTrekk}</li>`; $('trekk-antall').textContent = '0'; return; }
    for (let i = 0; i < hist.length; i += 2) {
      const li = document.createElement('li');
      const hvitFlue = this.state.menneske !== 'w' ? ' flue' : '';
      const svartFlue = this.state.menneske !== 'b' ? ' flue' : '';
      li.innerHTML = `<span class="nr">${i / 2 + 1}.</span>`
        + `<span class="san${hvitFlue}${i === hist.length - 1 ? ' nå' : ''}">${escapeHtml(hist[i])}</span>`
        + `<span class="san${svartFlue}${i + 1 === hist.length - 1 ? ' nå' : ''}">${escapeHtml(hist[i + 1] || '')}</span>`;
      ol.appendChild(li);
    }
    $('trekk-antall').textContent = `${hist.length} trekk`;
    ol.scrollTop = ol.scrollHeight;
  }

  pgn() {
    const s = this.state;
    const fluenavn = `Fluesjakk (${VANSKELIGHET_NAVN[s.difficulty]})`;
    this.sjakk.setHeader('Event', 'Menneske mot fluehjerne');
    this.sjakk.setHeader('Site', location.host || 'lokalt');
    this.sjakk.setHeader('Date', new Date().toISOString().slice(0, 10).replace(/-/g, '.'));
    this.sjakk.setHeader('White', s.menneske === 'w' ? s.name : fluenavn);
    this.sjakk.setHeader('Black', s.menneske === 'b' ? s.name : fluenavn);
    const res = !s.resultat ? '*' : s.resultat.utfall === 'remis' ? '1/2-1/2'
      : (s.resultat.utfall === 'vinner') === (s.menneske === 'w') ? '1-0' : '0-1';
    this.sjakk.setHeader('Result', res);
    const pgn = this.sjakk.pgn();
    return pgn.trimEnd().endsWith(res) ? pgn : `${pgn} ${res}`;
  }

  _delTekst() {
    const s = this.state, r = s.resultat;
    return delTekst({
      utfall: r.utfall, vanskelighet: VANSKELIGHET_NAVN[s.difficulty],
      antallTrekk: s.antallTrekk, varighet: klokke(s.varighet),
      adresse: location.href.split('#')[0],
    });
  }

  // ---------------------------------------------------------------- resultat
  _visResultat(rad) {
    const s = this.state, r = s.resultat;
    const { tittel, under } = resultatTekst(r.utfall, s.name, r.grunn);
    $('resultat-eyebrow').textContent = `${T.partietErFerdig} · ${VANSKELIGHET_NAVN[s.difficulty]}`;
    $('resultat-tittel').textContent = tittel;
    $('resultat-under').textContent = `${under} ${s.antallTrekk} trekk · ${klokke(s.varighet)} · flua tenkte i ${ms(s.tenktTotalt)}.`;
    $('btn-neste-spiller').hidden = !this.vennelag || this.vennelag.nå >= this.vennelag.navn.length - 1;
    $('btn-omkamp').textContent = this.vennelag ? 'Spill runden om igjen' : T.spillerIgjen;
    $('del-hint').textContent = '';
    this._tegnDelbilde(rad);
    $('resultat-dialog').showModal();
  }

  _tegnDelbilde() {
    const canvas = $('del-canvas'), ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    const s = this.state, r = s.resultat;
    ctx.fillStyle = '#08090a'; ctx.fillRect(0, 0, W, H);
    const g = ctx.createRadialGradient(W * 0.86, -60, 20, W * 0.86, -60, 760);
    g.addColorStop(0, 'rgba(182,240,74,.15)'); g.addColorStop(1, 'rgba(182,240,74,0)');
    ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);

    if (this.viz.silhouette) {
      const sil = this.viz.silhouette, b = this.viz.bounds;
      const bw = 360, bh = 280, ox = 790, oy = 40;
      const sc = Math.min(bw / (b.maxX - b.minX), bh / (b.maxY - b.minY));
      const cx = ox + (bw - sc * (b.maxX - b.minX)) / 2, cy = oy + (bh - sc * (b.maxY - b.minY)) / 2;
      for (let i = 0; i < sil.cls.length; i++) {
        const [cr, cg, cb] = this.viz.colorOf(sil.cls[i]);
        ctx.fillStyle = `rgba(${cr},${cg},${cb},.42)`;
        ctx.fillRect(cx + (sil.xy[2 * i] - b.minX) * sc, cy + (sil.xy[2 * i + 1] - b.minY) * sc, 2.2, 2.2);
      }
    }
    try { const img = this._brettbilde(); ctx.drawImage(img, 880, 340, 240, 240); } catch { /* ignorer */ }
    if (this.flueSvg) {
      try { const img = svgTilBilde(this.flueSvg); ctx.drawImage(img, 58, 42, 120, 120); } catch { /* ignorer */ }
    }

    const MAXW = 640;
    const linje = (tekst, x, y, skrift, farge) => {
      const m = /^(.*?)(\d+(?:\.\d+)?)px(.*)$/.exec(skrift);
      let str = +m[2];
      const sett = () => { ctx.font = `${m[1]}${str}px${m[3]}`; };
      sett(); ctx.fillStyle = farge;
      while (ctx.measureText(tekst).width > MAXW && str > +m[2] * 0.7) { str -= 1; sett(); }
      let t = tekst;
      while (t.length > 3 && ctx.measureText(t).width > MAXW) t = t.slice(0, -2).trimEnd() + '…';
      ctx.fillText(t, x, y);
    };
    const mono = 'ui-monospace, Menlo, Consolas, monospace';
    const sans = 'system-ui, -apple-system, Segoe UI, sans-serif';

    linje('FLUESJAKK · FLUEWIRE-CONNECTOME', 210, 88, `600 22px ${mono}`, '#b6f04a');
    const tittel = r.utfall === 'vinner' ? 'Jeg slo en fluehjerne' : r.utfall === 'tap' ? 'En fluehjerne slo meg' : 'Remis mot en fluehjerne';
    linje(tittel, 210, 164, `640 62px ${sans}`, '#eef1f3');
    linje('i sjakk.', 210, 234, `640 62px ${sans}`, '#eef1f3');
    linje(`${s.name} · ${VANSKELIGHET_NAVN[s.difficulty]} · ${s.menneske === 'w' ? 'hvit' : 'svart'}`, 210, 312, `30px ${sans}`, '#ffb454');
    linje(`${s.antallTrekk} trekk · ${klokke(s.varighet)} · ${resultatTekst(r.utfall, s.name, r.grunn).under}`, 210, 356, `26px ${mono}`, '#a4adb4');
    const h = this.hjerne.info?.header || {};
    linje(`${tall(h.n || 0)} nevroner · ${tall(h.nnz_total || h.nnz || 0)} ekte koblinger`, 210, 396, `26px ${mono}`, '#a4adb4');
    linje('FlyWire-connectomet · koblet som den ekte flua', 210, 432, `26px ${mono}`, '#a4adb4');
    const trekk = this.sjakk.history();
    linje(trekk.map((m, i) => (i % 2 === 0 ? `${i / 2 + 1}.` : '') + m).join(' '), 210, 500, `22px ${mono}`, '#838c94');
    linje(location.host || 'Fluesjakk', 210, 570, `600 24px ${sans}`, '#b6f04a');
  }

  _brettbilde() {
    const svg = this.brett.svg.cloneNode(true);
    svg.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
    svg.setAttribute('width', '480'); svg.setAttribute('height', '480');
    const css = getComputedStyle(document.documentElement);
    const v = (n) => css.getPropertyValue(n).trim();
    svg.querySelectorAll('.cb-dots, .cb-promo, .cb-coords, .cb-cursor').forEach((e) => e.remove());
    const style = document.createElementNS('http://www.w3.org/2000/svg', 'style');
    style.textContent = `.cb-light{fill:${v('--sq-light')}}.cb-dark{fill:${v('--sq-dark')}}.cb-last{fill:${v('--hl-last')}}.cb-check{fill:${v('--hl-check')}}.cb-selected{fill:none}
      .cb-piece{stroke-width:3.2;stroke-linejoin:round}
      .cb-w{fill:${v('--piece-w')};stroke:${v('--piece-w-ink')}}.cb-w .ink{fill:${v('--piece-w-ink')};stroke:${v('--piece-w-ink')}}
      .cb-b{fill:${v('--piece-b')};stroke:${v('--piece-b-ink')}}.cb-b .ink{fill:${v('--piece-b-ink')};stroke:${v('--piece-b-ink')}}
      .cb-vanish{display:none}`;
    svg.prepend(style);
    svg.querySelectorAll('.cb-piece').forEach((el) => {
      const m = /translate\(([-\d.]+)px,\s*([-\d.]+)px\)/.exec(el.style.transform || '');
      if (m) el.setAttribute('transform', `translate(${m[1]} ${m[2]})`);
      el.removeAttribute('style');
    });
    return svgTilBilde(new XMLSerializer().serializeToString(svg));
  }

  async _kopierBilde() {
    try {
      const blob = await new Promise((res) => $('del-canvas').toBlob(res, 'image/png'));
      await navigator.clipboard.write([new ClipboardItem({ 'image/png': blob })]);
      $('del-hint').textContent = 'Bildet er kopiert';
    } catch (err) {
      $('del-hint').textContent = `Kunne ikke kopiere (${err.name}) — prøv «Lagre bilde»`;
    }
  }

  async _lagreBilde() {
    try {
      const blob = await new Promise((res) => $('del-canvas').toBlob(res, 'image/png'));
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url; a.download = `fluesjakk-${this.state.resultat.utfall}-${Date.now()}.png`;
      document.body.appendChild(a); a.click(); a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 4000);
      $('del-hint').textContent = 'Lagrer …';
    } catch (err) { $('del-hint').textContent = `Nedlasting ble blokkert (${err.name})`; }
  }

  // ---------------------------------------------------------------- toppliste
  _lesListe() { try { return JSON.parse(localStorage.getItem('fluesjakk.toppliste.v1') || '[]'); } catch { return []; } }
  _lagreListe(rader) { try { localStorage.setItem('fluesjakk.toppliste.v1', JSON.stringify(rader.slice(0, 200))); } catch { /* ignorer */ } }

  _ranger(rader) {
    const orden = { vinner: 0, remis: 1, tap: 2 };
    return [...rader].sort((a, b) => (orden[a.resultat] - orden[b.resultat]) || (a.trekk - b.trekk) || (a.varighet - b.varighet));
  }

  _tegnToppliste() {
    const rader = this._ranger(this._lesListe());
    const t = $('toppliste-tabell');
    if (!rader.length) { t.innerHTML = `<tr><td class="tom">${T.topplisteTom}</td></tr>`; return; }
    const navn = { vinner: 'seier', tap: 'tap', remis: 'remis' };
    t.innerHTML = `<tr><th>#</th><th>${T.spillernavn}</th><th>${T.resultatKol}</th><th>${T.vanskelighetskol}</th><th>${T.trekkKol}</th><th>${T.tidKol}</th><th>${T.datoKol}</th></tr>`
      + rader.slice(0, 50).map((r, i) => `<tr><td>${i + 1}</td><td>${escapeHtml(r.navn)}</td>
          <td class="${r.resultat === 'vinner' ? 'vinner' : r.resultat === 'tap' ? 'tap' : 'remis'}">${navn[r.resultat] || r.resultat}</td>
          <td>${VANSKELIGHET_NAVN[r.vanskelighet] || r.vanskelighet}</td><td>${r.trekk}</td>
          <td>${klokke(r.varighet)}</td><td>${String(r.dato).slice(0, 10)}</td></tr>`).join('');
  }

  // ---------------------------------------------------------------- vennelag
  _startVennelag() {
    const navn = $('vennelag-navn').value.split('\n').map((x) => x.trim()).filter(Boolean).slice(0, 32);
    if (navn.length < 2) { this.toast('Skriv inn minst to navn'); return; }
    $('vennelag-dialog').close();
    this.vennelag = {
      navn, nå: 0, resultater: [],
      vanskelighet: $('vennelag-vanskelighet').value,
      farge: $('vennelag-farge').value,
    };
    this.spill({ difficulty: this.vennelag.vanskelighet, color: this.vennelag.farge, name: navn[0] });
  }

  _nesteVennelagSpiller() {
    const p = this.vennelag;
    if (!p) return;
    p.nå = Math.min(p.nå + 1, p.navn.length - 1);
    this.spill({ difficulty: p.vanskelighet, color: p.farge, name: p.navn[p.nå] });
  }

  _tegnVennelag() {
    const p = this.vennelag;
    if (!p) {
      document.querySelector('.vennelag-bar')?.remove();
      return;
    }
    // Vennelagsstillingen vises som en egen rad over trekklisten.
    let el = document.querySelector('.vennelag-bar');
    if (!el) {
      el = document.createElement('div');
      el.className = 'panel vennelag-bar';
      $('skjerm-parti').querySelector('.side').prepend(el);
    }
    const rangerte = this._ranger(p.resultater.filter(Boolean));
    const leder = rangerte[0];
    const ferdige = p.resultater.filter(Boolean).length;
    const rader = p.navn.map((n, i) => {
      const r = p.resultater[i];
      const naa = i === p.nå && !r;
      const res = r
        ? `${r.resultat === 'vinner' ? T.sloFlua : r.resultat === 'tap' ? T.tapte : T.uavgjort} · ${r.trekk} trekk · ${klokke(r.varighet)}`
        : naa ? T.spillerNa : T.venter;
      return `<li class="${naa ? 'naa' : r ? 'ferdig' : ''}"><span>${i + 1}. ${escapeHtml(n)}</span><span class="res">${res}</span></li>`;
    });
    const slutt = ferdige === p.navn.length
      ? (leder && leder.resultat === 'vinner' ? T.sloFluaRaskest(escapeHtml(leder.navn)) : T.ingenSloFlua)
      : T.antallSpilt(ferdige, p.navn.length);
    el.innerHTML = `<div class="viz-topp"><span>${T.vennelag}</span><span>${slutt}</span></div>
      <ol class="vennelag-liste">${rader.join('')}</ol>
      ${ferdige === p.navn.length ? `<button class="btn btn-ghost btn-small" id="btn-vennelag-slutt" type="button">${T.avsluttVennelag}</button>` : ''}`;
    $('btn-vennelag-slutt')?.addEventListener('click', () => { this.vennelag = null; this._tegnVennelag(); });
  }
}

// ============================================================================ statiske tekster
const OM_GRUNNLAG = `
  <p>Motstanderen er et rekurrent nevralt nettverk der koblingene er hentet fra
  <a href="https://flywire.ai" rel="noopener">FlyWire</a>-rekonstruksjonen av en voksen fruktflue
  (<i>Drosophila melanogaster</i>): 134 209 nevroner og 2,7 millioner nevron-til-nevron-koblinger,
  med signaltypen til hver celle bestemt fra elektronmikroskopibilder.</p>
  <p>Connectomet, celletypene og nevrotransmitterne er forskningsresultater fra FlyWire-konsortiet
  og er ikke laget av meg. Det samme gjelder Lichess-partiene modellen er trent på.
  Selve oppskriften — hvordan brettet presenteres for fluen og hvordan trekkene leses ut — er en
  modelleringsbeslutning som er beskrevet i <code>docs/SPEC.md</code>.</p>
  <p>Se <b>CREDITS.md</b> i prosjektet for full attribusjon og lisenser.</p>`;

const OM_KRAV = `
  <p><b>Windows:</b> Fluesjakk.exe (Windows 10 eller nyere, 64-bit). Ingen installasjon, ingen Python,
  ingen Node og ingen terminal.</p>
  <p><b>Nettleser:</b> en moderne nettleser med støtte for ES-moduler og Web Workers
  (Chrome, Edge, Firefox eller Safari i nyere versjon). Har nettleseren WebGPU, kjører nettverket på
  skjermkortet og flua svarer merkbart raskere.</p>
  <p><b>Plass:</b> omtrent 120 MB til modellen (nedlastet én gang) og rundt 35 MB
  arbeidsminne mens spillet kjører. Superflue bruker mer når den søker.</p>`;

const OM_LISENS = `
  <p>Koden i dette prosjektet er utgitt under <b>MIT-lisensen</b>. Modellvektene er avledet av
  FlyWire-dataene og er lisensiert under <b>CC BY-NC 4.0</b> — de kan brukes fritt til
  ikke-kommersielle formål, med attribusjon. Sjakkreglene kommer fra
  <a href="https://github.com/jhlywa/chess.js" rel="noopener">chess.js</a> (BSD-2-Clause).
  Lichess-partiene modellen er trent på er CC0.</p>
  <p>Det betyr at du gjerne kan lese, endre og dele videre koden, men at spillet ikke kan selges
  eller brukes kommersielt så lenge det bygger på FlyWire-connectomet.</p>`;

function svgTilBilde(svgTekst) {
  return new Promise((resolve, reject) => {
    const blob = new Blob([svgTekst], { type: 'image/svg+xml;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const img = new Image();
    img.onload = () => { URL.revokeObjectURL(url); resolve(img); };
    img.onerror = (e) => { URL.revokeObjectURL(url); reject(e); };
    img.src = url;
  });
}

window.fluesjakk = new App();
