// node --test web/test/app.test.mjs -- hele siden i headless Chromium (rå CDP, ingen avhengigheter).
// Serverer web/ med en syntetisk modell og styrer window.fluesjakk. Hoppes over når ingen Chromium
// finnes (sett CHROME=/sti/til/chrome for å peke på en).
//
// Testene dekker grensesnittet, ikke motoren: spillflyt, vennelag, feilhåndtering, innstillinger,
// øyepanelet og tankegjenspillingen. Motorens tall testes i flybrain/parity-testene.
import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { createServer } from 'node:http';
import { existsSync, readdirSync, readFileSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir, homedir } from 'node:os';
import { dirname, join, extname } from 'node:path';
import { fileURLToPath } from 'node:url';

import { synthModel } from './support/synthmodel.mjs';

const WEB = join(dirname(fileURLToPath(import.meta.url)), '..');
const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.css': 'text/css', '.svg': 'image/svg+xml', '.json': 'application/json', '.flyb': 'application/octet-stream' };

/** Chromium på Linux, macOS og Windows (Chrome så vel som Edge). */
function findChrome() {
  const cands = [
    process.env.CHROME, process.env.CHROME_PATH,
    '/usr/bin/chromium', '/usr/bin/chromium-browser', '/usr/bin/google-chrome', '/usr/bin/google-chrome-stable',
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    join(homedir(), '.cache', 'ms-playwright'),                       // håndteres under
  ];
  const pw = join(homedir(), '.cache', 'ms-playwright');
  if (existsSync(pw)) for (const d of readdirSync(pw)) if (d.startsWith('chromium')) cands.push(join(pw, d, 'chrome-linux64', 'chrome'), join(pw, d, 'chrome-linux', 'chrome'));

  const win = process.platform === 'win32';
  if (win) {
    for (const root of [process.env['ProgramFiles'], process.env['ProgramFiles(x86)'], process.env.LOCALAPPDATA]) {
      if (!root) continue;
      cands.push(join(root, 'Google', 'Chrome', 'Application', 'chrome.exe'));
      cands.push(join(root, 'Microsoft', 'Edge', 'Application', 'msedge.exe'));
    }
  }
  return cands.find((c) => c && existsSync(c)) || null;
}
const CHROME = findChrome();
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ------------------------------------------------------------------ statisk server (web/ + modell)
// en fly3-aktig blob med liten retina (øyepanelet) og, for den retina-løse stien, en blob uten
// tilleggsfunksjoner slik en eldre eksport så ut
const models = { retina: synthModel({ nRet: 16, seed: 3 }), legacy: synthModel({ legacy: true }) };
const serve = { model: true, legacy: false };
const server = createServer((req, res) => {
  const path = decodeURIComponent(new URL(req.url, 'http://x').pathname);
  if (path.startsWith('/model/')) {
    if (!serve.model) { res.writeHead(404); res.end('ingen modell'); return; }
    const model = serve.legacy ? models.legacy : models.retina;
    if (path === '/model/brain.json') { res.writeHead(200, { 'content-type': MIME['.json'] }); res.end(JSON.stringify(model.header)); return; }
    if (path === '/model/brain.flyb') { res.writeHead(200, { 'content-type': MIME['.flyb'], 'content-length': model.blob.byteLength }); res.end(model.blob); return; }
    res.writeHead(404); res.end(); return;
  }
  const file = join(WEB, path === '/' ? 'index.html' : path);
  if (!file.startsWith(WEB) || !existsSync(file)) { res.writeHead(404); res.end(); return; }
  res.writeHead(200, { 'content-type': MIME[extname(file)] || 'application/octet-stream' });
  res.end(readFileSync(file));
});

// ------------------------------------------------------------------ minimal CDP-klient
class Page {
  constructor(ws) {
    this.ws = ws; this.id = 0; this.pending = new Map(); this.errors = [];
    ws.onmessage = (ev) => {
      const m = JSON.parse(ev.data);
      if (m.id && this.pending.has(m.id)) { this.pending.get(m.id)(m); this.pending.delete(m.id); }
      else if (m.method === 'Runtime.exceptionThrown') this.errors.push(m.params.exceptionDetails?.exception?.description || m.params.exceptionDetails?.text);
    };
  }
  send(method, params = {}) { return new Promise((r) => { const i = ++this.id; this.pending.set(i, r); this.ws.send(JSON.stringify({ id: i, method, params })); }); }
  /** evaluerer et JS-uttrykk (kan være et løfte) og returnerer JSON-verdien */
  async eval(expr) {
    const m = await this.send('Runtime.evaluate', { expression: expr, returnByValue: true, awaitPromise: true });
    if (m.result?.exceptionDetails) throw new Error(`sideeval feilet: ${m.result.exceptionDetails.exception?.description || m.result.exceptionDetails.text}\n${expr}`);
    return m.result?.result?.value;
  }
  async waitFor(expr, { timeout = 15000, what = expr } = {}) {
    const t0 = Date.now();
    while (Date.now() - t0 < timeout) { if (await this.eval(expr)) return; await sleep(40); }
    throw new Error(`tidsavbrudd: ventet på ${what}`);
  }
  click(sel) { return this.eval(`document.querySelector(${JSON.stringify(sel)}).click(), true`); }
  async key(key, { code = `Key${key.toUpperCase()}`, modifiers = 0 } = {}) {
    await this.send('Input.dispatchKeyEvent', { type: 'keyDown', key, code, text: modifiers ? undefined : key, modifiers });
    await this.send('Input.dispatchKeyEvent', { type: 'keyUp', key, code, modifiers });
  }
  async goto(url) {
    await this.send('Page.navigate', { url });
    await this.waitFor(`document.readyState === 'complete' && !!window.fluesjakk`, { what: 'sidelasting' });
  }
}

let chrome, page, base, profile;
const SNAP = `(() => { const a = window.fluesjakk, $ = (i) => document.getElementById(i); return {
  status: $('status').textContent, feil: $('status').classList.contains('feil'), prov: !$('btn-prov').hidden,
  humor: $('flue-avatar').dataset.humor, faded: $('kommentar').classList.contains('fade'), kommentar: $('kommentar').textContent,
  tenker: a.state?.tenker, flyttbar: a.brett?.movable ?? null, angre: $('btn-angre').disabled, hist: a.sjakk.history(),
  topp: $('spiller-topp').textContent, bunn: $('spiller-bunn').textContent, orientering: a.brett?.orientation,
  vennelag: a.vennelag ? { navn: a.vennelag.navn, naa: a.vennelag.naa, resultater: a.vennelag.resultater.map((r) => r && r.navn) } : null,
  vennelagBar: !!document.querySelector('.vennelag-bar'),
  død: a.hjerne.dead, sims: $('kl-sok').textContent,
  startSkjerm: !$('skjerm-start').hidden, partiSkjerm: !$('skjerm-parti').hidden, slikSkjerm: !$('skjerm-slik').hidden }; })()`;
const HUMAN_TURN = `(() => { const a = window.fluesjakk; return !!a.state && (!!a.state.resultat || (!a.state.tenker && a.sjakk.turn() === a.state.menneske)); })()`;

async function loadPage() {
  await page.goto(base);
  await page.waitFor(`!document.getElementById('btn-start').disabled`, { what: 'modell lastet', timeout: 30000 });
}
async function startGame({ difficulty = 'flue', color = 'white', name = 'Tester' } = {}) {
  if (await page.eval(`document.getElementById('resultat-dialog').open`)) await page.eval(`document.getElementById('resultat-dialog').close(), true`);
  if (await page.eval(`document.getElementById('skjerm-parti').hidden`)) await page.click('#btn-merke');
  await page.eval(`document.querySelector('input[name=vanskelighet][value=${difficulty}]').checked = true;
    document.querySelector('input[name=farge][value=${color}]').checked = true;
    document.getElementById('spiller-navn').value = ${JSON.stringify(name)}; true`);
  await page.click('#btn-start');
  await page.waitFor(HUMAN_TURN, { what: 'menneskets tur' });
}
const humanMove = (from, to) => page.eval(`window.fluesjakk.menneskeligTrekk('${from}', '${to}'), true`);

before(async () => {
  if (!CHROME) return;
  await new Promise((r) => server.listen(0, '127.0.0.1', r));
  base = `http://127.0.0.1:${server.address().port}/`;
  profile = mkdtempSync(join(tmpdir(), 'fluesjakk-test-'));
  const port = 9400 + Math.floor(Math.random() * 400);
  // stderr holdes åpen: når nettleseren ikke kommer opp, er det den eneste ledetråden vi har
  chrome = spawn(CHROME, ['--headless=new', '--no-sandbox', '--disable-gpu', '--hide-scrollbars', '--window-size=1400,900', `--remote-debugging-port=${port}`, `--user-data-dir=${profile}`, 'about:blank'], { stdio: ['ignore', 'ignore', 'pipe'] });
  let feilutskrift = '';
  chrome.stderr?.on('data', (b) => { feilutskrift = (feilutskrift + b).slice(-2000); });
  // Vent til feilsøkingsporten svarer MED en ferdigregistrert side. Endepunktet begynner å svare
  // før vinduet er registrert som target, så et svar alene er ikke nok — det var nettopp derfor
  // dette var periodisk rødt i CI. En kald nettleser på en lastet runner kan dessuten bruke godt
  // over ti sekunder, så vi venter tålmodig og sier fra om hva vi faktisk prøvde.
  let side = null;
  for (let i = 0; i < 600 && !side; i++) {
    if (chrome.exitCode !== null) break;
    try { side = (await (await fetch(`http://127.0.0.1:${port}/json/list`)).json()).find((t) => t.type === 'page') || null; } catch { /* porten er ikke åpen ennå */ }
    if (!side) await sleep(100);
  }
  assert.ok(side, `nettleseren startet ikke (${CHROME}, avsluttet med kode ${chrome.exitCode})\n${feilutskrift}`);
  const ws = new WebSocket(side.webSocketDebuggerUrl);
  await new Promise((r, j) => { ws.onopen = r; ws.onerror = j; });
  page = new Page(ws);
  await page.send('Runtime.enable'); await page.send('Page.enable');
});
after(async () => {
  page?.ws.close(); chrome?.kill('SIGKILL'); server.close();
  if (profile) await sleep(200).then(() => rmSync(profile, { recursive: true, force: true }));
});

const opts = { skip: CHROME ? false : 'fant ingen Chromium (sett CHROME=/sti/til/chrome)' };

// ---------------------------------------------------------------- navigasjon og skjermer
test('merket går tilbake til forsiden, og «Slik fungerer det» er en egen side', opts, async () => {
  await loadPage();
  let s = await page.eval(SNAP);
  assert.equal(s.startSkjerm, true);

  await page.click('#btn-slik');
  s = await page.eval(SNAP);
  assert.equal(s.slikSkjerm, true); assert.equal(s.startSkjerm, false);
  const tittel = await page.eval(`document.querySelector('#slik-innhold h1').textContent`);
  assert.equal(tittel, 'Slik fungerer Fluesjakk');
  const steg = await page.eval(`document.querySelectorAll('#slik-innhold .flyt li').length`);
  assert.equal(steg, 8, 'rørledningen har åtte steg');

  await page.click('#btn-slik-spill');
  s = await page.eval(SNAP);
  assert.equal(s.startSkjerm, true); assert.equal(s.slikSkjerm, false);
  assert.deepEqual(page.errors, [], 'ingen ubehandlede feil i siden');
});

test('vanskelighetskortene er norske, forhåndsvalgt til Flue, og styrer partiet', opts, async () => {
  await loadPage();
  const kort = await page.eval(`[...document.querySelectorAll('#vanskelighetskort .kort')].map((k) => ({
    navn: k.querySelector('.kort-navn').textContent, valgt: k.querySelector('input').checked, id: k.querySelector('input').value }))`);
  assert.deepEqual(kort.map((k) => k.navn), ['Larve', 'Flue', 'Superflue']);
  assert.deepEqual(kort.map((k) => k.id), ['larve', 'flue', 'superflue']);
  assert.equal(kort.filter((k) => k.valgt).length, 1);
  assert.equal(kort.find((k) => k.valgt).id, 'flue', 'Flue er standardvalget');

  await startGame({ difficulty: 'larve' });
  assert.match(await page.eval(`document.getElementById('flue-navn').textContent`), /Flua \(Larve\)/);
  await humanMove('e2', 'e4');
  await page.waitFor(HUMAN_TURN);
  const sims = (await page.eval(SNAP)).sims;
  assert.equal(sims, 'bare policy', `larve bruker verken søk eller 1-trekk-sjekk (viste «${sims}»)`);
});

// Regresjonsvakt: nivånavnene grensesnittet viser er norske, men arbeideren kjenner dem under
// engelske protokollnøkler (SPEC §9). Oversettes det ene men ikke det andre, faller nivået stille
// tilbake til «fly» — og spillet ser riktig ut helt til noen legger merke til at superflue svarer
// på millisekunder. Denne testen sjekker at hvert nivå faktisk driver riktig motormodus.
test('hvert vanskelighetsnivå driver riktig motormodus', opts, async () => {
  await loadPage();
  const nivaer = [
    ['larve', 'Larve', /^bare policy$/, 'policy-sampling uten framsyn'],
    ['flue', 'Flue', /^1 trekk × 3$/, '1-trekk-sjekk over de tre beste kandidatene'],
    ['superflue', 'Superflue', /^\d+ sim$/, 'Monte-Carlo-tresøk'],
  ];
  for (const [id, navn, monster, hva] of nivaer) {
    await startGame({ difficulty: id });
    assert.match(await page.eval(`document.getElementById('flue-navn').textContent`), new RegExp(`\\(${navn}\\)`));
    await humanMove('e2', 'e4');
    await page.waitFor(HUMAN_TURN);
    const sims = (await page.eval(SNAP)).sims;
    assert.match(sims, monster, `${id} skal kjøre ${hva}, men viste «${sims}»`);
    if (id === 'superflue') {
      const n = Number(sims.split(' ')[0]);
      assert.ok(n >= 40, `superflue skal søke gjennom mange varianter, søkte bare ${n}`);
    }
  }
  assert.deepEqual(page.errors, []);
});

// ---------------------------------------------------------------- spillet
test('snu bytter spilleretikettene og ignoreres bak en dialog og med Ctrl', opts, async () => {
  await loadPage();
  await startGame({ color: 'white' });
  let s = await page.eval(SNAP);
  assert.equal(s.orientering, 'white');
  assert.match(s.topp, /Flua · Flue/); assert.match(s.bunn, /Tester/);
  await page.click('#btn-snu');
  s = await page.eval(SNAP);
  assert.equal(s.orientering, 'black');
  assert.match(s.topp, /Tester/, 'spilleren følger brikkene til toppen');
  assert.match(s.bunn, /Flua · Flue/);
  await page.key('f');
  assert.equal((await page.eval(SNAP)).orientering, 'white');
  await page.key('f', { modifiers: 2 });                                 // Ctrl+F er nettleserens søk
  assert.equal((await page.eval(SNAP)).orientering, 'white');
  await page.click('#btn-om');
  assert.equal(await page.eval(`document.getElementById('om-dialog').open`), true);
  await page.key('f');
  assert.equal((await page.eval(SNAP)).orientering, 'white', 'f bak en åpen dialog skal ikke snu brettet');
  await page.eval(`document.getElementById('om-dialog').close(), true`);
});

test('et helt parti spilles, sjakk vises, og resultatet havner på topplisten', opts, async () => {
  await loadPage();
  await startGame({ color: 'white', name: 'Oliver' });
  await humanMove('e2', 'e4');
  await page.waitFor(HUMAN_TURN);
  const s = await page.eval(SNAP);
  assert.equal(s.hist.length, 2);
  assert.match(s.status, /Din tur/);
  assert.equal(await page.eval(`document.getElementById('tur-merke').textContent`), 'Din tur');

  // gi opp avslutter partiet og lagrer raden
  await page.click('#btn-gi-opp');
  await page.waitFor(`document.getElementById('resultat-dialog').open`);
  assert.match(await page.eval(`document.getElementById('resultat-tittel').textContent`), /Flua vant/);
  await page.eval(`document.getElementById('resultat-dialog').close(), true`);
  await page.click('#btn-toppliste');
  const rader = await page.eval(`[...document.querySelectorAll('#toppliste-tabell tr')].map((tr) => tr.textContent)`);
  assert.ok(rader.some((r) => r.includes('Oliver') && r.includes('tap')), `topplisten har raden: ${rader.join(' | ')}`);
  await page.eval(`document.getElementById('toppliste-dialog').close(), true`);
});

test('«Nytt parti» fra spillet går tilbake til forsiden, og navnet huskes', opts, async () => {
  await loadPage();
  await startGame({ name: 'Kari' });
  await page.click('#btn-nytt');
  let s = await page.eval(SNAP);
  assert.equal(s.startSkjerm, true); assert.equal(s.partiSkjerm, false);
  assert.equal(await page.eval(`document.getElementById('spiller-navn').value`), 'Kari');
  assert.deepEqual(page.errors, []);
});

// ---------------------------------------------------------------- vennelag
test('vennelaget spiller alle mot samme flue, og et nytt enkeltparti rører ikke resultatene', opts, async () => {
  await page.eval(`document.getElementById('vennelag-navn').value = 'Alice\\nBob'; window.fluesjakk._startVennelag(); true`);
  await page.waitFor(HUMAN_TURN);
  assert.equal((await page.eval(SNAP)).vennelagBar, true);
  await page.click('#btn-gi-opp');
  await page.waitFor(`document.getElementById('resultat-dialog').open`);
  await page.click('#btn-neste-spiller');
  await page.waitFor(HUMAN_TURN);
  await page.click('#btn-gi-opp');
  await page.waitFor(`document.getElementById('resultat-dialog').open`);
  let s = await page.eval(SNAP);
  assert.deepEqual(s.vennelag.resultater, ['Alice', 'Bob']);
  assert.match(await page.eval(`document.querySelector('.vennelag-bar').textContent`), /Ingen slo flua/);
  await page.eval(`document.getElementById('resultat-dialog').close(), true`);

  await page.click('#btn-merke');
  s = await page.eval(SNAP);
  assert.equal(s.vennelag, null, 'merket avslutter vennelaget'); assert.equal(s.vennelagBar, false);
  await startGame({ name: 'Carol' });
  await page.click('#btn-gi-opp');
  await page.waitFor(`document.getElementById('resultat-dialog').open`);
  assert.equal((await page.eval(SNAP)).vennelag, null, 'et enkeltparti skal ikke røre et avsluttet vennelag');
  assert.equal(await page.eval(`document.getElementById('btn-omkamp').textContent`), 'Spill igjen');
  await page.eval(`document.getElementById('resultat-dialog').close(), true`);
});

// ---------------------------------------------------------------- feilhåndtering
test('et avvist fluetrekk gir en varig «Prøv igjen» i stedet for et låst brett', opts, async () => {
  await startGame();
  await page.eval(`const a = window.fluesjakk; a._ekteTrekk = a.hjerne.move; a.hjerne.move = () => Promise.reject(new Error('boom')); true`);
  await humanMove('e2', 'e4');
  await page.waitFor(`!document.getElementById('btn-prov').hidden`, { what: 'prøv igjen-knappen' });
  let s = await page.eval(SNAP);
  assert.equal(s.feil, true); assert.equal(s.tenker, false); assert.equal(s.faded, false);
  assert.match(s.status, /Fluehjernen sluttet å svare/);
  assert.equal(s.flyttbar, null, 'det er fortsatt fluas tur'); assert.equal(s.angre, true);
  assert.equal(await page.eval(`document.getElementById('varsel').hidden`), true, 'ingen flyktig melding');
  assert.ok(s.kommentar.length > 0);

  await page.eval(`window.fluesjakk.hjerne.move = window.fluesjakk._ekteTrekk; true`);
  await page.click('#btn-prov');
  await page.waitFor(HUMAN_TURN, { what: 'fluas svar etter nytt forsøk' });
  s = await page.eval(SNAP);
  assert.equal(s.hist.length, 2); assert.equal(s.prov, false); assert.equal(s.feil, false);
  assert.match(s.status, /Din tur/);
  assert.match(s.sims, /^1 trekk × [1-3]$/, 'Flue-nivået viser 1-trekk-sjekken over N kandidater');
});

test('en død arbeider startes på nytt av «Prøv igjen», og partiet fortsetter', opts, async () => {
  await startGame();
  await humanMove('e2', 'e4');
  // drep arbeideren midt i trekket: terminate() utløser ingen onerror, så kræsjen rapporteres i tillegg
  await page.eval(`const b = window.fluesjakk.hjerne; b.worker.terminate(); b.worker.onerror({ message: 'arbeideren krasjet' }); true`);
  await page.waitFor(`!document.getElementById('btn-prov').hidden`, { what: 'prøv igjen-knappen' });
  let s = await page.eval(SNAP);
  assert.equal(s.død, true); assert.equal(s.tenker, false);
  await page.click('#btn-prov');
  await page.waitFor(HUMAN_TURN, { what: 'fluas svar etter omstart', timeout: 30000 });
  s = await page.eval(SNAP);
  assert.equal(s.død, false); assert.equal(s.hist.length, 2); assert.equal(s.hist[0], 'e4');
  await humanMove('d2', 'd4');
  await page.waitFor(HUMAN_TURN);
  assert.equal((await page.eval(SNAP)).hist.length, 4, 'den nye arbeideren fortsetter å svare');
  assert.deepEqual(page.errors, []);
});

// ---------------------------------------------------------------- innstillinger
test('innstillingene lagres, styrer visningen og kan nullstilles', opts, async () => {
  await loadPage();
  await page.click('#btn-innstillinger');
  const antall = await page.eval(`document.querySelectorAll('#innstillinger-liste .innstilling-rad').length`);
  assert.ok(antall >= 8, `innstillingslisten har ${antall} valg`);

  await page.eval(`document.querySelector('[data-sett=teknisk]').checked = true;
    document.querySelector('[data-sett=teknisk]').dispatchEvent(new Event('change', { bubbles: true })); true`);
  assert.equal(await page.eval(`document.getElementById('teknisk-panel').hidden`), false, 'teknisk modus viser panelet');
  await page.eval(`document.querySelector('[data-sett=lovligeTrekk]').checked = false;
    document.querySelector('[data-sett=lovligeTrekk]').dispatchEvent(new Event('change', { bubbles: true })); true`);
  const lagret = await page.eval(`JSON.parse(localStorage.getItem('fluesjakk.innstillinger.v1'))`);
  assert.equal(lagret.teknisk, true); assert.equal(lagret.lovligeTrekk, false);

  await page.eval(`document.getElementById('sett-tema').value = 'skog';
    document.getElementById('sett-tema').dispatchEvent(new Event('change', { bubbles: true })); true`);
  assert.equal(await page.eval(`document.documentElement.dataset.tema`), 'skog');

  await page.click('#btn-nullstill');
  const etter = await page.eval(`JSON.parse(localStorage.getItem('fluesjakk.innstillinger.v1') || 'null')`);
  assert.equal(etter, null, 'nullstilling tømmer lagringen');
  assert.equal(await page.eval(`document.documentElement.dataset.tema`), 'kveld');
  assert.equal(await page.eval(`document.getElementById('teknisk-panel').hidden`), true);
  await page.eval(`document.getElementById('innstillinger-dialog').close(), true`);
});

test('tekniske fakta om modellen vises i topplinjen og i teknisk modus', opts, async () => {
  await loadPage();
  const hode = await page.eval(`document.getElementById('specimen').textContent`);
  assert.match(hode, /nevroner/); assert.match(hode, /koblinger/); assert.match(hode, /motor/);
  assert.match(hode, new RegExp(String(models.retina.header.n)));

  await startGame();
  await humanMove('e2', 'e4');
  await page.waitFor(HUMAN_TURN);
  await page.click('#btn-innstillinger');
  await page.eval(`document.querySelector('[data-sett=teknisk]').checked = true;
    document.querySelector('[data-sett=teknisk]').dispatchEvent(new Event('change', { bubbles: true })); true`);
  await page.eval(`document.getElementById('innstillinger-dialog').close(), true`);
  const rader = await page.eval(`Object.fromEntries([...document.querySelectorAll('#teknisk-rutenett dt')].map((dt) => [dt.textContent, dt.nextElementSibling.textContent]))`);
  assert.equal(rader['Nevroner'], String(models.retina.header.n));
  assert.equal(rader['Tidssteg'], String(models.retina.header.steps));
  assert.ok(rader['Valgt trekk'], 'det valgte trekket vises');
  assert.match(rader['Verdi'], /^-?\d/);
  assert.match(rader['Beregningstid'], /ms|s$/, 'beregningstiden vises');
});

// ---------------------------------------------------------------- øyepanelet og tankegjenspillingen
const EYE = `(() => { const a = window.fluesjakk, $ = (i) => document.getElementById(i), oye = a.oye; return {
  ingen: $('oye-panel').classList.contains('eye-none'), skjult: $('oye-panel').hidden, tekst: $('oye-tekst').textContent,
  retina: oye.retina ? oye.retina.n : null, plassert: !!oye.layout && oye.layout.x.length, flueFarge: oye.flyColor,
  drive: oye.drive ? oye.drive.length : null, modus: oye.mode, puls: oye.pulse.squares,
  tips: $('oye-panel').querySelector('.oye-tip').hidden ? null : $('oye-panel').querySelector('.oye-tip').textContent,
  spesOyne: $('spes-oye').hidden ? null : $('spes-oye').textContent,
  lede: $('lede-oye').hidden ? null : $('lede-oye').textContent,
  replay: !$('replay').hidden, steg: $('replay-steg').textContent, spiller: a.viz.playing, pos: a.viz.pos,
  steps: a.viz.steps, trace: a.viz.hasTrace, stripe: !$('strip-canvas').hidden,
  grupper: a.grupper.map((g) => g.name), stripeRader: a.stripe.groups.length, hjerneUnder: $('hjerne-under').textContent }; })()`;

test('øyepanelet viser retinaen fra fluas side, blikket etter hvert trekk, og tankegjenspillingen', opts, async () => {
  await loadPage();
  let e = await page.eval(EYE);
  assert.equal(e.ingen, false); assert.equal(e.retina, 16);
  assert.match(e.spesOyne, /16 fotoreseptorer/);
  assert.match(e.lede, /16/); assert.match(e.lede, /venstre øye ser linjene/);
  assert.ok(e.grupper.includes('retina'), `stripe-diagrammet har en netthinne-rad: ${e.grupper}`);

  await startGame({ color: 'white' });
  await page.waitFor(`!!window.fluesjakk.oye.layout`, { what: 'øylerreting på lerretet' });
  e = await page.eval(EYE);
  assert.equal(e.flueFarge, 'b', 'flua spiller svart: fotoreseptorene ser det speilvendte brettet');
  assert.equal(e.plassert, 16, 'hver fotoreseptor er plassert på lerretet');
  assert.equal(e.drive, null, 'flua har ikke sett ennå');
  assert.match(e.tekst, /Gult er fluas brikker/);

  await humanMove('e2', 'e4');
  await page.waitFor(HUMAN_TURN);
  e = await page.eval(EYE);
  assert.equal(e.drive, 16, 'retina-strømmen fra stillingen flua så på');
  assert.equal(e.puls.length, 2, 'siste trekks ruter pulserer');
  assert.equal(e.replay, true); assert.equal(e.stripe, true); assert.equal(e.trace, true); assert.equal(e.steps, 3);
  assert.ok(e.stripeRader >= 2, 'stripe-rader');
  assert.match(e.hjerneUnder, /3 tidssteg/);

  await page.waitFor(`!window.fluesjakk.viz.playing`, { what: 'gjenspillingen er ferdig' });
  e = await page.eval(EYE);
  assert.equal(e.steg, 'steg 3 / 3'); assert.equal(e.pos, 2);
  await page.eval(`window.fluesjakk.viz.seek(0.5), true`);
  e = await page.eval(EYE);
  assert.equal(e.steg, 'steg 1 / 3'); assert.equal(e.pos, 0.5);
  assert.equal(await page.eval(`document.getElementById('replay-scrub').value`), '0.5', 'skyveknappen følger med');
  await page.click('#replay-play');
  assert.equal((await page.eval(EYE)).spiller, true);
  await page.waitFor(`!window.fluesjakk.viz.playing`, { what: 'gjenspillingen er ferdig' });
  assert.equal((await page.eval(EYE)).steg, 'steg 3 / 3');

  // «føler»: inngangsstrømmen farger fotoreseptorene, og et hover navngir én av dem
  await page.click('[data-oye=foler]');
  e = await page.eval(EYE);
  assert.equal(e.modus, 'foler');
  assert.match(e.tekst, /Inngangsstrømmen til hver fotoreseptor/); assert.match(e.tekst, /trekk 1/);
  // øypanelet ligger nederst på siden, så det må rulles inn i vinduet før musepekeren kan treffe det
  await page.eval(`document.getElementById('oye-panel').scrollIntoView({ block: 'center' }), true`);
  await sleep(150);
  const at = await page.eval(`(() => { const a = window.fluesjakk; const r = a.oye.canvas.getBoundingClientRect(); return { x: r.left + a.oye.layout.x[0], y: r.top + a.oye.layout.y[0] }; })()`);
  await page.send('Input.dispatchMouseEvent', { type: 'mouseMoved', x: at.x, y: at.y });
  await page.waitFor(`!document.querySelector('.oye-tip').hidden`, { what: 'øyeverktøyet' });
  e = await page.eval(EYE);
  assert.match(e.tips, /^R(1-6|7|8) · (venstre|høyre) øye · ser [a-h][1-8] · .* · strøm [-+]?\d/);
  await page.send('Input.dispatchMouseEvent', { type: 'mouseMoved', x: 5, y: 5 });

  // «Se igjen» vurderer stillingen på nytt og spiller av tanken; angre glemmer blikket
  await page.click('#replay-se');
  await page.waitFor(`window.fluesjakk.viz.playing`, { what: 'gjenspilling etter «se igjen»' });
  await page.waitFor(`!window.fluesjakk.viz.playing`);
  await page.click('#btn-angre');
  e = await page.eval(EYE);
  assert.equal(e.drive, null, 'angre tømmer fluas forrige blikk');
  assert.equal((await page.eval(SNAP)).hist.length, 0);
  assert.deepEqual(page.errors, [], 'ingen ubehandlede feil i siden');
});

test('en modell uten retina skjuler øyekartet med en forklaring og spiller likevel av tanken', opts, async () => {
  serve.legacy = true;
  try {
    await loadPage();
    let e = await page.eval(EYE);
    assert.equal(e.ingen, true); assert.equal(e.retina, null);
    assert.match(e.tekst, /ingen retina/);
    assert.equal(e.spesOyne, null); assert.equal(e.lede, null);
    assert.ok(!e.grupper.includes('retina'));
    await startGame({ color: 'white' });
    await humanMove('e2', 'e4');
    await page.waitFor(HUMAN_TURN);
    e = await page.eval(EYE);
    assert.equal(e.drive, null); assert.equal(e.ingen, true);
    assert.equal(e.replay, true); assert.equal(e.trace, true);
    assert.equal((await page.eval(SNAP)).hist.length, 2);
    assert.deepEqual(page.errors, [], 'ingen ubehandlede feil i siden');
  } finally { serve.legacy = false; }
});

// ---------------------------------------------------------------- manglende modell
test('uten modell vises en norsk velkomst med nedlasting, og en feilet nedlasting gir menneskelig tekst', opts, async () => {
  serve.model = false;
  try {
    await page.goto(base);
    await page.waitFor(`!document.getElementById('panel-nedlasting').hidden`, { what: 'nedlastingspanelet' });
    assert.equal(await page.eval(`document.getElementById('panel-oppsett').hidden`), true);
    const tekst = await page.eval(`document.getElementById('panel-nedlasting').textContent`);
    assert.match(tekst, /Velkommen til Fluesjakk/);
    assert.match(tekst, /Last ned modellen/);
    assert.match(tekst, /CC BY-NC 4\.0/, 'lisensen forklares før nedlasting');

    // en nettverksfeil skal bli en menneskelig norsk melding, ikke en stakksporing
    await page.eval(`window.__ekteFetch = window.fetch;
      window.fetch = (u, o) => String(u).includes('huggingface') ? Promise.reject(new Error('nettet er nede')) : window.__ekteFetch(u, o); true`);
    await page.click('#btn-nedlasting');
    await page.waitFor(`!document.getElementById('nedlasting-feil').hidden`, { what: 'nedlastingsfeilen' });
    const feil = await page.eval(`document.getElementById('nedlasting-feil').textContent`);
    assert.match(feil, /Fikk ikke lastet ned modellen/);
    assert.ok(!/Error:|Traceback|undefined/.test(feil), `ingen tekniske detaljer lekker ut: ${feil}`);
    assert.ok(await page.eval(`!!document.getElementById('btn-laste-prov')`), 'feilen tilbyr et nytt forsøk');
    await page.eval(`window.fetch = window.__ekteFetch; true`);
  } finally { serve.model = true; }
});

// ---------------------------------------------------------------- norsk språk
test('ingen engelske rester i grensesnittet', opts, async () => {
  serve.model = true;
  await loadPage();
  const forbudt = /\b(Play|New game|Your move|Thinking|Resign|Undo|Flip|Difficulty|White|Black|Settings|Close|Retry|Draw|Check|Checkmate|Leaderboard|Party|Sound|Load|Download|Cancel|Copy|Share|Score|Player|Moves|Time|Result|Wins?|Loss(es)?|Back|Next|Start|About|How it works)\b/;
  const flater = [
    ['forsiden', `document.getElementById('skjerm-start').innerText`],
    ['innstillingene', `(() => { document.getElementById('innstillinger-dialog').showModal(); const t = document.getElementById('innstillinger-dialog').innerText; document.getElementById('innstillinger-dialog').close(); return t; })()`],
    ['om-dialogen', `(() => { document.getElementById('om-dialog').showModal(); const t = document.getElementById('om-dialog').innerText; document.getElementById('om-dialog').close(); return t; })()`],
    ['slik-siden', `document.getElementById('slik-innhold').innerText`],
    ['feilsøkingen', `document.getElementById('feilsok-innhold').innerText`],
    ['hjelpetekstene', `[...document.querySelectorAll('#vanskelighetskort .kort-tekst')].map((e) => e.textContent).join(' ')`],
  ];
  for (const [navn, expr] of flater) {
    const tekst = await page.eval(expr);
    const treff = tekst.match(new RegExp(forbudt.source, 'g'));
    assert.equal(treff, null, `${navn} inneholder engelske ord: ${treff && [...new Set(treff)].join(', ')}`);
  }
  // og æ, ø og å skal faktisk være til stede — ikke erstattet av ascii
  const alt = await page.eval(`document.getElementById('skjerm-start').innerText + document.getElementById('slik-innhold').innerText`);
  for (const tegn of ['æ', 'ø', 'å']) assert.ok(alt.includes(tegn), `tegnet «${tegn}» mangler i teksten`);
  assert.deepEqual(page.errors, []);
});
