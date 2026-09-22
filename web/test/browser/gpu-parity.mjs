// node web/test/browser/gpu-parity.mjs   (not part of `node --test`: needs a Chromium with WebGPU
// and the real model in web/model/). Serves web/, launches headless Chromium with WebGPU on the
// hardware adapter, opens test/browser/gpu-parity.html and checks FlyBrainGPU against FlyBrain:
//   |Δlogit| <= 1e-2 on every policy logit, |Δvalue| <= 1e-2, same legal argmax, on 5 positions,
// plus the activity trace and the retina drive (vision blobs), then prints the measured latencies.
// FLY_MODEL_DIR=<dir> serves another export (e.g. a scratch fly3-style blob) instead of web/model/;
// FLY_VECTORS=<model.json> (default tests/vectors/model.json when it belongs to the served blob)
// additionally compares the real WebGPU engine with the Python reference vectors (SPEC §8, top-K
// legal logits + value + argmax). Exit code 1 on a mismatch, 2 when WebGPU could not start.
//
// Chromium is found like web/test/app.test.mjs (CHROME=/path/to/chrome overrides). Flags that make
// WebGPU work headless on Linux (Chromium 149, NVIDIA/AMD via Vulkan):
//   --headless=new --enable-unsafe-webgpu --ignore-gpu-blocklist --enable-features=Vulkan --use-angle=vulkan
// (do NOT pass --disable-gpu / --use-angle=swiftshader-webgl; SwiftShader can be forced with
//  GPU_ANGLE=swiftshader for a software run). The page must be served over http: navigator.gpu is
// not exposed on about:blank.
import { spawn } from 'node:child_process';
import { createServer } from 'node:http';
import { existsSync, readdirSync, readFileSync, statSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir, homedir } from 'node:os';
import { dirname, join, extname, resolve, basename } from 'node:path';
import { fileURLToPath } from 'node:url';

const WEB = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const MODEL_DIR = process.env.FLY_MODEL_DIR ? resolve(process.env.FLY_MODEL_DIR) : join(WEB, 'model');
const VECTORS = process.env.FLY_VECTORS ? resolve(process.env.FLY_VECTORS) : join(WEB, '..', 'tests', 'vectors', 'model.json');
const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.css': 'text/css', '.json': 'application/json', '.flyb': 'application/octet-stream', '.gz': 'application/octet-stream' };
const TOL = 1e-2;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function findChrome() {
  const cands = [process.env.CHROME, process.env.CHROME_PATH, '/usr/bin/chromium', '/usr/bin/chromium-browser', '/usr/bin/google-chrome', '/usr/bin/google-chrome-stable'];
  const pw = join(homedir(), '.cache', 'ms-playwright');
  if (existsSync(pw)) for (const d of readdirSync(pw).sort().reverse()) if (d.startsWith('chromium-')) cands.push(join(pw, d, 'chrome-linux64', 'chrome'), join(pw, d, 'chrome-linux', 'chrome'));
  return cands.find((c) => c && existsSync(c)) || null;
}

if (!existsSync(join(MODEL_DIR, 'brain.json'))) { console.error(`no ${join(MODEL_DIR, 'brain.json')} — run \`fly export-web\` first (or set FLY_MODEL_DIR)`); process.exit(2); }
// the vectors are served only when they were generated from the served blob
let vectorsOk = false;
if (existsSync(VECTORS)) {
  const hdr = JSON.parse(readFileSync(join(MODEL_DIR, 'brain.json'), 'utf8'));
  const vec = JSON.parse(readFileSync(VECTORS, 'utf8'));
  vectorsOk = !vec.blob_sha256 || vec.blob_sha256 === hdr.blob_sha256;
  if (!vectorsOk) console.log(`vectors ${VECTORS} belong to ${vec.run_name}@${vec.exported_at}, not to the served ${hdr.run_name}@${hdr.exported_at}: Python comparison skipped`);
} else if (process.env.FLY_VECTORS) { console.error(`no vectors at ${VECTORS}`); process.exit(2); }
console.log(`model: ${MODEL_DIR}${vectorsOk ? `\nvectors: ${VECTORS}` : ''}`);
const CHROME = findChrome();
if (!CHROME) { console.error('no Chromium found (set CHROME=/path/to/chrome)'); process.exit(2); }

const server = createServer((req, res) => {
  const path = decodeURIComponent(new URL(req.url, 'http://x').pathname);
  if (path === '/vectors.json') {
    if (!vectorsOk) { res.writeHead(404); res.end(); return; }
    res.writeHead(200, { 'content-type': MIME['.json'] }); res.end(readFileSync(VECTORS)); return;
  }
  const file = path.startsWith('/model/') ? join(MODEL_DIR, basename(path)) : join(WEB, path === '/' ? 'index.html' : path);
  if (path.startsWith('/model/') && !file.startsWith(MODEL_DIR)) { res.writeHead(404); res.end(); return; }
  if (!(file.startsWith(WEB) || file.startsWith(MODEL_DIR)) || !existsSync(file) || statSync(file).isDirectory()) { res.writeHead(404); res.end(); return; }
  if (req.method === 'HEAD') { res.writeHead(200, { 'content-type': MIME[extname(file)] || 'application/octet-stream', 'content-length': statSync(file).size }); res.end(); return; }
  res.writeHead(200, { 'content-type': MIME[extname(file)] || 'application/octet-stream' });
  res.end(readFileSync(file));
});

class Page {
  constructor(ws) {
    this.ws = ws; this.id = 0; this.pending = new Map(); this.console = [];
    ws.onmessage = (ev) => {
      const m = JSON.parse(ev.data);
      if (m.id && this.pending.has(m.id)) { this.pending.get(m.id)(m); this.pending.delete(m.id); }
      else if (m.method === 'Runtime.consoleAPICalled') this.console.push(m.params.args.map((a) => a.value ?? a.description).join(' '));
      else if (m.method === 'Runtime.exceptionThrown') this.console.push('EXCEPTION ' + (m.params.exceptionDetails?.exception?.description || m.params.exceptionDetails?.text));
    };
  }
  send(method, params = {}) { return new Promise((r) => { const i = ++this.id; this.pending.set(i, r); this.ws.send(JSON.stringify({ id: i, method, params })); }); }
  async eval(expr) {
    const m = await this.send('Runtime.evaluate', { expression: expr, returnByValue: true, awaitPromise: true });
    if (m.result?.exceptionDetails) throw new Error(`page eval failed: ${m.result.exceptionDetails.exception?.description || m.result.exceptionDetails.text}`);
    return m.result?.result?.value;
  }
}

await new Promise((r) => server.listen(0, '127.0.0.1', r));
const base = `http://127.0.0.1:${server.address().port}/`;
const profile = mkdtempSync(join(tmpdir(), 'flychess-gpu-'));
const port = 9400 + Math.floor(Math.random() * 400);
const angle = process.env.GPU_ANGLE || 'vulkan';
const flags = ['--headless=new', '--no-sandbox', '--enable-unsafe-webgpu', '--ignore-gpu-blocklist', '--enable-features=Vulkan', `--use-angle=${angle}`,
  ...(angle === 'swiftshader' ? ['--enable-unsafe-swiftshader'] : []),
  '--hide-scrollbars', `--remote-debugging-port=${port}`, `--user-data-dir=${profile}`, 'about:blank'];
console.log(`chromium: ${CHROME}\nflags: ${flags.slice(0, -3).join(' ')}`);
const chrome = spawn(CHROME, flags, { stdio: 'ignore' });
let code = 0;
try {
  let targets = null;
  for (let i = 0; i < 100 && !targets; i++) { try { targets = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json(); } catch { await sleep(100); } }
  if (!targets) throw new Error('chromium did not start');
  const ws = new WebSocket(targets.find((t) => t.type === 'page').webSocketDebuggerUrl);
  await new Promise((r, j) => { ws.onopen = r; ws.onerror = j; });
  const page = new Page(ws);
  await page.send('Runtime.enable'); await page.send('Page.enable');
  await page.send('Page.navigate', { url: base + 'test/browser/gpu-parity.html' });
  for (let i = 0; i < 200 && !(await page.eval('typeof window.gpuParity === "function"')); i++) await sleep(50);
  const gpuAvail = await page.eval('(async () => { if (!navigator.gpu) return "navigator.gpu undefined"; for (let i = 0; i < 4; i++) { const a = await navigator.gpu.requestAdapter(); if (a) return "ok"; await new Promise((r) => setTimeout(r, 300)); } return "requestAdapter() returned null"; })()');
  if (gpuAvail !== 'ok') { console.error(`WebGPU unavailable in headless Chromium: ${gpuAvail}`); code = 2; }
  else {
    const reps = Number(process.env.REPS || 30);
    const r = await page.eval(`window.gpuParity({ reps: ${reps}, emulate: ${process.env.NO_EMULATE ? 'false' : 'true'} })`);
    for (const line of page.console) console.log('  ' + line);
    console.log('\nadapter:', JSON.stringify(r.adapterInfo), 'limits:', JSON.stringify(r.limits), 'timestamps:', r.timestamps);
    const bad = r.positions.filter((p) => !(p.dLogit <= TOL && p.dValue <= TOL && p.sameArgmax && p.dTrace <= TOL && p.dRetina <= TOL));
    const worst = (k) => Math.max(...r.positions.map((p) => p[k] ?? 0));
    console.log(`\nfeatures: ${JSON.stringify(r.features)}`);
    console.log(`worst over ${r.positions.length} positions: |Δlogit| ${worst('dLogit').toExponential(2)}  |Δvalue| ${worst('dValue').toExponential(2)}  |Δactivity| ${worst('dActivity').toExponential(2)}  |Δtrace| ${worst('dTrace').toExponential(2)}  |Δretina| ${worst('dRetina').toExponential(2)}` + (r.positions[0].dLogitEmu !== undefined ? `  (GPU vs f32 emulation: |Δlogit| ${worst('dLogitEmu').toExponential(2)}, |Δactivity| ${worst('dActivityEmu').toExponential(2)})` : ''));
    const t = r.timing;
    console.log(`GPU forward median ${t.gpuForwardMs.median.toFixed(2)} ms (no activity readback ${t.gpuForwardNoActivityMs.median.toFixed(2)} ms; with trace ${t.gpuForwardTraceMs.median.toFixed(2)} ms; step ${t.gpuStepMs.toFixed(3)} ms) — JS forward median ${t.jsForwardMs.median.toFixed(1)} ms (with trace ${t.jsForwardTraceMs.median.toFixed(1)} ms) — ${(t.jsForwardMs.median / t.gpuForwardMs.median).toFixed(1)}x`);
    if (bad.length) { console.error(`\nPARITY FAILURE on: ${bad.map((p) => p.name).join(', ')} (tolerance ${TOL})`); code = 1; }
    else console.log(`\nPARITY OK (|Δlogit| <= ${TOL}, |Δvalue| <= ${TOL}, same argmax, trace + retina drive) on ${r.positions.length} positions`);
    if (r.python) {
      const py = r.python;
      const badPy = py.positions.filter((p) => !(p.gpu.dLogit <= TOL && p.gpu.dValue <= TOL && p.gpu.sameArgmax && p.js.dLogit <= TOL && p.js.dValue <= TOL && p.js.sameArgmax));
      console.log(`\nPython vectors (${py.runName}, ${py.positions.length} positions): WebGPU max |Δlogit| ${py.gpu.logit.toExponential(2)} |Δvalue| ${py.gpu.value.toExponential(2)}; JS max |Δlogit| ${py.js.logit.toExponential(2)} |Δvalue| ${py.js.value.toExponential(2)}`);
      if (badPy.length) { console.error(`PYTHON PARITY FAILURE on: ${badPy.map((p) => p.name).join(', ')}`); code = 1; }
      else console.log(`PYTHON PARITY OK for both engines (tolerance ${TOL})`);
    }

    // --- the worker: backend report, moves, bench, device loss → JS fallback
    console.log('\nworker (engine/worker.js on the real model):');
    page.console.length = 0;
    await page.send('Page.navigate', { url: base + 'test/browser/worker-gpu.html' });
    for (let i = 0; i < 200 && !(await page.eval('typeof window.workerGpu === "function"')); i++) await sleep(50);
    const w = await page.eval('window.workerGpu()');
    for (const line of page.console) console.log('  ' + line);
    const problems = [];
    const SIMS = { webgpu: 400, js: 40 };   // worker.js DIFFICULTY.superfly.sims
    if (w.readyBackend !== 'webgpu') problems.push(`ready.backend = ${w.readyBackend} (${JSON.stringify(w.gpuInfo)})`);
    if (!w.features || typeof w.features.vision !== 'boolean') problems.push(`ready.features missing: ${JSON.stringify(w.features)}`);
    if (w.features?.vision && !(w.retina && w.retina.n > 0 && w.retina.uv === 2 * w.retina.n && w.retina.legend >= 1)) problems.push(`ready.retina metadata incomplete: ${JSON.stringify(w.retina)}`);
    if (!w.features?.vision && w.retina !== null) problems.push(`ready.retina should be null without vision`);
    for (const [d, m] of Object.entries(w.moves)) {
      if (!m.move) problems.push(`${d}: no move`); if (m.backend !== 'webgpu') problems.push(`${d}: backend ${m.backend}`); if (m.sample !== w.sampleN) problems.push(`${d}: activity sample ${m.sample}`);
      if (m.trace !== null) problems.push(`${d}: a move without trace:true carried a trace`);
      if (w.features?.vision ? m.retinaDrive !== w.retina.n : m.retinaDrive !== null) problems.push(`${d}: retinaDrive length ${m.retinaDrive}`);
    }
    if (w.moves.superfly.sims !== SIMS.webgpu || w.thinkingEvents < SIMS.webgpu / 10 - 1) problems.push(`superfly: sims ${w.moves.superfly.sims}, thinking events ${w.thinkingEvents}`);
    if (w.tracedMove.trace !== w.steps * w.sampleN || !w.tracedMove.move) problems.push(`move with trace:true: ${JSON.stringify(w.tracedMove)}`);
    if (w.eval.trace !== w.steps * w.sampleN || w.eval.traceSteps !== w.steps || w.eval.sample !== w.sampleN) problems.push(`eval trace: ${JSON.stringify(w.eval)}`);
    if (!(w.eval.traceLastVsSample <= 1e-6)) problems.push(`eval: last trace row differs from activitySample by ${w.eval.traceLastVsSample}`);
    if (!w.evalAgree) problems.push(`eval: webgpu and js workers disagree (${JSON.stringify(w.eval)} vs ${JSON.stringify(w.evalJs)})`);
    if (!(w.evalJs.dTrace <= TOL)) problems.push(`eval: webgpu and js traces differ by ${w.evalJs.dTrace}`);
    if (w.evalJs.readyBackend !== 'js') problems.push(`gpu:false load reported ${w.evalJs.readyBackend}`);
    if (w.afterLose.backend !== 'js' || !w.afterLose.move || !w.afterLose.backendMsg) problems.push(`after device loss: ${JSON.stringify(w.afterLose)}`);
    if (w.afterLoseSuperfly.backend !== 'js' || w.afterLoseSuperfly.sims !== SIMS.js) problems.push(`superfly after device loss: ${JSON.stringify(w.afterLoseSuperfly)}`);
    if (problems.length) { console.error('\nWORKER FAILURE:\n  ' + problems.join('\n  ')); code = 1; }
    else console.log(`\nWORKER OK: webgpu backend for larva/fly/superfly (superfly ${w.moves.superfly.thinkMs.toFixed(0)} ms for ${w.moves.superfly.sims} sims), bench webgpu ${w.bench.webgpu.forwardMs?.toFixed(2)} ms (trace ${w.bench.webgpuTrace?.forwardMs?.toFixed(2)} ms) vs js ${w.bench.js.forwardMs?.toFixed(1)} ms, eval trace ${w.steps}×${w.sampleN}, device loss → js fallback`);
  }
  ws.close();
} catch (err) {
  console.error(err.stack || String(err));
  code = code || 1;
} finally {
  chrome.kill('SIGKILL'); server.close();
  await sleep(200); rmSync(profile, { recursive: true, force: true });
}
process.exit(code);
