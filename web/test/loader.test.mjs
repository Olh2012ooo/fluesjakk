// node --test web/test/loader.test.mjs
// loadBrain against a fake `fetch` + Cache API: every blob request must be versioned by the header
// and revalidated (`cache: 'no-cache'`), a blob that does not match brain.json (size or sha256) must
// be rejected and never stored, and a poisoned Cache API entry must be discarded and re-downloaded.
import { test, beforeEach, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { gzipSync } from 'node:zlib';

import { loadBrain, blobProblem, versionTag } from '../engine/loader.js';

// ---------------------------------------------------------------------------------------------
// a tiny valid export: one f32 array of 4 numbers (16 bytes) + one i32 array (16 bytes)
function makeExport({ values = [1, 2, 3, 4], exportedAt = '2026-01-01T00:00:00+00:00', sha = true } = {}) {
  const buffer = new ArrayBuffer(32);
  new Float32Array(buffer, 0, 4).set(values);
  new Int32Array(buffer, 16, 4).set([10, 20, 30, 40]);
  const blob = Buffer.from(buffer);
  const gz = gzipSync(blob);
  const header = {
    run_name: 'unit', exported_at: exportedAt, nnz: 4, total_bytes: blob.byteLength, gzip_bytes: gz.byteLength,
    arrays: [
      { name: 'a', dtype: 'f32', shape: [4], offset: 0, length_bytes: 16 },
      { name: 'b', dtype: 'i32', shape: [4], offset: 16, length_bytes: 16 },
    ],
  };
  if (sha) header.blob_sha256 = createHash('sha256').update(blob).digest('hex');
  return { header, blob, gz };
}

/** In-memory static host: `files` maps path -> {body:Buffer|string, type}. Records every request. */
function fakeFetch(files, log) {
  return async (url, init = {}) => {
    const u = new URL(url, 'https://example.test/');
    log.push({ path: u.pathname, search: u.search, method: init.method || 'GET', cache: init.cache });
    const f = files[u.pathname];
    if (!f) return new Response('not found', { status: 404 });
    const headers = { 'content-type': f.type || 'application/octet-stream', 'content-length': String(Buffer.byteLength(f.body)) };
    if (init.method === 'HEAD') return new Response(null, { status: 200, headers });
    return new Response(f.body, { status: 200, headers });
  };
}

/** Minimal Cache API (single named cache, keyed by URL string). */
function fakeCaches() {
  const store = new Map();
  const cache = {
    async match(key) { return store.has(key) ? new Response(store.get(key)) : undefined; },
    async put(key, resp) { store.set(key, Buffer.from(await resp.arrayBuffer())); },
    async keys() { return [...store.keys()].map((url) => ({ url })); },
    async delete(req) { return store.delete(typeof req === 'string' ? req : req.url); },
  };
  return { store, caches: { open: async () => cache } };
}

const BASE = 'https://example.test/model/';
let saved;
beforeEach(() => { saved = { fetch: globalThis.fetch, caches: globalThis.caches }; });
afterEach(() => { globalThis.fetch = saved.fetch; if (saved.caches === undefined) delete globalThis.caches; else globalThis.caches = saved.caches; });

function site(exp, { gz = true } = {}) {
  const files = {
    '/model/brain.json': { body: JSON.stringify(exp.header), type: 'application/json' },
    '/model/brain.flyb': { body: exp.blob },
  };
  if (gz) files['/model/brain.flyb.gz'] = { body: exp.gz, type: 'application/gzip' };
  return files;
}

// ---------------------------------------------------------------------------------------------
test('loadBrain versions every blob request by the header and revalidates it', async () => {
  const exp = makeExport();
  const log = [];
  globalThis.fetch = fakeFetch(site(exp), log);
  const { caches, store } = fakeCaches();
  globalThis.caches = caches;

  const r1 = await loadBrain(BASE);
  assert.equal(r1.fromCache, false);
  assert.deepEqual(Array.from(r1.arrays.a), [1, 2, 3, 4]);
  assert.deepEqual(Array.from(r1.arrays.b), [10, 20, 30, 40]);
  const blobReqs = log.filter((l) => l.path !== '/model/brain.json');
  assert.ok(blobReqs.length >= 1);
  const tag = encodeURIComponent(versionTag(exp.header));
  for (const l of blobReqs) {
    assert.equal(l.search, `?v=${tag}`, `${l.path} must carry the version tag`);
    assert.equal(l.cache, 'no-cache', `${l.path} must be revalidated`);
  }
  assert.equal(log[0].path, '/model/brain.json');
  assert.equal(log[0].cache, 'no-cache');
  // stored under the versioned key, and the second load is served from the cache
  assert.equal(store.size, 1);
  assert.ok([...store.keys()][0].includes(`?v=${tag}`));
  const r2 = await loadBrain(BASE);
  assert.equal(r2.fromCache, true);
  assert.deepEqual(Array.from(r2.arrays.a), [1, 2, 3, 4]);
});

test('a re-export changes the version tag, so the new blob is fetched with a different URL', async () => {
  const a = makeExport({ values: [1, 2, 3, 4], exportedAt: '2026-01-01T00:00:00+00:00' });
  const b = makeExport({ values: [5, 6, 7, 8], exportedAt: '2026-01-02T00:00:00+00:00' });
  assert.notEqual(versionTag(a.header), versionTag(b.header));
  const { caches, store } = fakeCaches();
  globalThis.caches = caches;
  const log = [];
  globalThis.fetch = fakeFetch(site(a), log);
  await loadBrain(BASE);
  globalThis.fetch = fakeFetch(site(b), log);
  const r = await loadBrain(BASE);
  assert.equal(r.fromCache, false);
  assert.deepEqual(Array.from(r.arrays.a), [5, 6, 7, 8]);
  const searches = new Set(log.filter((l) => l.path.endsWith('.gz')).map((l) => l.search));
  assert.equal(searches.size, 2, 'the two exports must not share a blob URL');
  assert.equal(store.size, 1, 'the old brain is evicted from the Cache API');
});

test('header/blob size mismatch rejects and is not cached', async () => {
  const good = makeExport({ sha: false });
  // a same-header, longer blob (what a stale HTTP cache would hand back after a layout change)
  const stale = Buffer.concat([good.blob, Buffer.alloc(16)]);
  const files = site(good, { gz: false });
  files['/model/brain.flyb'] = { body: stale };
  globalThis.fetch = fakeFetch(files, []);
  const { caches, store } = fakeCaches();
  globalThis.caches = caches;
  await assert.rejects(loadBrain(BASE), /48 bytes but brain\.json says 32.*out of sync/);
  assert.equal(store.size, 0, 'a mismatched blob must never be stored in the Cache API');
});

test('same-size stale blob is caught by blob_sha256 and not cached', async () => {
  const good = makeExport({ values: [1, 2, 3, 4] });
  const other = makeExport({ values: [9, 9, 9, 9] });
  const files = site(good);
  files['/model/brain.flyb.gz'] = { body: other.gz, type: 'application/gzip' };
  files['/model/brain.flyb'] = { body: other.blob };
  globalThis.fetch = fakeFetch(files, []);
  const { caches, store } = fakeCaches();
  globalThis.caches = caches;
  await assert.rejects(loadBrain(BASE), /sha256.*does not match.*out of sync/);
  assert.equal(store.size, 0);
});

test('a poisoned Cache API entry is discarded, re-downloaded and replaced', async () => {
  const exp = makeExport();
  const log = [];
  globalThis.fetch = fakeFetch(site(exp), log);
  const { caches, store } = fakeCaches();
  globalThis.caches = caches;
  // poison: the right key holding a blob of the wrong size (padded to 8, as the loader stores it)
  const key = `${BASE}brain.flyb?v=${encodeURIComponent(versionTag(exp.header))}`;
  store.set(key, Buffer.alloc(48));
  const r = await loadBrain(BASE);
  assert.equal(r.fromCache, false);
  assert.deepEqual(Array.from(r.arrays.a), [1, 2, 3, 4]);
  assert.ok(log.some((l) => l.path !== '/model/brain.json'), 'the blob was fetched from the network');
  assert.equal(store.size, 1);
  assert.equal(store.get(key).byteLength, 32, 'the cache now holds the verified blob');
  // and a same-size poisoned entry is caught by the digest
  store.set(key, Buffer.from(makeExport({ values: [9, 9, 9, 9] }).blob));
  const r2 = await loadBrain(BASE);
  assert.equal(r2.fromCache, false);
  assert.deepEqual(Array.from(r2.arrays.a), [1, 2, 3, 4]);
});

test('blobProblem tolerates alignment padding on cached blobs only', async () => {
  const exp = makeExport();
  const raw = exp.blob.buffer.slice(exp.blob.byteOffset, exp.blob.byteOffset + 32);
  assert.equal(await blobProblem(exp.header, raw), null);
  assert.equal(await blobProblem(exp.header, raw, { padded: true }), null);
  // an unaligned export (29 bytes) is stored padded to 32
  const h29 = { total_bytes: 29 };
  assert.equal(await blobProblem(h29, new ArrayBuffer(29)), null);
  assert.equal(await blobProblem(h29, new ArrayBuffer(32), { padded: true }), null);
  assert.match(await blobProblem(h29, new ArrayBuffer(32)), /32 bytes but brain\.json says 29/);
  assert.match(await blobProblem(h29, new ArrayBuffer(40), { padded: true }), /40 bytes/);
  // the digest is computed over the unpadded bytes
  const padded = new ArrayBuffer(40);
  new Uint8Array(padded).set(exp.blob);
  assert.equal(await blobProblem({ ...exp.header, total_bytes: 32 }, padded.slice(0, 32), { padded: true }), null);
  // headers without total_bytes / blob_sha256 (older exports) are not checked
  assert.equal(await blobProblem({ arrays: [] }, new ArrayBuffer(5)), null);
});
