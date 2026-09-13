// Browser regression tests. Drives index.html in headless Chrome (the installed
// Google Chrome via Playwright's "chrome" channel, falling back to Playwright's
// own Chromium if that is unavailable). Run with `pnpm test`.
// D3 loads from cdnjs, so the tests need network access.
import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { chromium } from 'playwright';

const here = path.dirname(fileURLToPath(import.meta.url));
const PAGE = pathToFileURL(path.join(here, '..', 'index.html')).href;
let browser;

before(async () => {
  try { browser = await chromium.launch({ channel: 'chrome', headless: true }); }
  catch { browser = await chromium.launch({ headless: true }); }
});
after(async () => { if (browser) await browser.close(); });

/** Open the page; collect page errors and console errors. Web fonts are stubbed so font outages do not fail the run. */
async function open({ width = 1440, reduced = true } = {}) {
  const context = await browser.newContext({ viewport: { width, height: 900 }, reducedMotion: reduced ? 'reduce' : 'no-preference' });
  await context.route(/fonts\.(googleapis|gstatic)\.com/, (route) => route.fulfill({ status: 200, contentType: 'text/css', body: '' }));
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', (e) => errors.push('pageerror: ' + e.message));
  page.on('console', (m) => { if (m.type() === 'error') errors.push('console: ' + m.text()); });
  await page.goto(PAGE);
  await page.waitForSelector('#fit-table tbody tr');
  return { page, context, errors };
}

/** Fraction of sampled pixels on a canvas that are not fully transparent. */
const painted = (page, sel) => page.$eval(sel, (c) => {
  const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
  let n = 0, t = 0;
  for (let i = 3; i < d.length; i += 4 * 37) { t++; if (d[i]) n++; }
  return n / t;
});

for (const width of [1440, 390]) {
  test(`page renders every chapter without errors or horizontal overflow at ${width}px`, async () => {
    const { page, context, errors } = await open({ width });
    assert.equal(await page.$$eval('section.chapter', (els) => els.length), 8);
    assert.equal(await page.$$eval('#scatter circle', (els) => els.length), 22, 'one dot per rentable chip');
    assert.equal(await page.$$eval('.chip:not(.ghost)', (els) => els.length), 22, 'one card per rentable chip');
    // every chip gets a hero tile, and every tile sits inside the canvas
    const tiles = await page.$eval('#hero-canvas', (c) => ({ n: c.__tiles.length, w: c.getBoundingClientRect().width, h: c.getBoundingClientRect().height, out: c.__tiles.filter((t) => t.x < 0 || t.y < 0 || t.x + t.s > c.getBoundingClientRect().width || t.y + t.s > c.getBoundingClientRect().height).map((t) => t.ch.id) }));
    assert.equal(tiles.n, 22);
    assert.deepEqual(tiles.out, [], `hero tiles outside the ${tiles.w}x${tiles.h} canvas`);
    assert.equal(await page.$$eval('.chip.ghost', (els) => els.length), 2, 'TPU 8t and 8i shown as announced');
    for (const sel of ['#hero-canvas', '#anat-gpu', '#anat-tpu']) assert.ok((await painted(page, sel)) > 0.2, `${sel} is blank at rest`);
    assert.equal(await page.$$eval('#topos canvas', (els) => els.length), 4);
    // scroll through the page so lazy layout and the TOC observer run
    await page.evaluate(async () => { for (let y = 0; y < document.body.scrollHeight; y += 600) { window.scrollTo(0, y); await new Promise((r) => setTimeout(r, 25)); } });
    await page.waitForTimeout(300);
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
    assert.equal(overflow, 0, 'page scrolls horizontally');
    assert.deepEqual(errors, []);
    await context.close();
  });
}

test('clicking a lineup card opens its spec sheet and clicking again closes it', async () => {
  const { page, context, errors } = await open();
  const card = page.locator('.chip[data-id="tpu7x"]');
  await card.click();
  assert.equal(await card.getAttribute('aria-pressed'), 'true');
  assert.match(await page.locator('#detail h3').innerText(), /Ironwood/);
  assert.equal(await page.$$eval('#detail-bars rect', (els) => els.length), 10, 'five bars, each with a track');
  await card.click();
  assert.ok(await page.$eval('#detail', (el) => el.hidden));
  assert.deepEqual(errors, []);
  await context.close();
});

test('scatter precision toggle relabels the axis and keeps every dot', async () => {
  const { page, context, errors } = await open();
  await page.locator('[data-prec="bf16"]').click();
  assert.equal(await page.locator('[data-prec="bf16"]').getAttribute('aria-pressed'), 'true');
  assert.equal(await page.locator('[data-prec="fp8"]').getAttribute('aria-pressed'), 'false');
  await page.waitForTimeout(600);
  assert.match(await page.$eval('#scatter', (s) => s.textContent), /BF16/);
  assert.equal(await page.$$eval('#scatter circle', (els) => els.filter((c) => +c.getAttribute('r') > 0).length), 22);
  assert.deepEqual(errors, []);
  await context.close();
});

test('fit calculator: 70B at FP8 with 40% headroom needs 98 GB, two H100s, one H200; training multiplies by 16', async () => {
  const { page, context, errors } = await open();
  const row = (name) => page.$$eval('#fit-table tbody tr', (trs, n) => { const tr = trs.find((t) => t.children[0].textContent.trim() === n); return [...tr.children].map((td) => td.textContent.trim()); }, name);
  assert.equal(await page.locator('#fit-need').innerText(), '98 GB');
  assert.equal((await row('H100 (High)'))[2], '2');
  assert.equal((await row('H200'))[2], '1');
  assert.equal((await row('T4'))[3].startsWith('no'), true, 'seven T4s exceed the four-GPU N1 limit');
  assert.match((await row('B200'))[4], /\*$/, 'B200 price is marked indicative');
  assert.doesNotMatch(await page.locator('#fit-cheap').innerText(), /B200|H200|GB200|GB300|Mega/, 'chips not sold on demand never win cheapest-on-demand');
  await page.selectOption('#fit-mode', 'train');
  assert.equal(await page.locator('#fit-need').innerText(), '1.57 TB');
  assert.equal((await row('B200'))[2], '10');
  assert.deepEqual(errors, []);
  await context.close();
});

test('roofline: the verdict follows the chip and the intensity slider', async () => {
  const { page, context, errors } = await open();
  assert.match(await page.locator('#roof-verdict').innerText(), /H200.*memory-bound/s);
  await page.locator('#roof-ai').fill('3.5'); // about 3,162 FLOP/byte
  assert.match(await page.locator('#roof-verdict').innerText(), /compute-bound/);
  await page.selectOption('#roof-chip', 'v5e');
  assert.match(await page.locator('#roof-verdict').innerText(), /TPU v5e.*ridge at about 480 FLOP\/byte/s);
  assert.deepEqual(errors, []);
  await context.close();
});

test('fit calculator: a 1B model at FP8 fits one T4 and the low cost renders with cents', async () => {
  const { page, context, errors } = await open();
  await page.locator('#fit-params').fill('0');
  assert.equal(await page.locator('#fit-need').innerText(), '1 GB');
  const t4 = await page.$$eval('#fit-table tbody tr', (trs) => [...trs.find((t) => t.children[0].textContent.trim() === 'T4').children].map((td) => td.textContent.trim()));
  assert.equal(t4[2], '1');
  assert.equal(t4[4], '$0.35');
  assert.match(await page.locator('#fit-cheap').innerText(), /^T4 /);
  assert.deepEqual(errors, []);
  await context.close();
});

test('data invariants: every chip record is complete and the host specs match the machine-type tables', async () => {
  const { page, context, errors } = await open();
  const chips = await page.evaluate(() => CHIPS.map((c) => ({ ...c })));
  assert.equal(chips.length, 22);
  for (const c of chips) {
    for (const k of ['mem', 'bw', 'fp8', 'bf16', 'domain', 'year']) assert.ok(Number.isFinite(c[k]) && c[k] > 0, `${c.id}.${k}`);
    const hostKnown = !['v2', 'v3', 'v4'].includes(c.id); // Google publishes no VM shape for these
    for (const k of ['vcpu', 'ram', 'net']) assert.equal(Number.isFinite(c[k]) && c[k] > 0, hostKnown, `${c.id}.${k}`);
    assert.ok(['FP8', 'INT8', 'FP16', 'BF16'].includes(c.peakBasis), `${c.id}.peakBasis`);
    assert.equal(typeof c.onDemand, 'boolean', `${c.id}.onDemand`);
    assert.ok(c.priceBasis, `${c.id}.priceBasis`);
    if (c.onDemand) assert.ok(c.price != null, `${c.id} is on demand but has no price`);
    assert.ok(c.fp8 >= c.bf16, `${c.id}: low-precision peak below BF16 peak`);
  }
  // spot checks against docs.cloud.google.com/compute/docs/gpus (read 12 Sep 2026)
  const by = Object.fromEntries(chips.map((c) => [c.id, c]));
  assert.deepEqual([by.h100m.vcpu, by.h100m.ram, by.h100m.net], [208, 1872, 1800]);
  assert.deepEqual([by.h100.vcpu, by.h100.ram, by.h100.net], [208, 1872, 1000]);
  assert.deepEqual([by.h100e.vcpu, by.h100e.ram, by.h100e.net], [208, 1872, 400]);
  assert.deepEqual([by.h200.vcpu, by.h200.ram, by.h200.net], [224, 2952, 3600]);
  assert.deepEqual([by.b200.vcpu, by.b200.ram, by.b200.net], [224, 3968, 3600]);
  assert.deepEqual([by.gb200.vcpu, by.gb200.ram, by.gb200.net], [140, 884, 2000]);
  assert.deepEqual([by.gb300.vcpu, by.gb300.ram, by.gb300.net], [144, 960, 3600]);
  // TPU hosts from docs.cloud.google.com/tpu/docs/{v5e,v5p,v6e,tpu7x} (read 13 Sep 2026): per-VM NIC, largest VM shape
  assert.deepEqual([by.v5e.vcpu, by.v5e.ram, by.v5e.net], [224, 384, 200]);
  assert.deepEqual([by.v5p.vcpu, by.v5p.ram, by.v5p.net], [208, 448, 200]);
  assert.deepEqual([by.v6e.vcpu, by.v6e.ram, by.v6e.net], [360, 1440, 800]);
  assert.deepEqual([by.tpu7x.vcpu, by.tpu7x.ram, by.tpu7x.net], [224, 960, 400]);
  for (const id of ['gb300', 'gb200', 'b200', 'h200', 'h100m']) assert.equal(by[id].onDemand, false, `${id} is not sold on demand`);
  assert.deepEqual(errors, []);
  await context.close();
});

test('spec sheet says when Google publishes no host shape for a generation', async () => {
  const { page, context, errors } = await open();
  await page.locator('.chip[data-id="v4"]').click();
  assert.match(await page.locator('#detail').innerText(), /not published for this generation/);
  await page.locator('.chip[data-id="v4"]').click();
  await page.locator('.chip[data-id="v5p"]').click();
  assert.match(await page.locator('#detail').innerText(), /208 vCPU · 448 GB RAM · 200 Gbps host network/);
  assert.deepEqual(errors, []);
  await context.close();
});
