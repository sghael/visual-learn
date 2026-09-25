// Browser regression tests. Drives index.html in headless Chrome (the installed
// Google Chrome via Playwright's "chrome" channel, falling back to Playwright's
// own Chromium if that is unavailable). Run with `pnpm test`.
// The page has no runtime dependencies; web fonts are stubbed so a font outage cannot fail the run.
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

/** Open the page; collect page errors and console errors. */
async function open({ width = 1440 } = {}) {
  const context = await browser.newContext({ viewport: { width, height: 900 } });
  await context.route(/fonts\.(googleapis|gstatic)\.com/, (route) => route.fulfill({ status: 200, contentType: 'text/css', body: '' }));
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', (e) => errors.push('pageerror: ' + e.message));
  page.on('console', (m) => { if (m.type() === 'error') errors.push('console: ' + m.text()); });
  await page.goto(PAGE);
  await page.waitForSelector('#fit-table tbody tr');
  return { page, context, errors };
}

/** Cells of the fit-table row whose first cell is `name`. Columns: chip, chips, $/hr, bar, memory, fast domain. */
const fitRow = (page, name) => page.$$eval('#fit-table tbody tr', (trs, n) => {
  const tr = trs.find((t) => t.children[0].textContent.trim() === n);
  return [...tr.children].map((td) => td.textContent.trim());
}, name);

for (const width of [1440, 390]) {
  test(`page renders every section and figure without errors or horizontal overflow at ${width}px`, async () => {
    const { page, context, errors } = await open({ width });
    assert.equal(await page.$$eval('main > section', (els) => els.length), 7);
    assert.equal(await page.$$eval('.topbar nav a', (els) => els.length), 7);
    assert.equal(await page.$$eval('canvas', (els) => els.length), 0, 'no canvas animations remain');
    // Figure 1: 132 SMs drawn, and a TensorCore with its MXUs
    assert.ok(await page.$$eval('#anat-gpu rect', (els) => els.length) >= 132 + 8);
    assert.ok(await page.$$eval('#anat-tpu rect', (els) => els.length) >= 5);
    // Figure 2: busy-cell counts match the captions (cells with i + j <= t - 1 in a 6 × 6 array)
    assert.deepEqual(await page.$$eval('#systolic > div', (ds) => ds.map((d) => [+d.dataset.cycle, d.querySelectorAll('rect.busy').length])), [[1, 1], [4, 10], [11, 36]]);
    // Figure 3: one row per rentable chip, grouped by family; details start closed
    assert.equal(await page.$$eval('#lineup-table tr.chip-row', (els) => els.length), 22);
    assert.equal(await page.$$eval('#lineup-table tr.group', (els) => els.length), 6);
    assert.equal(await page.$$eval('#lineup-table tr.detail:not([hidden])', (els) => els.length), 0);
    assert.match(await page.locator('#lineup').innerText(), /TPU 8t[\s\S]*TPU 8i/, 'TPU 8t and 8i described as announced');
    // Figures 4–6 are drawn
    assert.equal(await page.$$eval('#roofsvg path.roof-line', (els) => els.length), 1);
    assert.equal(await page.$$eval('#scatter circle', (els) => els.length), 22, 'one dot per rentable chip');
    assert.equal(await page.$$eval('#topos svg', (els) => els.filter((s) => s.childElementCount > 3).length), 4);
    // chart text is never smaller than 11 CSS px, measured after scaling
    const small = await page.$$eval('svg text', (ts) => ts.map((t) => {
      const svg = t.ownerSVGElement, vb = svg.viewBox.baseVal;
      const scale = vb && vb.width ? svg.getBoundingClientRect().width / vb.width : 1;
      return [t.textContent, +(parseFloat(getComputedStyle(t).fontSize) * scale).toFixed(1)];
    }).filter(([s, px]) => s.trim() && px < 11));
    assert.deepEqual(small, [], 'chart text below 11px');
    // scroll through the page so the top bar's section tracking runs
    await page.evaluate(async () => { for (let y = 0; y < document.body.scrollHeight; y += 600) { window.scrollTo(0, y); await new Promise((r) => setTimeout(r, 20)); } });
    await page.waitForTimeout(200);
    assert.equal(await page.$$eval('.topbar nav a[aria-current="true"]', (els) => els.length), 1);
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
    assert.equal(overflow, 0, 'page scrolls horizontally');
    assert.deepEqual(errors, []);
    await context.close();
  });
}

test('roofline and scatter are drawn at the real width of their container', async () => {
  const { page, context, errors } = await open({ width: 390 });
  const roof = await page.evaluate(() => [document.querySelector('#roof-host').clientWidth, +document.querySelector('#roofsvg').getAttribute('width')]);
  assert.equal(roof[1], roof[0]);
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.waitForTimeout(200);
  const wide = await page.evaluate(() => [document.querySelector('#roof-host').clientWidth, +document.querySelector('#roofsvg').getAttribute('width')]);
  assert.equal(wide[1], wide[0]);
  assert.ok(wide[0] > roof[0]);
  assert.deepEqual(errors, []);
  await context.close();
});

test('selecting a chip name in the lineup opens its details and selecting it again closes them', async () => {
  const { page, context, errors } = await open();
  const btn = page.locator('.chip-row[data-id="tpu7x"] .namebtn');
  await btn.click();
  assert.equal(await btn.getAttribute('aria-expanded'), 'true');
  assert.ok(await page.locator('#d-tpu7x').isVisible());
  assert.match(await page.locator('#d-tpu7x').innerText(), /first TPU with native FP8[\s\S]*9,216 chips · 9,216-chip pod, 3D torus/);
  await btn.press('Enter');
  assert.equal(await btn.getAttribute('aria-expanded'), 'false');
  assert.ok(await page.$eval('#d-tpu7x', (el) => el.hidden));
  assert.deepEqual(errors, []);
  await context.close();
});

test('lineup details say when Google publishes no host shape for a generation', async () => {
  const { page, context, errors } = await open();
  await page.locator('.chip-row[data-id="v4"] .namebtn').click();
  assert.match(await page.locator('#d-v4').innerText(), /not published for this generation/);
  await page.locator('.chip-row[data-id="v5p"] .namebtn').click();
  assert.match(await page.locator('#d-v5p').innerText(), /208 vCPU · 448 GB RAM · 200 Gbps host network/);
  assert.deepEqual(errors, []);
  await context.close();
});

test('lineup marks indicative and missing prices', async () => {
  const { page, context, errors } = await open();
  const price = (id) => page.$eval(`.chip-row[data-id="${id}"] td:nth-last-child(2)`, (td) => td.textContent.trim());
  assert.equal(await price('b200'), '$16.11*');
  assert.equal(await price('h100'), '$10.98');
  assert.equal(await price('gb300'), 'not public');
  assert.deepEqual(errors, []);
  await context.close();
});

test('roofline: the verdict follows the chip and the intensity slider', async () => {
  const { page, context, errors } = await open();
  // resting state: H200 at batch-64 decode, memory-bound
  assert.equal(await page.locator('#roof-ai-out').innerText(), '128');
  assert.equal(await page.locator('#roof-presets [data-ai="128"]').getAttribute('aria-pressed'), 'true');
  assert.match(await page.locator('#roof-verdict').innerText(), /H200.*memory-bound.*31% of peak/s);
  await page.locator('#roof-ai').fill('3.5'); // about 3,162 FLOP/byte
  assert.match(await page.locator('#roof-verdict').innerText(), /compute-bound/);
  assert.equal(await page.$$eval('#roof-presets [aria-pressed="true"]', (els) => els.length), 0, 'no preset claims a custom intensity');
  await page.selectOption('#roof-chip', 'v5e');
  assert.match(await page.locator('#roof-verdict').innerText(), /TPU v5e.*ridge at about 480 FLOP\/byte/s);
  assert.deepEqual(errors, []);
  await context.close();
});

test('roofline presets compute what their labels say: 2 FLOP/byte per sequence with one-byte weights', async () => {
  const { page, context, errors } = await open();
  const cases = [['Decode, batch 1', 2], ['Decode, batch 64', 128], ['Prefill, 1,024 tokens', 2048]];
  for (const [label, ai] of cases) {
    await page.getByRole('button', { name: label }).click();
    assert.equal(await page.getByRole('button', { name: label }).getAttribute('aria-pressed'), 'true');
    assert.equal(await page.locator('#roof-ai-out').innerText(), ai.toLocaleString('en-US'));
    // H200: 1,979 TFLOPS peak, 4.8 TB/s
    const att = Math.min(1979, ai * 4.8), share = att / 1979 * 100, pct = share < 1 ? 'under 1%' : Math.round(share) + '%';
    assert.match(await page.locator('#roof-verdict').innerText(), new RegExp(`At ${ai.toLocaleString('en-US')} FLOP/byte .*${ai < 412 ? 'memory' : 'compute'}-bound .*at most ${pct} of peak`, 's'));
    const shown = att >= 10 ? Math.round(att).toLocaleString('en-US') : att.toFixed(1);
    assert.equal(await page.locator('#roofsvg text', { hasText: '% of peak' }).textContent(), ai < 412 ? `${shown} TFLOPS, ${pct} of peak` : `${pct} of peak`);
    assert.equal(await page.locator('#roof-ridge').textContent(), 'ridge 412');
  }
  // decode at batch 1 is memory-bound on every chip on the page
  await page.getByRole('button', { name: 'Decode, batch 1' }).click();
  for (const id of await page.$$eval('#roof-chip option', (os) => os.map((o) => o.value))) {
    await page.selectOption('#roof-chip', id);
    assert.match(await page.locator('#roof-verdict').innerText(), /memory-bound/, `${id} at batch 1`);
  }
  assert.deepEqual(errors, []);
  await context.close();
});

test('fit calculator: 70B at FP8 with 40% headroom needs 98 GB, two H100s, one H200; a full fine-tune uses 16 bytes per parameter', async () => {
  const { page, context, errors } = await open();
  assert.equal(await page.locator('#fit-need').innerText(), '98 GB');
  assert.equal((await fitRow(page, 'H100 (High)'))[1], '2');
  assert.equal((await fitRow(page, 'H200'))[1], '1');
  assert.equal((await fitRow(page, 'T4'))[5].startsWith('no'), true, 'seven T4s exceed the four-GPU N1 limit');
  assert.match((await fitRow(page, 'B200'))[2], /\*$/, 'B200 price is marked indicative');
  assert.match(await page.locator('#fit-cheap').innerText(), /^L4 × 5, \$3\.50 per hour$/);
  assert.doesNotMatch(await page.locator('#fit-cheap').innerText(), /B200|H200|GB200|GB300|Mega/, 'chips not sold on demand never win cheapest-on-demand');
  await page.getByRole('button', { name: 'BF16' }).click();
  assert.equal(await page.locator('#fit-need').innerText(), '196 GB');
  await page.getByRole('button', { name: 'Full fine-tune' }).click();
  assert.equal(await page.locator('#fit-need').innerText(), '1.57 TB');
  assert.equal((await fitRow(page, 'B200'))[1], '10');
  assert.ok(await page.getByRole('button', { name: 'BF16' }).isDisabled(), 'weight precision does not apply to a full fine-tune');
  await page.getByRole('button', { name: 'Inference' }).click();
  assert.equal(await page.locator('#fit-need').innerText(), '196 GB', 'returning to inference restores the chosen precision');
  assert.deepEqual(errors, []);
  await context.close();
});

test('fit calculator: a 1B model at FP8 fits one T4 and the low cost renders with cents', async () => {
  const { page, context, errors } = await open();
  await page.locator('#fit-params').fill('0');
  assert.equal(await page.locator('#fit-params-out').innerText(), '1.0');
  assert.equal(await page.locator('#fit-need').innerText(), '1 GB');
  const t4 = await fitRow(page, 'T4');
  assert.equal(t4[1], '1');
  assert.equal(t4[2], '$0.35');
  assert.match(await page.locator('#fit-cheap').innerText(), /^T4 × 1, /);
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
  // TPU hosts from docs.cloud.google.com/compute/docs/tpus/tpu-machines and tpu/docs/v5e (read 13 Sep 2026): max VM network bandwidth, largest VM shape
  assert.deepEqual([by.v5e.vcpu, by.v5e.ram, by.v5e.net], [224, 384, 200]);
  assert.deepEqual([by.v5p.vcpu, by.v5p.ram, by.v5p.net], [208, 448, 200]);
  assert.deepEqual([by.v6e.vcpu, by.v6e.ram, by.v6e.net], [360, 1440, 200]);
  assert.deepEqual([by.tpu7x.vcpu, by.tpu7x.ram, by.tpu7x.net], [224, 960, 400]);
  for (const id of ['gb300', 'gb200', 'b200', 'h200', 'h100m']) assert.equal(by[id].onDemand, false, `${id} is not sold on demand`);
  // nvidia.com/en-us/data-center/gb200-nvl72 (read 25 Sep 2026): 720 PFLOPS FP8 and 360 PFLOPS FP16/BF16 with sparsity for 72 GPUs
  assert.deepEqual([by.gb200.fp8, by.gb200.bf16], [720e3 / 72 / 2, 360e3 / 72 / 2]);
  // docs.cloud.google.com/tpu/docs/v5p (read 25 Sep 2026): 459 TFLOPS BF16 and FP8, 95 GiB, 2,765 GB/s
  assert.deepEqual([by.v5p.fp8, by.v5p.bf16, by.v5p.mem, by.v5p.bw], [459, 459, 95, 2765]);
  assert.deepEqual(errors, []);
  await context.close();
});

test('scatter names every chip, including those whose points coincide', async () => {
  const { page, context, errors } = await open();
  const text = await page.$eval('#scatter', (s) => s.textContent);
  for (const name of ['GB300', 'GB200', 'B200', 'Ironwood', 'H200', 'H100 High · Mega · Edge', 'RTX PRO 6000', 'TPU v6e', 'A100 80 GB', 'A100 40 GB', 'TPU v5p', 'TPU v5e', 'TPU v4', 'V100', 'TPU v3', 'L4', 'T4', 'TPU v2', 'P100', 'P4']) {
    assert.ok(text.includes(name), `scatter label for ${name}`);
  }
  assert.deepEqual(errors, []);
  await context.close();
});
