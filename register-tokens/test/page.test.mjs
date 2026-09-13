// Browser regression tests. Drives index.html in headless Chrome (the installed
// Google Chrome via Playwright's "chrome" channel, falling back to Playwright's
// own Chromium if that is unavailable). Run with `pnpm test`.
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

/** Open the page with web fonts stubbed so the tests run offline; collect page and console errors. */
async function open({ width = 1440, reduced = true, fakeClock = false } = {}) {
  const context = await browser.newContext({ viewport: { width, height: 900 }, reducedMotion: reduced ? 'reduce' : 'no-preference' });
  await context.route(/fonts\.(googleapis|gstatic)\.com/, (route) => route.fulfill({ status: 200, contentType: 'text/css', body: '' }));
  const page = await context.newPage();
  if (fakeClock) await page.clock.install();
  const errors = [];
  page.on('pageerror', (e) => errors.push('pageerror: ' + e.message));
  page.on('console', (m) => { if (m.type() === 'error') errors.push('console: ' + m.text()); });
  await page.goto(PAGE);
  await page.waitForSelector('#fixGrid .cell');
  return { page, context, errors };
}

const outlierCount = (page, grid) => page.$$eval(`${grid} .cell.outlier`, (els) => els.length);
const cellColors = (page, grid) => page.$$eval(`${grid} .cell`, (els) => els.map((el) => getComputedStyle(el).backgroundColor).join('|'));
const setRange = (page, sel, value) => page.$eval(sel, (el, v) => { el.value = v; el.dispatchEvent(new Event('input', { bubbles: true })); }, String(value));

for (const width of [1440, 390]) {
  test(`page renders every grid without errors or horizontal overflow at ${width}px`, async () => {
    const { page, context, errors } = await open({ width });
    for (const grid of ['#heroImg', '#heroAttn', '#recapImg', '#normGrid', '#redGrid', '#fixGrid']) {
      assert.equal(await page.$$eval(`${grid} .cell`, (els) => els.length), 196, `${grid} has 14×14 cells`);
    }
    assert.equal(await page.$$eval('#recapStrip .tok', (els) => els.length), 197, 'CLS + 196 patch tokens');
    await page.evaluate(async () => { for (let y = 0; y < document.body.scrollHeight; y += 600) { window.scrollTo(0, y); await new Promise((r) => setTimeout(r, 20)); } });
    await page.waitForTimeout(300);
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
    assert.equal(overflow, 0, 'page scrolls horizontally');
    assert.deepEqual(errors, []);
    await context.close();
  });
}

test('hero: adding registers changes the attention map and the caption', async () => {
  const { page, context, errors } = await open();
  await page.locator('#heroSeg button[data-n="0"]').click();
  const before = await cellColors(page, '#heroAttn');
  assert.match(await page.locator('#heroCap').innerText(), /0 registers/);
  await page.locator('#heroSeg button[data-n="4"]').click();
  assert.notEqual(await cellColors(page, '#heroAttn'), before, 'attention map did not change');
  assert.match(await page.locator('#heroCap').innerText(), /4 registers/);
  assert.equal(await page.locator('#heroSeg button[data-n="4"]').getAttribute('aria-pressed'), 'true');
  assert.deepEqual(errors, []);
  await context.close();
});

test('artifacts: absent in early layers and in ViT-S, present in late layers of ViT-L; probe reports the token kind', async () => {
  const { page, context, errors } = await open();
  await setRange(page, '#layer', 3);
  assert.equal(await page.locator('#layerOut').innerText(), '3');
  assert.equal(await outlierCount(page, '#normGrid'), 0, 'layer 3 should have no artifacts');
  await setRange(page, '#layer', 20);
  const late = await outlierCount(page, '#normGrid');
  assert.ok(late > 0, 'layer 20 of ViT-L should have artifacts');
  assert.match(await page.$eval('#hist', (el) => el.textContent), new RegExp(`${late} artifacts`));
  // clicking an artifact vs a normal patch drives the probe panel
  await page.locator('#normGrid .cell.outlier').first().click();
  assert.equal(await page.locator('#probeKind').innerText(), 'artifact');
  assert.match(await page.locator('#probeKv').innerText(), /69\.0%/);
  await page.locator('#normGrid .cell:not(.outlier)').first().click();
  assert.equal(await page.locator('#probeKind').innerText(), 'normal');
  assert.match(await page.locator('#probeKv').innerText(), /65\.8%/);
  // DINOv2 ViT-S and ViT-B: slider max shrinks to 12 and no artifacts at any layer (paper Fig. 4c)
  for (const m of ['S', 'B']) {
    await page.locator(`#modelSeg button[data-m="${m}"]`).click();
    assert.equal(await page.$eval('#layer', (el) => el.max), '12');
    await setRange(page, '#layer', 12);
    assert.equal(await outlierCount(page, '#normGrid'), 0, `DINOv2 ViT-${m} never has artifacts`);
  }
  assert.deepEqual(errors, []);
  await context.close();
});

test('registers: one register removes every artifact and moves the high norms into [REG] tokens', async () => {
  const { page, context, errors } = await open();
  assert.ok((await outlierCount(page, '#fixGrid')) > 0, 'zero registers should show artifacts');
  assert.equal(await page.$$eval('#fixStrip .tok.reg', (els) => els.length), 0);
  assert.match(await page.locator('#fixStatus').innerText(), /present/);
  await setRange(page, '#nreg', 1);
  assert.equal(await outlierCount(page, '#fixGrid'), 0, 'one register should remove all artifacts');
  assert.equal(await page.$$eval('#fixStrip .tok.art', (els) => els.length), 0);
  assert.equal(await page.$$eval('#fixStrip .tok.reg', (els) => els.length), 1);
  assert.equal(await page.$$eval('#seq .blk.reg', (els) => els.length), 1);
  assert.match(await page.locator('#fixStatus').innerText(), /none/);
  // order follows the DINOv2 implementation: [CLS], registers, then patches
  const order = await page.$$eval('#seq .blk', (els) => els.map((el) => el.className.replace('blk ', '')).filter((c) => c !== 'dots'));
  assert.deepEqual(order.slice(0, 3), ['cls', 'reg', 'pat']);
  const stripOrder = await page.$$eval('#fixStrip .tok', (els) => els.map((el) => el.className.includes('cls') ? 'cls' : el.className.includes('reg') ? 'reg' : 'pat'));
  assert.deepEqual(stripOrder.slice(0, 3), ['cls', 'reg', 'pat']);
  await setRange(page, '#nreg', 8);
  assert.equal(await page.$$eval('#seq .blk.reg', (els) => els.length), 8);
  assert.equal(await page.locator('#pipeN').innerText(), '8');
  assert.match(await page.locator('#budgetKv').innerText(), /205/, 'sequence length 197 + 8');
  // view toggle swaps the heatmap and its legend
  const norms = await cellColors(page, '#fixGrid');
  await page.locator('#viewSeg button[data-v="attn"]').click();
  assert.notEqual(await cellColors(page, '#fixGrid'), norms);
  assert.match(await page.locator('#fixLegend').innerText(), /attention/);
  assert.deepEqual(errors, []);
  await context.close();
});

test('results table matches the paper (ICLR 2024, Tables 2 and 3)', async () => {
  const { page, context, errors } = await open();
  const rows = await page.$$eval('#results tbody tr', (trs) => trs.map((tr) => [...tr.children].map((td) => td.innerText.trim())));
  assert.deepEqual(rows, [
    ['DeiT-III (ViT-B)', '0', '84.7', '38.9', '0.511', '11.7'],
    ['', '4', '84.7', '39.1', '0.512', '27.1'],
    ['OpenCLIP (ViT-B)', '0', '78.2', '26.6', '0.702', '38.8'],
    ['', '4', '78.1', '26.7', '0.661', '37.1'],
    ['DINOv2 (ViT-L)', '0', '84.3', '46.6', '0.378', '35.3'],
    ['', '4', '84.8', '47.9', '0.366', '55.4'],
  ]);
  // regressions are styled as such, not as improvements
  const openclipReg = await page.$$eval('#results tbody tr:nth-child(4) td', (tds) => tds.map((td) => td.className));
  assert.deepEqual(openclipReg, ['', '', 'down', 'up', 'up', 'down']);
  assert.deepEqual(errors, []);
  await context.close();
});

test('patch cells and tokens are real buttons operable from the keyboard', async () => {
  const { page, context, errors } = await open();
  assert.equal(await page.$eval('#normGrid .cell', (el) => el.tagName), 'BUTTON');
  assert.equal(await page.$eval('#recapStrip .tok', (el) => el.tagName), 'BUTTON');
  await setRange(page, '#layer', 20);
  await page.focus('#normGrid .cell.outlier');
  await page.keyboard.press('Enter');
  assert.equal(await page.locator('#probeKind').innerText(), 'artifact');
  // focusing a token in the recap strip selects it, like hovering does
  await page.focus('#recapStrip .tok:nth-child(5)');
  assert.match(await page.locator('#recapInfo h3').innerText(), /token #4/);
  assert.equal(await page.$eval('#recapImg .cell.sel', (el) => el.dataset.i), '3');
  assert.deepEqual(errors, []);
  await context.close();
});

test('hero autoplay flips exactly three times, skips under reduced motion, and stops on a click', async () => {
  const cap = (page) => page.locator('#heroCap').innerText();
  // reduced motion: no timer at all
  let r = await open({ reduced: true, fakeClock: true });
  await r.page.clock.runFor(10000);
  assert.match(await cap(r.page), /0 registers/);
  assert.deepEqual(r.errors, []);
  await r.context.close();
  // normal motion: 0 -> 4 -> 0 -> 4 and then nothing
  r = await open({ reduced: false, fakeClock: true });
  await r.page.clock.runFor(1900); assert.match(await cap(r.page), /4 registers/);
  await r.page.clock.runFor(1800); assert.match(await cap(r.page), /0 registers/);
  await r.page.clock.runFor(1800); assert.match(await cap(r.page), /4 registers/);
  await r.page.clock.runFor(10000); assert.match(await cap(r.page), /4 registers/);
  assert.equal(await r.page.locator('#heroSeg button[data-n="4"]').getAttribute('aria-pressed'), 'true');
  assert.deepEqual(r.errors, []);
  await r.context.close();
  // a click during autoplay cancels it
  r = await open({ reduced: false, fakeClock: true });
  await r.page.clock.runFor(500);
  await r.page.locator('#heroSeg button[data-n="4"]').click();
  await r.page.clock.runFor(10000);
  assert.match(await cap(r.page), /4 registers/);
  assert.deepEqual(r.errors, []);
  await r.context.close();
});
