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

/** Open the page; collect page errors and console errors. Web fonts are stubbed so the tests run offline;
 *  D3 must load from cdnjs, which CI and the local machine both reach. */
async function open({ width = 1440, reduced = true } = {}) {
  const context = await browser.newContext({ viewport: { width, height: 900 }, reducedMotion: reduced ? 'reduce' : 'no-preference' });
  await context.route(/fonts\.(googleapis|gstatic)\.com/, (route) => route.fulfill({ status: 200, contentType: 'text/css', body: '' }));
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', (e) => errors.push('pageerror: ' + e.message));
  page.on('console', (m) => { if (m.type() === 'error') errors.push('console: ' + m.text()); });
  await page.goto(PAGE);
  await page.waitForFunction(() => window.QAT && window.QAT.lab);
  return { page, context, errors };
}

for (const width of [1440, 390]) {
  test(`page renders every figure without errors or horizontal overflow at ${width}px`, async () => {
    const { page, context, errors } = await open({ width });
    const svgs = ['#memChart', '#numLine', '#hmW', '#hmQ', '#hmE', '#loopSvg', '#steFwd', '#steBwd', '#driftSvg', '#labFit', '#labLoss', '#labHist', '#blockSvg'];
    for (const id of svgs) {
      const n = await page.$eval(id, (el) => el.childElementCount);
      assert.ok(n > 0, `${id} rendered nothing`);
    }
    await page.evaluate(async () => { for (let y = 0; y < document.body.scrollHeight; y += 600) { window.scrollTo(0, y); await new Promise((r) => setTimeout(r, 25)); } });
    await page.waitForTimeout(300);
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
    assert.equal(overflow, 0, 'page scrolls horizontally');
    assert.deepEqual(errors, []);
    await context.close();
  });
}

test('quantizer: symmetric grid, clipping at the edge level, and exact zero', async () => {
  const { page, context } = await open();
  const r = await page.evaluate(() => {
    const q = window.QAT.quantizeArray;
    return { four: q([-1, -0.5, 0, 0.5, 1], 4), two: q([-1, -0.3, 0.3, 1], 2), tiny: q([0.001, -0.002], 8), neg: q([-1, 0.25], 3) };
  });
  assert.equal(r.four.qmax, 7);
  assert.ok(Math.abs(r.four.s - 1 / 7) < 1e-12);
  assert.equal(r.four.out[2], 0, 'zero is representable exactly');
  assert.equal(r.four.out[4], 1, 'the max value lands on the top level');
  assert.equal(r.four.out[0], -1, 'the most negative value lands on -qmax, never on the unused -qmax-1 code');
  assert.deepEqual(r.four.mask, [1, 1, 1, 1, 1], 'a symmetric max lands on ±qmax, nothing is clipped');
  assert.equal(r.neg.out[0], -1, 'negative max defines the scale and is representable');
  assert.ok(Math.abs(r.neg.s - 1 / 3) < 1e-12);
  // 2-bit: levels {-2s, -s, 0, s} with s = 1; 0.3 rounds to 0, -1 to -1
  assert.deepEqual(r.two.out.map((v) => v + 0), [-1, 0, 0, 1]); // + 0 folds -0 into 0
  assert.equal(r.two.qmax, 1);
  assert.equal(r.tiny.out.length, 2);
  await context.close();
});

test('number line: the readout matches the formula and clips past alpha', async () => {
  const { page, context } = await open();
  await page.locator('#nlBits').fill('3');
  await page.locator('#nlAlpha').fill('1');
  const levels = await page.locator('#nlLevels').innerText();
  assert.equal(levels, '7'); // 2^3 - 1 symmetric levels; the -4 code is unused
  // drag the dot far right: x is clamped to 1.6, q clips to +3, x-hat = 1.0
  const dot = page.locator('#numLine circle');
  const box = await dot.boundingBox();
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(box.x + 600, box.y + box.height / 2, { steps: 5 });
  await page.mouse.up();
  assert.equal(await page.locator('#nlQ').innerText(), '3');
  assert.equal(await page.locator('#nlXh').innerText(), '1.000');
  // keyboard path: the range control drives the same value; the negative edge clips to -qmax, not -qmax-1
  await page.locator('#nlXin').fill('-1.6');
  assert.equal(await page.locator('#nlQ').innerText(), '-3');
  assert.equal(await page.locator('#nlXh').innerText(), '-1.000');
  assert.equal(await page.locator('#nlErr').innerText(), '−0.600');
  await context.close();
});

test('heatmap: per-row scales cut the outlier damage that a per-tensor scale spreads everywhere', async () => {
  const { page, context } = await open();
  const rel = () => page.locator('#hmRel').innerText().then((t) => parseFloat(t));
  const clean = await rel();
  await page.locator('#hmOutlier').check();
  const tensorOutlier = await rel();
  assert.ok(tensorOutlier > clean * 2, `outlier should blow up per-tensor error (${clean}% -> ${tensorOutlier}%)`);
  await page.locator('#hmGran button', { hasText: 'per row' }).click();
  const rowOutlier = await rel();
  assert.ok(rowOutlier < tensorOutlier, `per-row (${rowOutlier}%) should beat per-tensor (${tensorOutlier}%)`);
  await context.close();
});

test('loop diagram: every step button changes the caption', async () => {
  const { page, context } = await open();
  const seen = new Set();
  for (const b of await page.locator('#loopSteps button').all()) {
    await b.click();
    seen.add(await page.locator('#loopCaption').innerText());
  }
  assert.equal(seen.size, 6);
  await context.close();
});

test('drift: reset during playback stops the animation and restores the start', async () => {
  const { page, context } = await open({ reduced: false });
  await page.locator('#driftPlay').click();
  await page.waitForTimeout(400);
  await page.locator('#driftReset').click();
  await page.waitForTimeout(300);
  assert.equal(await page.locator('#driftPlay').innerText(), 'Play');
  const label = await page.$$eval('#driftSvg text', (els) => els.map((e) => e.textContent).find((s) => s.startsWith('W = ')));
  assert.equal(label, 'W = 1.42');
  await context.close();
});

test('training lab: auto-runs to completion at the bits it started with, controls lock mid-run, QAT ends below PTQ', async () => {
  const { page, context, errors } = await open({ reduced: false });
  await page.locator('#lab').scrollIntoViewIfNeeded();
  await page.waitForFunction(() => window.QAT.lab.running, null, { timeout: 5000 });
  const four = page.locator('#labBits button', { hasText: '4' });
  assert.equal(await four.isDisabled(), true, 'bit-width buttons lock during a run');
  assert.equal(await page.locator('#labSeed').isDisabled(), true, 'seed locks during a run');
  await four.click({ force: true }); // a forced click on a disabled control must not change the run
  await page.waitForFunction(() => window.QAT.lab.state && window.QAT.lab.state.phase >= 1, null, { timeout: 60000 });
  assert.equal(await page.evaluate(() => window.QAT.lab.state.bits), 3, 'PTQ and QAT are computed at the snapshot precision');
  await page.waitForFunction(() => !window.QAT.lab.running, null, { timeout: 90000 });
  const { l1, l2, status } = await page.evaluate(() => ({ l1: window.QAT.lab.state.l1, l2: parseFloat(document.querySelector('#labL2').innerText), status: document.querySelector('#labStatus').innerText }));
  assert.ok(l2 < l1, `QAT loss ${l2} should be below PTQ loss ${l1}`);
  assert.match(status, /Done at 3 bits/);
  assert.equal(await page.locator('#labRun').isDisabled(), false);
  // changing bits after completion clears the stale comparison instead of redrawing it on a different grid
  await four.click();
  assert.equal(await page.evaluate(() => window.QAT.lab.state), null);
  assert.equal(await page.locator('#labL2').innerText(), '–');
  assert.match(await page.locator('#labStatus').innerText(), /Settings changed/);
  assert.deepEqual(errors, []);
  await context.close();
});

test('training lab: under reduced motion the run completes synchronously with no intermediate frames', async () => {
  const { page, context, errors } = await open({ reduced: true });
  // open() resolves as soon as window.QAT.lab exists, which is after the synchronous run
  const s = await page.evaluate(() => ({ running: window.QAT.lab.running, phase: window.QAT.lab.state && window.QAT.lab.state.phase, l2: document.querySelector('#labL2').innerText }));
  assert.equal(s.running, false);
  assert.equal(s.phase, 1);
  assert.notEqual(s.l2, '–');
  assert.deepEqual(errors, []);
  await context.close();
});

test('transformer block: tap, tap after a theme redraw, and keyboard focus all show the tooltip without errors', async () => {
  const { page, context, errors } = await open();
  const tip = page.locator('#blockTip');
  const block = () => page.locator('#blockSvg g[role="button"]', { hasText: 'W_down' });
  await block().click();
  assert.match(await tip.innerText(), /down projection/);
  await page.evaluate(() => document.documentElement.setAttribute('data-theme', 'dark')); // triggers a full redraw of every figure
  await page.waitForTimeout(100);
  await block().click();
  assert.match(await tip.innerText(), /down projection/);
  await block().focus();
  await page.keyboard.press('Enter');
  assert.match(await tip.innerText(), /down projection/);
  assert.equal(await page.evaluate(() => getComputedStyle(document.querySelector('#blockTip')).opacity), '1');
  assert.deepEqual(errors, []);
  await context.close();
});
