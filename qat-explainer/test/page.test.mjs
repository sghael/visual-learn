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
 *  the page has no other network dependencies. */
async function open({ width = 1440, reduced = true } = {}) {
  const context = await browser.newContext({ viewport: { width, height: 900 }, reducedMotion: reduced ? 'reduce' : 'no-preference' });
  await context.route(/fonts\.(googleapis|gstatic)\.com/, (route) => route.fulfill({ status: 200, contentType: 'text/css', body: '' }));
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', (e) => errors.push('pageerror: ' + e.message));
  page.on('console', (m) => { if (m.type() === 'error') errors.push('console: ' + m.text()); });
  await page.goto(PAGE);
  await page.waitForFunction(() => window.QAT && window.QAT.lab && window.QAT.lab.result);
  return { page, context, errors };
}

const FIGURES = ['#memChart', '#numLine', '#errLine', '#alphaPlot', '#hmA', '#hmB', '#hmC', '#loopSvg', '#steFwd', '#steTrue', '#steSte', '#driftSvg',
  '#labFit2', '#labLoss2', '#labFit3', '#labLoss3', '#labFit4', '#labLoss4', '#seedPlot'];

for (const width of [1440, 390]) {
  test(`every figure renders, with legible text inside its frame and no horizontal overflow, at ${width}px`, async () => {
    const { page, context, errors } = await open({ width });
    for (const id of FIGURES) {
      const n = await page.$eval(id + ' svg', (el) => el.childElementCount);
      assert.ok(n > 0, `${id} rendered nothing`);
    }
    await page.evaluate(async () => { for (let y = 0; y < document.body.scrollHeight; y += 600) { window.scrollTo(0, y); await new Promise((r) => setTimeout(r, 20)); } });
    await page.waitForTimeout(200);
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
    assert.equal(overflow, 0, 'page scrolls horizontally');
    // SVGs are drawn at the pixel width of their container, so font-size in px is the rendered size.
    // Every label must be at least 11px and must lie inside its SVG (the old memory chart clipped "24 GB RTX 4…").
    const bad = await page.evaluate(() => {
      const out = [];
      document.querySelectorAll('.graphic svg text').forEach((t) => {
        if (!t.textContent.trim()) return;
        const svg = t.ownerSVGElement, k = svg.getBoundingClientRect().width / svg.viewBox.baseVal.width;
        const px = parseFloat(getComputedStyle(t).fontSize) * k;
        const b = t.getBBox(), W = svg.viewBox.baseVal.width;
        if (px < 11) out.push(`${t.textContent} is ${px.toFixed(1)}px`);
        if (b.x < -1 || b.x + b.width > W + 1) out.push(`${t.textContent} overflows its chart (${b.x.toFixed(0)}..${(b.x + b.width).toFixed(0)} of ${W})`);
      });
      return out;
    });
    assert.deepEqual(bad, []);
    assert.deepEqual(errors, []);
    await context.close();
  });
}

test('memory chart: every GPU reference line is labeled in full, and 70B at int4 is 35 GB', async () => {
  const { page, context } = await open({ width: 390 });
  const texts = await page.$$eval('#memChart svg text', (els) => els.map((e) => e.textContent));
  for (const label of ['16 GB laptop GPU', '24 GB RTX 4090', '80 GB H100']) assert.ok(texts.includes(label), `missing ${label}`);
  for (const v of ['0.5', '13.5', '35', '140', '810']) assert.ok(texts.includes(v), `missing value ${v}`);
  await context.close();
});

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
  // 2-bit: three symmetric levels {-s, 0, s} with s = 1; 0.3 rounds to 0, -1 to -1
  assert.deepEqual(r.two.out.map((v) => v + 0), [-1, 0, 0, 1]); // + 0 folds -0 into 0
  assert.equal(r.two.qmax, 1);
  assert.equal(r.tiny.out.length, 2);
  await context.close();
});

test('number line: the readout matches the formula, dragging clips past alpha, and the error is labeled by kind', async () => {
  const { page, context } = await open();
  await page.locator('#nlBits').fill('3');
  await page.locator('#nlAlpha').fill('1');
  assert.equal(await page.locator('#nlLevels').innerText(), '7'); // 2^3 - 1 symmetric levels; the -4 code is unused
  assert.equal(await page.locator('#nlS').innerText(), '0.333');
  // drag the dot far right: x is clamped to 1.6, q clips to +3, x-hat = 1.0
  const dot = page.locator('#numLine circle');
  const box = await dot.boundingBox();
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(box.x + 600, box.y + box.height / 2, { steps: 5 });
  await page.mouse.up();
  assert.equal(await page.locator('#nlQ').innerText(), '3');
  assert.equal(await page.locator('#nlXh').innerText(), '1.000');
  assert.equal(await page.locator('#nlKind').innerText(), 'clipping');
  // keyboard path: the range control drives the same value; the negative edge clips to -qmax, not -qmax-1
  await page.locator('#nlXin').fill('-1.6');
  assert.equal(await page.locator('#nlQ').innerText(), '−3');
  assert.equal(await page.locator('#nlXh').innerText(), '−1.000');
  assert.equal(await page.locator('#nlErr').innerText(), '−0.600');
  // inside the range the error is rounding error, at most s/2
  await page.locator('#nlXin').fill('0.5');
  assert.equal(await page.locator('#nlKind').innerText(), 'rounding');
  const err = Math.abs(parseFloat((await page.locator('#nlErr').innerText()).replace('−', '-')));
  assert.ok(err <= 1 / 3 / 2 + 5e-4, `rounding error ${err} exceeds s/2`);
  await context.close();
});

test('clipping range: the labeled best alpha is the computed minimum, and it moves outward as bits are added', async () => {
  const { page, context } = await open();
  const best = await page.evaluate(() => {
    const out = {};
    for (const b of [2, 4, 8]) { let m = null; for (let a = 0.2; a <= 1.5001; a += 0.01) { const v = window.QAT.quantMse(a, b).t; if (!m || v < m.v) m = { a, v }; } out[b] = m.a; }
    return out;
  });
  assert.ok(best[2] < best[4] && best[4] < best[8], `best alpha should grow with bits: ${JSON.stringify(best)}`);
  const label = async () => page.$$eval('#alphaPlot svg text', (els) => els.map((e) => e.textContent).find((s) => s.includes('lowest at')));
  assert.equal(await label(), `lowest at α = ${best[4].toFixed(2)}`);
  await page.locator('#nlBits').fill('2');
  assert.equal(await label(), `lowest at α = ${best[2].toFixed(2)}`);
  await context.close();
});

test('scale granularity: an outlier inflates per-tensor error, and per-row scales contain it', async () => {
  const { page, context } = await open();
  const shown = async (id) => parseFloat(await page.locator(id).innerText());
  const [a, b, c] = [await shown('#hmRelA'), await shown('#hmRelB'), await shown('#hmRelC')];
  assert.ok(b > a * 2, `outlier should blow up per-tensor error (${a}% -> ${b}%)`);
  assert.ok(c < b, `per-row (${c}%) should beat per-tensor (${b}%)`);
  const computed = await page.evaluate(() => window.QAT.heat.map((h) => +(h.rel * 100).toFixed(1)));
  assert.deepEqual([a, b, c], computed, 'labels match the computed relative errors');
  await context.close();
});

test('training-step diagram: all seven stages are drawn and the two columns never overlap on a phone', async () => {
  const { page, context } = await open({ width: 390 });
  const r = await page.evaluate(() => {
    const svg = document.querySelector('#loopSvg svg'), w = svg.viewBox.baseVal.width;
    const texts = [...svg.querySelectorAll('text')];
    const titles = texts.filter((t) => t.classList.contains('node-t')).map((t) => t.textContent);
    const left = texts.filter((t) => t.getBBox().x + t.getBBox().width / 2 < w / 2 - 20);
    const right = texts.filter((t) => t.getBBox().x + t.getBBox().width / 2 > w / 2 + 20);
    const maxLeft = Math.max(...left.map((t) => t.getBBox().x + t.getBBox().width));
    const minRight = Math.min(...right.map((t) => t.getBBox().x));
    return { titles, maxLeft, minRight };
  });
  assert.deepEqual(r.titles.sort(), ['Backpropagate', 'Fake-quantize', 'Layer output', 'Loss', 'Master weight', 'Optimizer step', 'Straight-through'].sort());
  assert.ok(r.maxLeft < r.minRight, `left column (to ${r.maxLeft}) runs into the right column (from ${r.minRight})`);
  await context.close();
});

test('STE: the surrogate gradient is 1 just inside the clipping range and 0 just outside', async () => {
  const { page, context } = await open();
  const g = await page.evaluate(() => [1.9, 2, 2.1, -2.1, 0].map((x) => window.QAT.steGrad(x, 2)));
  assert.deepEqual(g, [1, 1, 0, 0, 1]);
  await context.close();
});

test('drift: the quantized weight changes exactly when the master weight crosses a boundary, and the annotations name those steps', async () => {
  const { page, context } = await open();
  const d = await page.evaluate(() => window.QAT.drift);
  d.Wt.forEach((w, t) => assert.equal(d.Q[t], Math.max(-2, Math.min(2, Math.round(w))), `step ${t}`));
  assert.ok(d.flips.length >= 3, 'the trace should show several level changes');
  const cross = d.flips.find((f) => f.from === 0 && f.to === -1);
  const texts = await page.$$eval('#driftSvg svg text', (els) => els.map((e) => e.textContent));
  assert.ok(texts.some((s) => s.startsWith(`step ${cross.t}: W crosses −0.5`)), 'crossing annotation matches the simulation');
  assert.ok(texts.some((s) => /^steps \d+–\d+: Ŵ flips back and forth$/.test(s)), 'oscillation annotation present');
  await context.close();
});

test('training lab: the resting state is already trained, and every direct label matches the computed loss', async () => {
  const { page, context, errors } = await open({ reduced: false });
  const r = await page.evaluate(() => {
    const lab = window.QAT.lab, res = lab.result;
    const fmt = (v) => (v >= 0.001 ? v.toPrecision(2) : v.toExponential(1).replace('e-', 'e−'));
    const out = { seed: res.seed, rows: [] };
    for (const b of [2, 3, 4]) {
      const labels = Object.fromEntries([...document.querySelectorAll(`#labLoss${b} svg text[data-k]`)].map((t) => [t.dataset.k, t.textContent]));
      out.rows.push({
        b, labels, gain: document.querySelector('#labGain' + b).textContent,
        want: { ptq: 'PTQ ' + fmt(res.bits[b].l1), qat: 'QAT ' + fmt(res.bits[b].l2), fp: 'fp32 ' + fmt(res.l0) },
        wantGain: `QAT loss ${lab.ratioText(res.bits[b].ratio)} than PTQ`,
        l1: res.bits[b].l1, l2: res.bits[b].l2,
        recomputed: { l1: lab.loss(res.fp, b), l2: lab.loss(res.bits[b].qat, b) },
      });
    }
    return out;
  });
  assert.equal(r.seed, 2, 'default seed');
  for (const row of r.rows) {
    assert.deepEqual(row.labels, row.want, `labels at ${row.b} bits`);
    assert.equal(row.gain, row.wantGain);
    assert.ok(Math.abs(row.recomputed.l1 - row.l1) < 1e-12 && Math.abs(row.recomputed.l2 - row.l2) < 1e-12, 'stored losses are the losses of the stored models');
    if (row.b >= 3) assert.ok(row.l2 < row.l1, `QAT should beat PTQ at ${row.b} bits`);
  }
  assert.match(await page.locator('#labTakeaway').innerText(), /^Seed 2: compared with PTQ/);
  assert.deepEqual(errors, []);
  await context.close();
});

test('training lab: rapid seed changes end on the last seed clicked, with every figure in agreement', async () => {
  const { page, context, errors } = await open({ reduced: false });
  await page.locator('#labSeed button', { hasText: '5' }).click();
  await page.locator('#labSeed button', { hasText: '3' }).click();
  await page.waitForFunction(() => window.QAT.lab.result.seed === 3 && /Trained/.test(document.querySelector('#labStatus').textContent), null, { timeout: 30000 });
  await page.waitForTimeout(300); // let any queued run from the first click land, if the ordering were wrong
  assert.equal(await page.evaluate(() => window.QAT.lab.result.seed), 3);
  assert.equal(await page.locator('#labSeed button[aria-pressed="true"]').innerText(), '3');
  const seedLabels = await page.$$eval('#seedPlot svg text', (els) => els.map((e) => e.textContent).filter((s) => s.startsWith('seed ')));
  assert.deepEqual(seedLabels, ['seed 3', 'seed 3', 'seed 3']);
  // seed 3 is the run where 2-bit QAT ends worse than PTQ; the page must say so rather than claim a win
  assert.match(await page.locator('#labGain2').innerText(), /higher than PTQ/);
  assert.match(await page.locator('#labTakeaway').innerText(), /At 2 bits this QAT run ends worse/);
  assert.deepEqual(errors, []);
  await context.close();
});

test('training lab: the precomputed all-seed ratios in Figure 8 match a fresh run of every seed', async () => {
  const { page, context } = await open();
  const r = await page.evaluate(() => {
    const lab = window.QAT.lab, bad = [];
    for (let s = 1; s <= 8; s++) {
      const res = lab.train(s);
      for (const b of [2, 3, 4]) { const want = lab.SEED_RATIOS[b][s - 1], got = res.bits[b].ratio; if (Math.abs(got / want - 1) > 0.005) bad.push({ s, b, want, got }); }
    }
    return { bad, wins3: lab.SEED_RATIOS[3].every((v) => v > 1), wins4: lab.SEED_RATIOS[4].every((v) => v > 1), wins2: lab.SEED_RATIOS[2].filter((v) => v > 1).length };
  });
  assert.deepEqual(r.bad, []);
  assert.ok(r.wins3 && r.wins4, 'the caption says QAT wins on every seed at 3 and 4 bits');
  assert.equal(r.wins2, 7, 'the caption says QAT wins on seven of eight seeds at 2 bits');
  await context.close();
});

test('top bar: scrolling marks the current section and advances the progress line', async () => {
  const { page, context } = await open();
  await page.evaluate(() => document.getElementById('lab').scrollIntoView());
  await page.waitForTimeout(200);
  assert.equal(await page.locator('.topbar nav a[aria-current="true"]').innerText(), 'Lab');
  const w = await page.evaluate(() => parseFloat(document.getElementById('progress').style.width));
  assert.ok(w > 20 && w < 100, `progress ${w}%`);
  await context.close();
});
