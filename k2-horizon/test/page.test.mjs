// Browser regression tests. Drives index.html in headless Chrome (the installed
// Google Chrome via Playwright's "chrome" channel, falling back to Playwright's
// own Chromium). Run with `pnpm test`. The page has no third-party scripts;
// Google Fonts requests are stubbed so the tests run offline.
import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { chromium } from 'playwright';

const here = path.dirname(fileURLToPath(import.meta.url));
const PAGE = pathToFileURL(path.join(here, '..', 'index.html')).href;
const FIGURES = ['#fleetChart', '#layerMultiples', '#movaLedger', '#unoChart', '#benchChart'];
const N_WORDS = 36; // words in the Uno simulation sentence
let browser;

before(async () => {
  try { browser = await chromium.launch({ channel: 'chrome', headless: true }); }
  catch { browser = await chromium.launch({ headless: true }); }
});
after(async () => { if (browser) await browser.close(); });

async function open({ width = 1440, reduced = true, clock = false } = {}) {
  const context = await browser.newContext({ viewport: { width, height: 900 }, reducedMotion: reduced ? 'reduce' : 'no-preference' });
  await context.route(/fonts\.(googleapis|gstatic)\.com/, (route) => route.fulfill({ status: 200, contentType: 'text/css', body: '' }));
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', (e) => errors.push('pageerror: ' + e.message));
  page.on('console', (m) => { if (m.type() === 'error') errors.push('console: ' + m.text()); });
  if (clock) await page.clock.install();
  await page.goto(PAGE);
  await page.waitForSelector('#benchChart svg');
  return { page, context, errors };
}
const counts = async (page) => ({ ar: await page.locator('#arCount').innerText(), uno: await page.locator('#unoCount').innerText() });
const parseUno = (s) => { const m = s.match(/^(\d+) tokens · (\d+) passes/); return { tokens: +m[1], passes: +m[2] }; };

for (const width of [1440, 390]) {
  test(`every figure renders, chart text is legible, and nothing overflows at ${width}px`, async () => {
    const { page, context, errors } = await open({ width });
    for (const sel of FIGURES) {
      const n = await page.$eval(sel, (el) => el.querySelectorAll('svg *').length);
      assert.ok(n > 10, `${sel} rendered nothing`);
    }
    assert.equal(await page.$$eval('#layerMultiples svg', (els) => els.length), 3, 'three layer panels');
    const small = await page.$$eval('svg text', (els) => els.filter((t) => t.getBoundingClientRect().width > 0 && parseFloat(getComputedStyle(t).fontSize) < 11).map((t) => t.textContent));
    assert.deepEqual(small, [], 'chart text smaller than 11px');
    await page.evaluate(async () => { for (let y = 0; y < document.body.scrollHeight; y += 600) { window.scrollTo(0, y); await new Promise((r) => setTimeout(r, 15)); } });
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
    assert.equal(overflow, 0, 'page scrolls horizontally');
    assert.deepEqual(errors, []);
    await context.close();
  });
}

test('configuration table carries the values in each config.json', async () => {
  const { page, context, errors } = await open();
  const row = async (label) => page.$$eval('#configTable tbody tr', (trs, l) => {
    const tr = trs.find((r) => r.querySelector('th').textContent.trim() === l);
    return [...tr.querySelectorAll('td')].map((td) => td.textContent.trim());
  }, label);
  // 375B head_dim is 128 in its config (rope_head_dim is 64); the old page said 64.
  assert.deepEqual(await row('Head dimension'), ['64', '128', '128', '128', '128', '128']);
  assert.deepEqual(await row('MoVA value experts'), ['none', 'none', 'none', 'none', '64, top 4', 'none']);
  assert.deepEqual(await row('Feed-forward experts'), ['none', 'none', 'none', 'none', '100 + 1 shared, top 8', '192 + 1 shared, top 8']);
  assert.equal((await row('Vocabulary'))[0], '64,256');
  assert.deepEqual(await row('Context (tokens)'), ['131,072', '524,288', '524,288', '524,288', '524,288', '524,288']);
  assert.deepEqual(errors, []);
  await context.close();
});

test('layer small multiples show the configured expert counts and active experts', async () => {
  const { page, context, errors } = await open();
  const count = (variant, cls, on) => page.$$eval(`#layerMultiples [data-variant="${variant}"] rect.${cls}${on ? '.on' : ''}`, (els) => els.length);
  assert.equal(await count('dense', 'ex'), 0, 'dense has no experts');
  assert.equal(await count('moe', 'ex'), 192);
  assert.equal(await count('moe', 'ex', true), 8);
  assert.equal(await count('moe', 'vx'), 0, 'the 375B model has no MoVA');
  assert.equal(await count('mova', 'ex'), 100);
  assert.equal(await count('mova', 'ex', true), 8);
  assert.equal(await count('mova', 'vx'), 64);
  assert.equal(await count('mova', 'vx', true), 4);
  assert.deepEqual(errors, []);
  await context.close();
});

test('parameter ledger labels match the numbers in the prose', async () => {
  const { page, context, errors } = await open();
  const text = await page.$eval('#movaLedger', (el) => el.textContent);
  assert.match(text, /26\.5B stored · 2\.1B used/);
  assert.match(text, /7\.5B stored · 0\.47B used/);
  assert.match(text, /1\.6B, all used/);
  assert.deepEqual(errors, []);
  await context.close();
});

test('Uno rests on a finished run, and reset during a run stops the timer', async () => {
  const { page, context, errors } = await open({ clock: true, reduced: false });
  const rest = await counts(page);
  assert.equal(rest.ar, `${N_WORDS} tokens · ${N_WORDS} passes`);
  const u = parseUno(rest.uno);
  assert.equal(u.tokens, N_WORDS);
  assert.ok(u.passes < N_WORDS, 'Uno should finish in fewer passes than autoregressive decoding');
  await page.locator('#unoPlay').click(); // finished, so Play starts a fresh run
  assert.equal(await page.locator('#unoPlay').innerText(), 'Pause');
  await page.clock.runFor(600 * 5 + 50);
  const mid = parseUno((await counts(page)).uno);
  assert.equal(mid.passes, 5, 'one pass per tick');
  await page.locator('#unoReset').click();
  const atReset = await counts(page);
  assert.equal(atReset.ar, '0 tokens · 0 passes');
  assert.match(atReset.uno, /^0 tokens · 0 passes/);
  assert.equal(await page.locator('#unoPlay').innerText(), 'Play');
  await page.clock.runFor(10000);
  assert.deepEqual(await counts(page), atReset, 'a cancelled run kept advancing after reset');
  assert.deepEqual(errors, []);
  await context.close();
});

test('Uno pause holds the state, and Step advances exactly one pass', async () => {
  const { page, context, errors } = await open({ clock: true, reduced: false });
  await page.locator('#unoReset').click();
  await page.locator('#unoPlay').click();
  await page.clock.runFor(600 * 3 + 50);
  await page.locator('#unoPlay').click(); // pause
  const paused = await counts(page);
  assert.equal(paused.ar, '3 tokens · 3 passes');
  await page.clock.runFor(5000);
  assert.deepEqual(await counts(page), paused, 'pause did not stop the run');
  await page.locator('#unoStep').click();
  assert.equal((await counts(page)).ar, '4 tokens · 4 passes');
  assert.equal(parseUno((await counts(page)).uno).passes, 4);
  assert.deepEqual(errors, []);
  await context.close();
});

test('switching block size mid-animation stops the run and redraws a finished run for the new size', async () => {
  const { page, context, errors } = await open({ clock: true, reduced: false });
  await page.locator('#unoReset').click();
  await page.locator('#unoPlay').click();
  await page.clock.runFor(600 * 4 + 50);
  await page.locator('#unoBlock button[data-block="16"]').click();
  assert.equal(await page.locator('#unoBlock button[data-block="16"]').getAttribute('aria-pressed'), 'true');
  assert.equal(await page.locator('#unoPlay').innerText(), 'Play');
  const after = await counts(page);
  const u = parseUno(after.uno);
  assert.equal(u.tokens, N_WORDS);
  assert.ok(u.tokens / u.passes <= (16 + 1) / 2, 'tokens per pass above the (B + 1)/2 bound');
  await page.clock.runFor(10000);
  assert.deepEqual(await counts(page), after, 'the old timer survived the switch');
  // the agreement slider behaves the same way and shows its value
  await page.locator('#unoAcc').fill('95');
  assert.equal(await page.locator('#unoAccOut').innerText(), '95%');
  assert.deepEqual(errors, []);
  await context.close();
});

test('Uno readout per pass matches the simulation bounds for every setting', async () => {
  const { page, context, errors } = await open();
  const results = await page.evaluate((n) => {
    const out = [];
    for (const B of [4, 8, 16]) for (const p of [0.3, 0.7, 0.95]) for (let seed = 1; seed <= 40; seed++) {
      const r = window.K2.simulate(B, p, seed);
      out.push({ B, p, seed, ok: r.tokens === n && r.tokens / r.passes >= 1 && r.tokens / r.passes <= (B + 1) / 2 });
    }
    return out.filter((r) => !r.ok);
  }, N_WORDS);
  assert.deepEqual(results, [], 'simulation left the paper’s 1 ≤ TPF ≤ (B + 1)/2 range');
  // label versus computation: the lane readout is tokens / passes
  const { uno } = await counts(page);
  const u = parseUno(uno);
  assert.ok(uno.endsWith(`${(u.tokens / u.passes).toFixed(2)} per pass`));
  assert.deepEqual(errors, []);
  await context.close();
});

test('benchmark figure covers all six models and its labels match the data', async () => {
  const { page, context, errors } = await open();
  const buttons = page.locator('#benchModel button');
  assert.deepEqual(await buttons.allInnerTexts(), ['0.9B', '3.7B', '7B', '32B', '36B-A4B', '375B-A23B']);
  assert.equal(await page.locator('#benchModel button[aria-pressed="true"]').innerText(), '375B-A23B');
  // 375B: 18 rows, one blue dot each, the audited Terminal-Bench marker, and the Elo row on its own axis
  assert.equal(await page.$$eval('#benchChart circle.k2', (els) => els.length), 18);
  assert.equal(await page.$$eval('#benchChart circle.audited', (els) => els.length), 1);
  const notes = await page.$$eval('#benchChart text.row-note', (els) => els.map((e) => e.textContent));
  assert.ok(notes.includes('K2 70.2 (66.9 audited) · best other 80.9, GPT 5.6 Luna (max)'), notes.join('\n'));
  assert.ok(notes.includes('K2 1,441 · best other 1,584, Claude Sonnet 5 (max)'));
  assert.ok(notes.includes('K2 42.6 · best other 48.8, GPT 5.6 Luna (max)'), 'SWE Bench Pro best other is GPT 5.6 Luna, not Claude');
  assert.match(await page.locator('#benchTakeaway').innerText(), /top score in 1 of the 18 rows/);
  // every "best other" label equals the maximum of that row in the data table
  const mismatches = await page.evaluate(() => {
    const rows = [...document.querySelectorAll('#benchTable tbody tr')];
    const notes = [...document.querySelectorAll('#benchChart text.row-note')].map((e) => e.textContent);
    return rows.map((tr, i) => {
      const vals = [...tr.querySelectorAll('td')].slice(1).map((td) => parseFloat(td.textContent.replace(/,/g, ''))).filter((v) => !Number.isNaN(v));
      const best = Math.max(...vals);
      const m = notes[i].match(/best other ([\d,.]+)/);
      return m && parseFloat(m[1].replace(/,/g, '')) !== best ? tr.querySelector('th').textContent : null;
    }).filter(Boolean);
  });
  assert.deepEqual(mismatches, []);
  await page.locator('#benchModel button', { hasText: /^32B$/ }).click();
  assert.equal(await page.locator('#benchModel button[aria-pressed="true"]').innerText(), '32B');
  const head = await page.$$eval('#benchTable thead th', (els) => els.map((e) => e.textContent));
  assert.deepEqual(head, ['Benchmark', 'K2-Horizon-32B (Stage 1)', 'Qwen3.8-27B', 'Muse Glimmer-30B', 'IBM Granite 4.2 30B']);
  assert.equal(await page.$$eval('#benchChart circle.k2', (els) => els.length), 9);
  assert.equal(await page.$$eval('#benchChart circle.other', (els) => els.length), 27, '9 rows × 3 comparison models');
  assert.match(await page.locator('#benchTakeaway').innerText(), /top score in 0 of the 9 rows/);
  await page.locator('#benchModel button', { hasText: '3.7B' }).click();
  assert.match(await page.locator('#benchTakeaway').innerText(), /top score in 5 of the 8 rows/);
  await page.locator('#benchModel button', { hasText: /^7B$/ }).click();
  assert.equal(await page.$$eval('#benchChart circle.k2', (els) => els.length), 8);
  assert.match(await page.locator('#benchTakeaway').innerText(), /top score in 6 of the 7 rows/, 'BrowseComp has no comparison');
  assert.deepEqual(errors, []);
  await context.close();
});

test('controls are keyboard reachable buttons with pressed state', async () => {
  const { page, context, errors } = await open();
  await page.locator('#benchModel button').first().focus();
  await page.keyboard.press('Tab'); await page.keyboard.press('Tab');
  await page.keyboard.press('Enter');
  assert.equal(await page.locator('#benchModel button[aria-pressed="true"]').innerText(), '7B');
  for (const id of ['unoPlay', 'unoStep', 'unoReset']) assert.equal(await page.$eval('#' + id, (el) => el.tagName), 'BUTTON');
  assert.equal(await page.$eval('#unoAcc', (el) => el.type), 'range');
  assert.deepEqual(errors, []);
  await context.close();
});

test('the top bar marks the section in view', async () => {
  const { page, context, errors } = await open();
  await page.$eval('#bench', (el) => el.scrollIntoView());
  await page.waitForTimeout(100);
  assert.equal(await page.getAttribute('.topbar nav a[aria-current="true"]', 'href'), '#bench');
  await page.$eval('#uno', (el) => el.scrollIntoView());
  await page.waitForTimeout(100);
  assert.equal(await page.getAttribute('.topbar nav a[aria-current="true"]', 'href'), '#uno');
  assert.deepEqual(errors, []);
  await context.close();
});
