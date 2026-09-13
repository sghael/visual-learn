// Browser regression tests. Drives index.html in headless Chrome (the installed
// Google Chrome via Playwright's "chrome" channel, falling back to Playwright's
// own Chromium). Run with `pnpm test`. D3 is served from a local stub route so
// the tests do not depend on cdnjs being reachable.
import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { chromium } from 'playwright';

const here = path.dirname(fileURLToPath(import.meta.url));
const PAGE = pathToFileURL(path.join(here, '..', 'index.html')).href;
const D3_CACHE = path.join(here, '.d3.min.js');
const D3_URL = 'https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js';
const STAGES = ['fleetChart', 'layerDiagram', 'movaDiagram', 'unoDiagram', 'trainDiagram', 'dataDiagram', 'benchChart'];
let browser, d3src;

before(async () => {
  try { browser = await chromium.launch({ channel: 'chrome', headless: true }); }
  catch { browser = await chromium.launch({ headless: true }); }
  if (!fs.existsSync(D3_CACHE)) fs.writeFileSync(D3_CACHE, await (await fetch(D3_URL)).text());
  d3src = fs.readFileSync(D3_CACHE, 'utf8');
});
after(async () => { if (browser) await browser.close(); });

async function open({ width = 1440, reduced = true } = {}) {
  const context = await browser.newContext({ viewport: { width, height: 900 }, reducedMotion: reduced ? 'reduce' : 'no-preference' });
  await context.route(/fonts\.(googleapis|gstatic)\.com/, (route) => route.fulfill({ status: 200, contentType: 'text/css', body: '' }));
  await context.route(D3_URL, (route) => route.fulfill({ status: 200, contentType: 'application/javascript', body: d3src }));
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', (e) => errors.push('pageerror: ' + e.message));
  page.on('console', (m) => { if (m.type() === 'error') errors.push('console: ' + m.text()); });
  await page.goto(PAGE);
  await page.waitForSelector('#benchChart svg');
  return { page, context, errors };
}
const svgCount = (page, id) => page.$eval('#' + id, (el) => el.querySelectorAll('svg *').length);

for (const width of [1440, 390]) {
  test(`every stage renders without errors or horizontal overflow at ${width}px`, async () => {
    const { page, context, errors } = await open({ width });
    for (const id of STAGES) assert.ok((await svgCount(page, id)) > 10, `${id} rendered nothing`);
    await page.evaluate(async () => { for (let y = 0; y < document.body.scrollHeight; y += 600) { window.scrollTo(0, y); await new Promise((r) => setTimeout(r, 20)); } });
    await page.waitForTimeout(300);
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
    assert.equal(overflow, 0, 'page scrolls horizontally');
    assert.deepEqual(errors, []);
    await context.close();
  });
}

test('selecting a model updates the spec sheet and the fleet chart highlight', async () => {
  const { page, context, errors } = await open();
  await page.locator('#modelChips button', { hasText: '375B-A23B' }).click();
  assert.equal(await page.locator('#specTitle').innerText(), 'K2-Horizon-375B-A23B');
  const grid = await page.locator('#specGrid').innerText();
  assert.match(grid, /192 \+ 1 shared, top-8/);
  assert.match(grid, /524,288 tokens/);
  await page.locator('#modelChips button', { hasText: '0.9B' }).click();
  assert.match(await page.locator('#specGrid').innerText(), /64,256/);
  assert.match(await page.locator('#specGrid').innerText(), /131,072 tokens/);
  assert.deepEqual(errors, []);
  await context.close();
});

test('layer variants redraw with the right expert grids and routing lights experts', async () => {
  const { page, context, errors } = await open();
  const count = (sel) => page.$$eval(sel, (els) => els.length);
  assert.equal(await count('#layerDiagram .ex'), 0, 'dense has no experts');
  await page.locator('[data-variant="moe"]').click();
  assert.equal(await count('#layerDiagram .ex'), 192);
  assert.equal(await count('#layerDiagram .vx'), 0);
  await page.locator('[data-variant="mova"]').click();
  assert.equal(await count('#layerDiagram .ex'), 100);
  assert.equal(await count('#layerDiagram .vx'), 16);
  await page.locator('#routeBtn').click();
  await page.waitForTimeout(300);
  const lit = await page.$$eval('#layerDiagram .ex', (els) => els.filter((e) => e.getAttribute('stroke-width') === '1.5').length);
  assert.equal(lit, 8, 'exactly eight experts active');
  assert.match(await page.locator('#layerCap').innerText(), /4 of 64 value experts/);
  assert.deepEqual(errors, []);
  await context.close();
});

test('MoVA sliders change the pool, the active count never exceeds the pool, and stats follow', async () => {
  const { page, context, errors } = await open();
  await page.locator('#mvExperts').fill('8');
  await page.locator('#mvTop').fill('8');
  const rects = await page.$$eval('#movaDiagram rect[stroke-width="1.5"]', (els) => els.length);
  assert.equal(rects, 8);
  assert.match(await page.locator('#movaStat').innerText(), /8×[\s\S]*8×[\s\S]*1:1/);
  await page.locator('#mvExperts').fill('128');
  await page.locator('#mvTop').fill('2');
  assert.match(await page.locator('#movaStat').innerText(), /128×[\s\S]*2×[\s\S]*64:1/);
  assert.deepEqual(errors, []);
  await context.close();
});

test('Uno: reset mid-run stops the old timer, and block size applies to the next draft', async () => {
  const { page, context, errors } = await open();
  const counts = () => page.$eval('#unoDiagram', (el) => [...el.querySelectorAll('text')].map((t) => t.textContent).filter((t) => /tokens ·/.test(t)));
  const before = await counts();
  assert.match(before[1], /^[1-9]\d*\/\d+ tokens/, 'Uno lane shows progress at rest');
  await page.locator('#unoPlay').click();
  await page.waitForTimeout(1000);
  await page.locator('#unoReset').click();
  const atReset = await counts();
  assert.match(atReset[0], /^0\/\d+ tokens · 0 forward passes/);
  assert.equal(await page.locator('#unoPlay').innerText(), '▶ Play');
  await page.waitForTimeout(1200);
  assert.deepEqual(await counts(), atReset, 'a cancelled run kept advancing after reset');
  await page.locator('#unoBlock').fill('8');
  assert.equal(await page.locator('#unoBlockN').innerText(), '8');
  await page.locator('#unoPlay').click();
  await page.waitForTimeout(700);
  await page.locator('#unoPlay').click(); // pause
  const paused = await counts();
  assert.match(paused[0], /^[1-9]\d*\//);
  await page.waitForTimeout(1000);
  assert.deepEqual(await counts(), paused, 'pause did not stop the run');
  assert.deepEqual(errors, []);
  await context.close();
});

test('benchmark model select redraws with that model card\'s competitors', async () => {
  const { page, context, errors } = await open();
  await page.selectOption('#benchModel', '375B-A23B');
  assert.match(await page.locator('#benchLegend').innerText(), /Claude Sonnet 5/);
  assert.equal(await page.$$eval('#benchChart rect', (els) => els.length), 24, '6 rows × 4 competitors');
  await page.selectOption('#benchModel', '3.7B');
  assert.equal(await page.$$eval('#benchChart rect', (els) => els.length), 10, '5 rows × 2 competitors');
  assert.deepEqual(errors, []);
  await context.close();
});

test('theme toggle redraws every chart and persists', async () => {
  const { page, context, errors } = await open();
  await page.locator('#themeBtn').click();
  assert.equal(await page.locator('#themeBtn').innerText(), 'Theme: light');
  await page.locator('#themeBtn').click();
  assert.equal(await page.getAttribute('html', 'data-theme'), 'dark');
  for (const id of STAGES) assert.ok((await svgCount(page, id)) > 10, `${id} empty after theme change`);
  assert.equal(await page.evaluate(() => localStorage.getItem('k2theme')), 'dark');
  assert.deepEqual(errors, []);
  await context.close();
});
