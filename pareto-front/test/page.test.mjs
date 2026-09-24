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

async function open(width = 1440) {
  const context = await browser.newContext({ viewport: { width, height: 900 }, reducedMotion: 'reduce' });
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', e => errors.push(e.message));
  page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
  await page.goto(PAGE);
  await page.waitForSelector('#model-picker option:nth-child(2)', { state: 'attached' });
  return { context, page, errors };
}

for (const width of [1440, 390]) {
  test(`chart and controls render at ${width}px`, async () => {
    const { context, page, errors } = await open(width);
    assert.equal(await page.locator('#date').textContent(), '2026-09-23');
    assert.equal(await page.locator('#play').innerText(), 'Play');
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth), 0);
    await page.screenshot({ path: `/private/tmp/pareto-front-${width}.png`, fullPage: true });
    assert.deepEqual(errors, []);
    await context.close();
  });
}

test('date slider updates visible models and its spoken date', async () => {
  const { context, page, errors } = await open();
  const first = await page.evaluate(() => DATA.frames[0]);
  await page.locator('#scrub').focus();
  await page.locator('#scrub').press('Home');
  assert.equal(await page.locator('#date').textContent(), first.date);
  assert.equal(await page.locator('#scrub').getAttribute('aria-valuetext'), first.date);
  assert.equal(await page.locator('#model-picker option').count(), first.points.length + 1);
  assert.deepEqual(errors, []);
  await context.close();
});

test('clicking a dot selects it, and a later release disappears at an earlier date', async () => {
  const { context, page, errors } = await open();
  await page.locator('#scrub').focus();
  await page.locator('#scrub').press('Home');
  const first = await page.evaluate(() => {
    const p = DATA.frames[0].points[0];
    return { id: p.id, name: models[p.id].name, point: xy(p.cost, p.score) };
  });
  const box = await page.locator('#chart').boundingBox();
  await page.mouse.click(box.x + first.point[0] * box.width / 1200, box.y + first.point[1] * box.height / 760);
  assert.equal(await page.locator('#model-picker').inputValue(), first.id);
  assert.ok((await page.locator('#model-detail').innerText()).includes(first.name));
  assert.equal(await page.locator('#play').innerText(), 'Play');

  await page.locator('#scrub').press('End');
  assert.equal(await page.locator('#model-picker').inputValue(), first.id);
  const later = await page.evaluate(() => DATA.frames.at(-1).points.find(p => !DATA.frames[0].points.some(q => q.id === p.id)).id);
  await page.locator('#model-picker').selectOption(later);
  assert.equal(await page.locator('#model-picker').inputValue(), later);
  await page.locator('#scrub').press('Home');
  assert.equal(await page.locator('#model-picker').inputValue(), '');
  assert.match(await page.locator('#model-detail').innerText(), /Select a dot/);
  assert.deepEqual(errors, []);
  await context.close();
});
