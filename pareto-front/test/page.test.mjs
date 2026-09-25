import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { chromium } from 'playwright';

const here = path.dirname(fileURLToPath(import.meta.url));
const PAGE = pathToFileURL(path.join(here, '..', 'index.html')).href;
const DATA = JSON.parse(fs.readFileSync(path.join(here, '..', 'data.json'), 'utf8'));
const byId = Object.fromEntries(DATA.models.map(m => [m.id, m]));
const LATEST = 'Sep 23, 2026';
let browser;

before(async () => {
  try { browser = await chromium.launch({ channel: 'chrome', headless: true }); }
  catch { browser = await chromium.launch({ headless: true }); }
});
after(async () => { if (browser) await browser.close(); });

/** Opens the page. With fakeClock, page timers only advance through page.clock.runFor(). */
async function open({ width = 1440, reduced = true, fakeClock = false } = {}) {
  const context = await browser.newContext({ viewport: { width, height: 900 }, reducedMotion: reduced ? 'reduce' : 'no-preference' });
  await context.route(/fonts\.(googleapis|gstatic)\.com/, route => route.fulfill({ status: 200, contentType: 'text/css', body: '' }));
  const page = await context.newPage();
  if (fakeClock) await page.clock.install();
  const errors = [];
  page.on('pageerror', e => errors.push('pageerror: ' + e.message));
  page.on('console', m => { if (m.type() === 'error') errors.push('console: ' + m.text()); });
  await page.goto(PAGE);
  await page.waitForSelector('#chart svg circle.fdot');
  await page.waitForSelector('#multiples svg');
  return { context, page, errors };
}
const dateText = page => page.locator('#date').textContent();

function frontierOf(points) {
  const ranked = points.slice().sort((a, b) => a.cost - b.cost || b.score - a.score);
  const out = []; let top = -Infinity;
  for (const p of ranked) if (p.score > top + 1e-9) { out.push(p.id); top = p.score; }
  return out;
}

for (const width of [1440, 390]) {
  test(`resting state at ${width}px: latest date, labeled frontier, no autoplay, no overflow`, async (t) => {
    const { context, page, errors } = await open({ width, reduced: false });
    t.after(() => context.close());
    assert.equal(await dateText(page), LATEST);
    assert.equal(await page.locator('#play').innerText(), 'Play');
    assert.equal(await page.locator('#reset').isDisabled(), true);
    const last = DATA.frames.at(-1);
    assert.equal(await page.locator('#chart circle.fdot').count(), last.frontier.length);
    assert.equal(await page.locator('#chart circle.dot').count(), last.count - last.frontier.length);
    assert.ok(await page.locator('#chart text.label').count() >= 5, 'frontier points carry direct labels');
    assert.equal(await page.locator('#multiples svg').count(), DATA.snapshots.length);
    // No motion without a button press.
    await page.waitForTimeout(700);
    assert.equal(await dateText(page), LATEST);
    // Chart text stays legible, and the page never scrolls sideways.
    const smallest = await page.$$eval('svg text', ts => Math.min(...ts.map(t => parseFloat(getComputedStyle(t).fontSize))));
    assert.ok(smallest >= 11, `smallest chart text is ${smallest}px`);
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth), 0);
    assert.deepEqual(errors, []);
    await page.screenshot({ path: path.join(os.tmpdir(), `pareto-front-${width}.png`), fullPage: true });
  });
}

test('generated frontiers match a fresh calculation from the model list', () => {
  for (const f of [...DATA.frames, ...DATA.snapshots]) {
    const pts = DATA.models.filter(m => m.release <= f.date);
    assert.equal(f.count, pts.length, f.date);
    assert.deepEqual(f.frontier, frontierOf(pts), f.date);
  }
  assert.equal(DATA.frames.at(-1).date, DATA.end);
  assert.equal(DATA.snapshots.at(-1).date, DATA.end);
});

test('numbers in the prose and captions match the data', async (t) => {
  const { context, page, errors } = await open();
  t.after(() => context.close());
  const text = await page.locator('main').innerText();
  const last = DATA.frames.at(-1);
  const front = last.frontier.map(id => byId[id]);
  assert.match(text, new RegExp(`for ${DATA.models.length} models`));
  assert.match(text, new RegExp(`holds ${DATA.excluded} retired models`));
  assert.match(text, new RegExp(`of the ${front.length} frontier points`));
  // The most expensive model and the cheapest frontier model that beats it.
  const priciest = DATA.models.reduce((a, b) => (b.cost > a.cost ? b : a));
  assert.equal(priciest.name, 'Claude Fable 5 (max, Opus 4.8 fallback)');
  assert.match(text, new RegExp(`scores ${priciest.score} for \\$${priciest.cost.toFixed(2)} per task`));
  const beat = front.find(p => p.score >= priciest.score);
  assert.equal(beat.name, 'Claude Opus 5.5 (medium)');
  assert.match(text, new RegExp(`scores ${beat.score} for \\$${beat.cost.toFixed(2)}, so Fable 5`));
  // Figure 1 takeaway: points gained up to MiMo-V2.6-Pro, then points and cost ratio after it.
  const mimo = front.find(p => p.name === 'MiMo-V2.6-Pro'), top = front.at(-1);
  assert.equal(Math.round(mimo.score - front[0].score), 25);
  assert.equal(Math.round(top.score - mimo.score), 11);
  assert.equal(Math.round(top.cost / mimo.cost), 45);
  assert.match(text, /gains 25 points; the next 11 points cost 45 times as much/);
  // Figure 2: top scores and the 1/120 cost comparison.
  const snap = d => DATA.snapshots.find(s => s.date === d).frontier.map(id => byId[id]);
  assert.equal(Math.max(...snap('2025-09-30').map(p => p.score)), 20.7);
  assert.equal(Math.max(...snap(DATA.end).map(p => p.score)), 57.6);
  const cheapest20 = d => snap(d).find(p => p.score > 20);
  assert.equal(cheapest20('2025-09-30').name, 'Claude 4.5 Sonnet (Reasoning)');
  assert.equal(cheapest20(DATA.end).name, 'GPT-6 Luna (low)');
  assert.equal(Math.round(cheapest20('2025-09-30').cost / cheapest20(DATA.end).cost / 10) * 10, 120);
  const late = front.filter(p => p.release >= '2026-09-21').length;
  assert.equal(late, 11);
  assert.match(text, /Eleven of the 13 frontier models/);
  assert.deepEqual(errors, []);
});

test('date slider changes the visible models and its spoken value', async (t) => {
  const { context, page, errors } = await open();
  t.after(() => context.close());
  const first = DATA.frames[0];
  await page.locator('#scrub').focus();
  await page.locator('#scrub').press('Home');
  assert.equal(await dateText(page), 'Dec 26, 2024');
  assert.match(await page.locator('#scrub').getAttribute('aria-valuetext'), /^Dec 26, 2024, 1 models$/);
  assert.equal(await page.locator('#model-picker option').count(), first.count + 1);
  assert.equal(await page.locator('#chart circle.fdot').count(), first.frontier.length);
  assert.equal(await page.locator('#reset').isDisabled(), false);
  await page.locator('#reset').click();
  assert.equal(await dateText(page), LATEST);
  assert.equal(await page.locator('#model-picker option').count(), DATA.frames.at(-1).count + 1);
  assert.deepEqual(errors, []);
});

test('clicking a dot selects it, and a later release drops out at an earlier date', async (t) => {
  const { context, page, errors } = await open();
  t.after(() => context.close());
  await page.locator('#scrub').focus();
  await page.locator('#scrub').press('Home');
  const first = byId[DATA.frames[0].frontier[0]];
  const dot = page.locator('#chart circle.fdot').first();
  await dot.scrollIntoViewIfNeeded();
  const box = await dot.boundingBox();
  await page.mouse.click(box.x + box.width / 2, box.y + box.height / 2);
  assert.equal(await page.locator('#model-picker').inputValue(), first.id);
  const detail = await page.locator('#model-detail').innerText();
  assert.ok(detail.includes(first.name));
  assert.match(detail, /On the frontier/);
  assert.equal(await page.locator('#chart circle.ring').count(), 1);
  assert.equal(await page.locator('#play').innerText(), 'Play');

  await page.locator('#scrub').press('End');
  assert.equal(await page.locator('#model-picker').inputValue(), first.id, 'selection survives while the model is visible');
  const later = DATA.models.find(m => m.release > DATA.frames[0].date);
  await page.locator('#model-picker').selectOption(later.id);
  assert.equal(await page.locator('#model-picker').inputValue(), later.id);
  await page.locator('#scrub').press('Home');
  assert.equal(await page.locator('#model-picker').inputValue(), '');
  assert.match(await page.locator('#model-detail').innerText(), /Select a dot/);
  assert.equal(await page.locator('#chart circle.ring').count(), 0);
  assert.deepEqual(errors, []);
});

test('a dominated model names the frontier model that beats it and links to it', async (t) => {
  const { context, page, errors } = await open({ width: 390 });
  t.after(() => context.close());
  await page.locator('#model-picker').selectOption('claude-fable-5');
  const detail = await page.locator('#model-detail').innerText();
  assert.match(detail, /^Claude Fable 5 \(max, Opus 4\.8 fallback\) \(Anthropic, released Jun 9, 2026\) scores 49\.6 for \$8\.75 per task\./);
  assert.match(detail, /Off the frontier: Claude Opus 5\.5 \(medium\) scores 51\.2 for \$1\.34\./);
  assert.equal(await page.locator('#chart circle.ring').count(), 2);
  assert.equal(await page.locator('#chart line.link').count(), 1);
  assert.ok(await page.locator('#chart text.label.strong', { hasText: 'Claude Fable 5' }).count() === 1, 'selected model is labeled on the chart');
  // A tap on empty space (left of the cheapest model) clears the selection.
  await page.locator('#chart svg').scrollIntoViewIfNeeded();
  const svg = await page.locator('#chart svg').boundingBox();
  await page.mouse.click(svg.x + 42, svg.y + svg.height / 2);
  assert.equal(await page.locator('#model-picker').inputValue(), '');
  assert.deepEqual(errors, []);
});

test('Play steps through the dates from the start, Pause holds, and the run stops at the latest date', async (t) => {
  const { context, page, errors } = await open({ reduced: false, fakeClock: true });
  t.after(() => context.close());
  await page.locator('#play').click();
  assert.equal(await page.locator('#play').innerText(), 'Pause');
  assert.equal(await dateText(page), 'Dec 26, 2024', 'Play from the resting state restarts at the first date');
  await page.clock.runFor(180 * 3 + 10);
  assert.equal(await page.locator('#scrub').inputValue(), '3');
  await page.locator('#play').click();
  assert.equal(await page.locator('#play').innerText(), 'Play');
  await page.clock.runFor(2000);
  assert.equal(await page.locator('#scrub').inputValue(), '3', 'paused playback does not advance');
  await page.locator('#play').click(); // resume from where it paused
  await page.clock.runFor(180 * DATA.frames.length + 1000);
  assert.equal(await dateText(page), LATEST);
  assert.equal(await page.locator('#play').innerText(), 'Play');
  await page.clock.runFor(2000);
  assert.equal(await dateText(page), LATEST, 'nothing fires after the run ends');
  assert.deepEqual(errors, []);
});

test('Reset and the slider cancel a run in progress; no stale timer moves the date afterwards', async (t) => {
  const { context, page, errors } = await open({ reduced: false, fakeClock: true });
  t.after(() => context.close());
  await page.locator('#play').click();
  await page.clock.runFor(100); // parked inside the first step's wait
  await page.locator('#reset').click();
  assert.equal(await dateText(page), LATEST);
  assert.equal(await page.locator('#play').innerText(), 'Play');
  await page.clock.runFor(5000);
  assert.equal(await dateText(page), LATEST, 'the cancelled run resumed after Reset');

  await page.locator('#play').click();
  await page.clock.runFor(180 * 5 + 50);
  await page.locator('#model-picker').selectOption(DATA.frames[5].frontier[0]); // select mid-run
  assert.equal(await page.locator('#play').innerText(), 'Pause', 'choosing a model does not stop the run');
  await page.locator('#scrub').focus();
  await page.locator('#scrub').press('ArrowRight'); // scrubbing mid-run takes over
  const held = await page.locator('#scrub').inputValue();
  assert.equal(await page.locator('#play').innerText(), 'Play');
  await page.clock.runFor(5000);
  assert.equal(await page.locator('#scrub').inputValue(), held, 'a stale timer moved the slider');
  assert.deepEqual(errors, []);
});
