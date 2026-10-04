// Headless browser regression tests. Use installed Chrome locally and bundled
// Chromium in CI. TEST_PAGE_URL also permits checking the published subpath.
import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { chromium } from 'playwright';

const here = path.dirname(fileURLToPath(import.meta.url));
const PAGE = process.env.TEST_PAGE_URL || pathToFileURL(path.join(here, '..', 'index.html')).href;
const SECTIONS = ['serial', 'round', 'parallel', 'sampling', 'speed', 'limits'];
let browser;

before(async () => {
  try { browser = await chromium.launch({ channel: 'chrome', headless: true }); }
  catch { browser = await chromium.launch({ headless: true }); }
});
after(async () => { if (browser) await browser.close(); });

async function open({ width = 1440, reduced = true, fakeClock = false } = {}) {
  const context = await browser.newContext({
    viewport: { width, height: 900 }, reducedMotion: reduced ? 'reduce' : 'no-preference',
  });
  await context.route(/fonts\.(googleapis|gstatic)\.com/, route =>
    route.fulfill({ status: 200, contentType: 'text/css', body: '' }));
  const page = await context.newPage();
  if (fakeClock) await page.clock.install();
  const errors = [];
  page.on('pageerror', error => errors.push('pageerror: ' + error.message));
  page.on('console', message => {
    if (message.type() === 'error') errors.push('console: ' + message.text());
  });
  await page.goto(PAGE);
  await page.waitForSelector('#speed-chart text');
  return { page, context, errors };
}

const text = (page, id) => page.locator('#' + id).innerText();
const normalized = value => value.replace(/\s+/g, ' ').trim();
const step = page => text(page, 'round-step');
const closeTo = (actual, expected, message = '') =>
  assert.ok(Math.abs(actual - expected) < 1e-10, `${message}: expected ${expected}, got ${actual}`);
const expectation = (alpha, k) => Array.from({ length: k + 1 }, (_, i) => alpha ** i).reduce((a, b) => a + b, 0);

async function tableValues(page) {
  return page.locator('#prob-table [data-token]').evaluateAll(rows => Object.fromEntries(rows.map(row => {
    const columns = ['target', 'draft', 'accepted', 'correction', 'output'];
    const cells = Array.from(row.querySelectorAll('td'));
    // A token may be a row header or the first ordinary cell.
    const values = cells.length === 6 ? cells.slice(1) : cells;
    return [row.dataset.token, Object.fromEntries(columns.map((name, index) => [name,
      (row.querySelector(`[data-col="${name}"]`) || values[index]).textContent.trim(),
    ]))];
  })));
}

function percent(value) {
  assert.match(value, /%/, `expected a percentage, got ${value}`);
  return Number(value.replace(/[^\d.+-]/g, ''));
}

for (const width of [1440, 390]) {
  test(`all figures render without errors or page overflow at ${width}px`, async () => {
    const { page, context, errors } = await open({ width });
    try {
      assert.deepEqual(await page.locator('[data-widget]').evaluateAll(els => els.map(el => el.dataset.widget)),
        ['round', 'probability', 'speed']);
      for (const id of SECTIONS) assert.equal(await page.locator(`section#${id}`).count(), 1, `missing section ${id}`);
      await page.evaluate(async () => {
        for (let y = 0; y < document.body.scrollHeight; y += 600) {
          window.scrollTo(0, y);
          await new Promise(resolve => setTimeout(resolve, 20));
        }
      });
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth), 0,
        'the page scrolls horizontally');
      const small = await page.locator('svg text').evaluateAll(els => els.filter(el =>
        el.textContent.trim() && el.getBoundingClientRect().width).map(el => {
        const svg = el.ownerSVGElement;
        const scale = svg.viewBox.baseVal.width ? svg.getBoundingClientRect().width / svg.viewBox.baseVal.width : 1;
        return [el.textContent, parseFloat(getComputedStyle(el).fontSize) * scale];
      }).filter(([, px]) => px < 10.95));
      assert.deepEqual(small, [], 'chart text must render at least 11px tall');
      const escaping = await page.locator('[data-widget]').evaluateAll(els => els.filter(el => {
        const r = el.getBoundingClientRect();
        return r.left < -1 || r.right > document.documentElement.clientWidth + 1;
      }).map(el => el.dataset.widget));
      assert.deepEqual(escaping, [], 'interactive figures must stay inside the viewport');
      const overlaps = await page.evaluate(() => {
        const notes = Array.from(document.querySelectorAll('.sidenote'));
        const figures = Array.from(document.querySelectorAll('.fig.wide'));
        return notes.flatMap((note, noteIndex) => figures.flatMap((figure, figureIndex) => {
          const a = note.getBoundingClientRect(), b = figure.getBoundingClientRect();
          const overlapX = Math.min(a.right, b.right) - Math.max(a.left, b.left);
          const overlapY = Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top);
          return overlapX > 1 && overlapY > 1 ? [{ noteIndex, figureIndex, overlapX, overlapY }] : [];
        }));
      });
      assert.deepEqual(overlaps, [], 'wide figures must clear floated sidenotes');
      const speedControls = await page.locator('.speed-controls .control').evaluateAll(els => els.map(el => {
        const slider = el.querySelector('input[type="range"]');
        const range = slider.getBoundingClientRect(), output = el.querySelector('output').getBoundingClientRect();
        const parent = el.getBoundingClientRect();
        return {
          id: slider.id, width: range.width,
          centerDifference: Math.abs((range.top + range.bottom) / 2 - (output.top + output.bottom) / 2),
          contained: range.left >= parent.left - 1 && output.right <= parent.right + 1,
          separate: output.left >= range.right - 1,
        };
      }));
      assert.equal(speedControls.length, 4);
      for (const control of speedControls) {
        assert.ok(control.width > 75, `${control.id}: usable slider width`);
        assert.ok(control.centerDifference <= 1.5, `${control.id}: slider and output must share a row`);
        assert.ok(control.contained && control.separate, `${control.id}: slider and output must fit without overlap`);
      }
      assert.deepEqual(errors, []);
    } finally { await context.close(); }
  });
}

test('the initial example is a labeled completed illustration and never starts itself', async () => {
  const { page, context, errors } = await open({ reduced: false, fakeClock: true });
  try {
    assert.equal(await page.locator('#round-case').inputValue(), 'third');
    assert.equal(await step(page), 'Step 6 of 6');
    assert.equal(await text(page, 'round-play'), 'Replay');
    assert.match(normalized(await text(page, 'round-output')), /We walked along the river/);
    assert.equal(Number(await text(page, 'round-committed')), 3);
    await page.locator('#round').scrollIntoViewIfNeeded();
    await page.clock.runFor(12000);
    assert.equal(await step(page), 'Step 6 of 6');
    assert.deepEqual(errors, []);
  } finally { await context.close(); }
});

test('round presets commit the accepted prefix and one replacement or bonus token', async () => {
  const { page, context, errors } = await open();
  try {
    const cases = [
      ['third', 'We walked along the river', 3],
      ['all', 'We walked along the river . Then', 5],
      ['first', 'We walked along', 1],
    ];
    for (const [value, output, committed] of cases) {
      await page.locator('#round-case').selectOption(value);
      assert.equal(await step(page), 'Step 6 of 6');
      assert.equal(await text(page, 'round-play'), 'Replay');
      assert.ok(normalized(await text(page, 'round-output')).includes(output), value + ': committed sequence');
      assert.equal(Number(await text(page, 'round-committed')), committed, value + ': committed count');
    }
    assert.deepEqual(errors, []);
  } finally { await context.close(); }
});

test('manual stepping reaches each state and stops at the completed round', async () => {
  const { page, context, errors } = await open();
  try {
    await page.locator('#round-reset').click();
    assert.equal(await step(page), 'Step 0 of 6');
    for (let n = 1; n <= 6; n++) {
      await page.locator('#round-next').click();
      assert.equal(await step(page), `Step ${n} of 6`);
    }
    assert.equal(await text(page, 'round-play'), 'Replay');
    assert.deepEqual(errors, []);
  } finally { await context.close(); }
});

test('reset during playback cancels the old timer and replay finishes only once', async () => {
  const { page, context, errors } = await open({ reduced: false, fakeClock: true });
  try {
    await page.locator('#round-play').click();
    assert.equal(await step(page), 'Step 1 of 6');
    await page.clock.runFor(1850);
    assert.equal(await step(page), 'Step 3 of 6');
    await page.locator('#round-reset').click();
    assert.equal(await step(page), 'Step 0 of 6');
    await page.clock.runFor(12000);
    assert.equal(await step(page), 'Step 0 of 6');
    await page.locator('#round-play').click();
    assert.equal(await step(page), 'Step 1 of 6');
    await page.clock.runFor(12000);
    assert.equal(await step(page), 'Step 6 of 6');
    assert.equal(await text(page, 'round-play'), 'Replay');
    assert.deepEqual(errors, []);
  } finally { await context.close(); }
});

test('switching an example mid-animation cancels playback and shows the new completed result', async () => {
  const { page, context, errors } = await open({ reduced: false, fakeClock: true });
  try {
    for (const [value, output, committed] of [['all', 'We walked along the river . Then', 5], ['first', 'We walked along', 1]]) {
      await page.locator('#round-play').click();
      await page.clock.runFor(950);
      assert.equal(await step(page), 'Step 2 of 6');
      await page.locator('#round-case').selectOption(value);
      assert.equal(await step(page), 'Step 6 of 6');
      await page.clock.runFor(12000);
      assert.equal(await step(page), 'Step 6 of 6');
      assert.equal(await text(page, 'round-play'), 'Replay');
      assert.ok(normalized(await text(page, 'round-output')).includes(output));
      assert.equal(Number(await text(page, 'round-committed')), committed);
    }
    assert.deepEqual(errors, []);
  } finally { await context.close(); }
});

test('reduced-motion playback immediately reaches its labeled final state', async () => {
  const { page, context, errors } = await open({ reduced: true, fakeClock: true });
  try {
    await page.locator('#round-reset').click();
    assert.equal(await step(page), 'Step 0 of 6');
    await page.locator('#round-play').click();
    assert.equal(await step(page), 'Step 6 of 6');
    await page.clock.runFor(12000);
    assert.equal(await step(page), 'Step 6 of 6');
    assert.equal(await text(page, 'round-play'), 'Replay');
    assert.deepEqual(errors, []);
  } finally { await context.close(); }
});

test('probability correction preserves target probabilities at the default and slider extremes', async () => {
  const { page, context, errors } = await open();
  try {
    const target = [.2, .5, .3];
    const tokens = ['lake', 'river', 'sea'];
    assert.equal(await page.locator('#q-lake').inputValue(), '50');
    assert.match(await text(page, 'prob-overlap'), /70\s*%/);
    for (const slider of [50, 0, 100, 20]) {
      await page.locator('#q-lake').fill(String(slider));
      const s = slider / 100, q = [s, (1 - s) * .6, (1 - s) * .4];
      const accepted = q.map((value, i) => Math.min(value, target[i]));
      const rejection = 1 - accepted.reduce((a, b) => a + b, 0);
      const values = await tableValues(page);
      for (let i = 0; i < tokens.length; i++) {
        const row = values[tokens[i]];
        for (const [key, value] of Object.entries({ target: target[i], draft: q[i], accepted: accepted[i], output: target[i] })) {
          assert.ok(Math.abs(percent(row[key]) - value * 100) <= .11, `${slider}, ${tokens[i]} ${key}`);
        }
        const correction = Math.max(target[i] - q[i], 0) / rejection;
        assert.ok(Math.abs(percent(row.correction) - correction * 100) <= .11, `${slider}, ${tokens[i]} correction`);
      }
      assert.ok(Math.abs(percent(await text(page, 'prob-overlap')) - (1 - rejection) * 100) <= .11);
    }
    assert.deepEqual(errors, []);
  } finally { await context.close(); }
});

test('the exact-match probability preset has no rejection and editing clears its selected state', async () => {
  const { page, context, errors } = await open();
  try {
    await page.locator('#prob-exact').click();
    assert.equal(await page.locator('#prob-exact').getAttribute('aria-pressed'), 'true');
    assert.match(await text(page, 'prob-overlap'), /100\s*%/);
    assert.match(await text(page, 'prob-message'), /No rejection/i);
    const values = await tableValues(page);
    for (const row of Object.values(values)) {
      assert.equal(percent(row.draft), percent(row.target));
      assert.equal(percent(row.output), percent(row.target));
      assert.match(row.correction, /—/);
    }
    await page.locator('#q-lake').fill('30');
    assert.notEqual(await page.locator('#prob-exact').getAttribute('aria-pressed'), 'true');
    await page.locator('#prob-default').click();
    assert.equal(await page.locator('#q-lake').inputValue(), '50');
    assert.match(await text(page, 'prob-overlap'), /70\s*%/);
    assert.deepEqual(errors, []);
  } finally { await context.close(); }
});

test('speed presets set the controls and report their actual modeled values', async () => {
  const { page, context, errors } = await open();
  try {
    assert.match(await text(page, 'speed-value'), /2\.40×/);
    assert.match(await text(page, 'tokens-value'), /3\.36/);
    assert.match(await text(page, 'cost-value'), /1\.40/);
    for (const [preset, alpha, k, c, v] of [
      ['balanced', .8, 4, .1, 1], ['poor', .2, 4, .1, 1], ['expensive', .8, 4, .5, 1.5],
    ]) {
      await page.locator(`[data-speed-preset="${preset}"]`).click();
      for (const [id, value] of [['acceptance', alpha * 100], ['draft-length', k], ['draft-cost', c * 100], ['verify-cost', v * 100]]) {
        assert.equal(Number(await page.locator('#' + id).inputValue()), value, `${preset}: ${id}`);
      }
      const cost = v + k * c, tokens = expectation(alpha, k);
      assert.ok((await text(page, 'speed-value')).includes((tokens / cost).toFixed(2) + '×'));
      assert.ok((await text(page, 'tokens-value')).includes(tokens.toFixed(2)));
      assert.ok((await text(page, 'cost-value')).includes(cost.toFixed(2)));
    }
    assert.deepEqual(errors, []);
  } finally { await context.close(); }
});

test('speed slider endpoints are finite and reflect acceptance, draft length, and verification cost', async () => {
  const { page, context, errors } = await open();
  try {
    for (const [alpha, k, c, v] of [[0, 1, 0, 1], [1, 12, 0, 1], [1, 12, 1, 3], [0, 12, 1, 3]]) {
      for (const [id, value] of [['acceptance', alpha * 100], ['draft-length', k], ['draft-cost', c * 100], ['verify-cost', v * 100]]) {
        await page.locator('#' + id).fill(String(value));
      }
      const cost = v + k * c, tokens = expectation(alpha, k);
      assert.ok((await text(page, 'speed-value')).includes((tokens / cost).toFixed(2) + '×'));
      assert.ok((await text(page, 'tokens-value')).includes(tokens.toFixed(2)));
      assert.ok((await text(page, 'cost-value')).includes(cost.toFixed(2)));
      assert.doesNotMatch(await page.locator('[data-widget="speed"]').innerText(), /NaN|Infinity/);
      if (alpha === 0 && c === 0 && v === 1) {
        assert.match(await text(page, 'speed-value'), /1\.00×/);
        assert.match(await text(page, 'speed-note'), /Same modeled latency as ordinary decoding\./);
      }
    }
    assert.deepEqual(errors, []);
  } finally { await context.close(); }
});

test('the math model handles equal and disjoint distributions and the acceptance endpoints', async () => {
  const { page, context, errors } = await open();
  try {
    const result = await page.evaluate(() => {
      const math = window.SpeculativeMath;
      return {
        same: math.residual([.2, .5, .3], [.2, .5, .3]),
        disjoint: math.residual([1, 0], [0, 1]),
        default: math.residual([.2, .5, .3], [.5, .3, .2]),
        expected: [math.expectedTokens(0, 4), math.expectedTokens(1, 4), math.expectedTokens(.8, 4)],
        speed: [math.speedup(0, 4, .1), math.speedup(1, 4, .1), math.speedup(.8, 4, .5, 1.5)],
      };
    });
    assert.deepEqual(result.same.accepted, [.2, .5, .3]);
    closeTo(result.same.rejection, 0, 'equal distributions');
    assert.equal(result.same.correction, null);
    assert.deepEqual(result.same.output, [.2, .5, .3]);
    assert.deepEqual(result.disjoint.accepted, [0, 0]);
    closeTo(result.disjoint.rejection, 1, 'disjoint distributions');
    assert.deepEqual(result.disjoint.correction, [1, 0]);
    assert.deepEqual(result.disjoint.output, [1, 0]);
    result.default.accepted.forEach((value, i) => closeTo(value, [.2, .3, .2][i]));
    closeTo(result.default.rejection, .3);
    result.default.correction.forEach((value, i) => closeTo(value, [0, 2 / 3, 1 / 3][i]));
    result.default.output.forEach((value, i) => closeTo(value, [.2, .5, .3][i]));
    result.expected.forEach((value, i) => closeTo(value, [1, 5, 3.3616][i]));
    result.speed.forEach((value, i) => closeTo(value, [1 / 1.4, 5 / 1.4, 3.3616 / 3.5][i]));
    assert.deepEqual(errors, []);
  } finally { await context.close(); }
});

test('native controls support keyboard interaction and visible focus', async () => {
  const { page, context, errors } = await open();
  try {
    await page.locator('#round-reset').focus();
    await page.keyboard.press('Enter');
    assert.equal(await step(page), 'Step 0 of 6');
    await page.locator('#round-next').focus();
    await page.keyboard.press('Space');
    assert.equal(await step(page), 'Step 1 of 6');
    await page.locator('#q-lake').focus();
    await page.keyboard.press('ArrowRight');
    assert.equal(await page.locator('#q-lake').inputValue(), '51');
    const focus = await page.locator('#q-lake').evaluate(el => {
      const css = getComputedStyle(el);
      return { active: document.activeElement === el, visible: (css.outlineStyle !== 'none' && parseFloat(css.outlineWidth) > 0) || css.boxShadow !== 'none' };
    });
    assert.deepEqual(focus, { active: true, visible: true });
    const inaccessible = await page.locator('[data-widget] button, [data-widget] input, [data-widget] select').evaluateAll(els =>
      els.filter(el => !el.disabled && el.tabIndex < 0).map(el => el.id));
    assert.deepEqual(inaccessible, []);
    assert.deepEqual(errors, []);
  } finally { await context.close(); }
});

test('section navigation identifies the section currently being read', async () => {
  const { page, context, errors } = await open();
  try {
    await page.evaluate(() => window.scrollTo(0, document.getElementById('sampling').offsetTop + 180));
    await page.waitForFunction(() => document.querySelector('.topbar nav a[aria-current="true"]')?.getAttribute('href') === '#sampling');
    assert.equal(await page.locator('.topbar nav a[aria-current="true"]').count(), 1);
    assert.deepEqual(errors, []);
  } finally { await context.close(); }
});
