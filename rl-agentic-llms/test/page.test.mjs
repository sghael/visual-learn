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
 *  D3 still loads from cdnjs, so the tests need network for that one script. */
async function open({ width = 1440, reduced = true } = {}) {
  const context = await browser.newContext({ viewport: { width, height: 900 }, reducedMotion: reduced ? 'reduce' : 'no-preference' });
  await context.route(/fonts\.(googleapis|gstatic)\.com/, (route) => route.fulfill({ status: 200, contentType: 'text/css', body: '' }));
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', (e) => errors.push('pageerror: ' + e.message));
  page.on('console', (m) => { if (m.type() === 'error') errors.push('console: ' + m.text()); });
  await page.goto(PAGE);
  await page.waitForSelector('#labs-grid .lab');
  return { page, context, errors };
}

for (const width of [1440, 390]) {
  test(`page renders every widget without errors or horizontal overflow at ${width}px`, async () => {
    const { page, context, errors } = await open({ width });
    const widgets = await page.$$eval('[data-widget]', (els) => els.map((el) => [el.dataset.widget, el.children.length]));
    assert.equal(widgets.length, 15);
    for (const [name, children] of widgets) assert.ok(children > 0, `${name} rendered nothing`);
    await page.evaluate(async () => { for (let y = 0; y < document.body.scrollHeight; y += 600) { window.scrollTo(0, y); await new Promise((r) => setTimeout(r, 25)); } });
    await page.waitForTimeout(300);
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
    assert.equal(overflow, 0, 'page scrolls horizontally');
    assert.deepEqual(errors, []);
    await context.close();
  });
}

test('GRPO: a group with no variance yields zero advantage everywhere, and re-toggling restores signal', async () => {
  const { page, context, errors } = await open();
  await page.uncheck('#grpo-format'); // the format bonus alone would keep the group from being zero-variance
  const wrong = page.locator('#grpo-samples .tog.bad');
  while (await wrong.count()) await wrong.first().click(); // make every rollout correct
  assert.equal(await page.locator('#grpo-signal').innerText(), 'none');
  const advs = await page.$$eval('#grpo-samples .adv', (els) => els.map((e) => e.innerText.trim()));
  assert.deepEqual(advs, advs.map(() => '+0.00'));
  await page.locator('#grpo-samples .tog').first().click();
  assert.notEqual(await page.locator('#grpo-signal').innerText(), 'none');
  const first = await page.locator('#grpo-samples .adv').first().innerText();
  assert.match(first, /^-/, 'the lone wrong rollout gets a negative advantage');
  assert.deepEqual(errors, []);
  await context.close();
});

test('policy gradient: a positive-advantage update raises the sampled action and lowers the others', async () => {
  const { page, context, errors } = await open();
  await page.selectOption('#pg-action', '2');
  await page.click('#pg-step');
  await page.waitForTimeout(400);
  const vals = await page.$$eval('#pg-viz .val', (els) => els.map((e) => +e.textContent));
  assert.ok(vals[2] > 0.2 && vals[0] < 0.2 && vals[4] < 0.2, `unexpected distribution ${vals}`);
  assert.equal(await page.locator('#pg-n').innerText(), '1');
  await page.click('#pg-reset');
  assert.equal(await page.locator('#pg-n').innerText(), '0');
  assert.deepEqual(errors, []);
  await context.close();
});

test('trajectory: flipping the outcome flips the sign of every policy-token advantage; observations stay masked', async () => {
  const { page, context, errors } = await open();
  const creds = () => page.$$eval('#traj .turn.policy .cred', (els) => els.map((e) => e.innerText.trim()));
  assert.ok((await creds()).every((c) => c.startsWith('A=+')));
  await page.locator('#traj-outcome button', { hasText: 'fail' }).click();
  assert.ok((await creds()).every((c) => c.startsWith('A=-')));
  assert.equal(await page.locator('#traj .turn.env .cred', { hasText: 'masked' }).count(), 3, 'the three tool outputs are masked; the grader turn shows R');
  await page.locator('#traj-credit button', { hasText: 'shaping' }).click();
  assert.match(await page.locator('#traj-adv').innerText(), /\+0\.10 per valid tool call/);
  assert.deepEqual(errors, []);
  await context.close();
});

test('RLVR run: reset during training stops the run, clears the curves, and restores the caption', async () => {
  const { page, context, errors } = await open({ reduced: false });
  const caption = await page.locator('#rlvr-msg').innerText();
  await page.click('#rlvr-play');
  await page.waitForFunction(() => +document.querySelector('#rlvr-step').textContent >= 4050, null, { timeout: 15000 });
  assert.notEqual(await page.locator('#rlvr-msg').innerText(), caption, 'caption changes after step 4000');
  await page.click('#rlvr-reset');
  assert.equal(await page.locator('#rlvr-step').innerText(), '0');
  assert.equal(await page.locator('#rlvr-msg').innerText(), caption);
  assert.equal(await page.locator('#rlvr-play').innerText(), '▶ Train');
  await page.waitForTimeout(300);
  assert.equal(await page.locator('#rlvr-step').innerText(), '0', 'timer really stopped');
  assert.deepEqual(errors, []);
  await context.close();
});

test('reward model: preferring A twice moves P(A ≻ B) above 0.5 and reset returns it', async () => {
  const { page, context, errors } = await open();
  await page.locator('#rlhf-pair button', { hasText: 'Prefer A' }).click();
  await page.locator('#rlhf-pair button', { hasText: 'Prefer A' }).click();
  assert.ok(+(await page.locator('#rlhf-p').innerText()) > 0.6);
  await page.click('#rlhf-reset');
  assert.equal(await page.locator('#rlhf-p').innerText(), '0.50');
  assert.deepEqual(errors, []);
  await context.close();
});
