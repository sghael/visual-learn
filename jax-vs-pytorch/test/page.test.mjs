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

/** Open the page; collect page errors and console errors. Web fonts are stubbed so the tests run offline.
 *  With fakeClock, page timers and animation frames only advance through page.clock.runFor(). */
async function open({ width = 1440, reduced = true, fakeClock = false } = {}) {
  const context = await browser.newContext({ viewport: { width, height: 900 }, reducedMotion: reduced ? 'reduce' : 'no-preference' });
  await context.route(/fonts\.(googleapis|gstatic)\.com/, (route) => route.fulfill({ status: 200, contentType: 'text/css', body: '' }));
  const page = await context.newPage();
  if (fakeClock) await page.clock.install();
  const errors = [];
  page.on('pageerror', (e) => errors.push('pageerror: ' + e.message));
  page.on('console', (m) => { if (m.type() === 'error') errors.push('console: ' + m.text()); });
  await page.goto(PAGE);
  await page.waitForSelector('#w-sharding .stage-body');
  return { page, context, errors };
}

for (const width of [1440, 390]) {
  test(`page renders every widget without errors or horizontal overflow at ${width}px`, async () => {
    const { page, context, errors } = await open({ width });
    const widgets = await page.$$eval('[data-widget]', (els) => els.map((el) => [el.dataset.widget, el.querySelector('.stage-body') ? el.querySelector('.stage-body').children.length : 0]));
    assert.equal(widgets.length, 9);
    for (const [name, children] of widgets) assert.ok(children > 0, `${name} rendered nothing`);
    // scroll through the page so visibility-triggered animations start
    await page.evaluate(async () => { for (let y = 0; y < document.body.scrollHeight; y += 600) { window.scrollTo(0, y); await new Promise((r) => setTimeout(r, 25)); } });
    await page.waitForTimeout(400);
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
    assert.equal(overflow, 0, 'page scrolls horizontally');
    assert.deepEqual(errors, []);
    await context.close();
  });
}

test('switching tracing examples during autoplay neither throws nor keeps the old stepper alive', async () => {
  const { page, context, errors } = await open({ reduced: false });
  await page.locator('#w-tracing').scrollIntoViewIfNeeded(); // autoplay starts on visibility
  const readout = page.locator('#w-tracing .stepper .readout');
  await page.waitForFunction(() => /step <b>[1-9]/.test(document.querySelector('#w-tracing .stepper .readout').innerHTML), null, { timeout: 10000 });
  await page.locator('#w-tracing .stage-head .seg button', { hasText: 'Python if' }).click();
  await page.waitForTimeout(2500); // longer than one autoplay tick of the old stepper
  assert.match(await readout.innerText(), /step 0 \/ 3/);
  assert.deepEqual(errors, []);
  await context.close();
});

test('a hero run cancelled by reset cannot touch the replacement run, even when its timers resume', async () => {
  // Fake time makes the race deterministic: every wait and animation frame in the
  // widget runs on the page clock, so we can park run A inside a wait, start run B,
  // and then let A's timer fire while B is still in its first phase.
  const { page, context, errors } = await open({ reduced: false, fakeClock: true });
  await page.locator('#w-pipeline').scrollIntoViewIfNeeded();
  const controls = page.locator('#w-pipeline .stage-head .controls .btn');
  const cls = (id) => page.$eval(`#w-pipeline .box.${id}`, (el) => el.getAttribute('class'));
  const boxes = () => page.$$eval('#w-pipeline .box', (els) => els.map((el) => el.getAttribute('class')).join('|'));

  await controls.nth(2).click(); // Reset: cancels the visibility-triggered run, if any
  await controls.nth(0).click(); // run A starts at clock time 0
  await page.clock.runFor(100);  // A is parked in its first waits: torch lane fires at 300, jax lane at 420
  await controls.nth(2).click(); // Reset while A is mid-flight
  await controls.nth(0).click(); // run B starts at 100; its torch wait fires at 400, jax wait at 520
  await page.clock.runFor(250);  // now 350: A's torch timer fired, B's has not
  const tPy = await cls('torch');
  assert.match(tPy, /\bon\b/, 'run A resumed after reset and cleared the Python-call box that run B owns');
  assert.doesNotMatch(tPy, /\bdone\b/);
  await page.clock.runFor(100);  // now 450: A's jax timer fired, B's has not
  const jPy = await cls('jax');
  assert.match(jPy, /\bon\b/, 'run A resumed after reset and advanced the jax lane that run B owns');
  assert.doesNotMatch(jPy, /\bdone\b/);

  await page.clock.runFor(30000); // let B finish
  assert.equal(await page.$$eval('#w-pipeline .stage-head .controls .btn', (els) => els[1].disabled), false, 'run B did not complete');
  const counts = await page.locator('#w-pipeline .counts').innerText();
  for (const re of [/traced 1×/, /compiled 1×/, /executed 1×/, /calls 1\b/, /kernel launches 3\b/]) assert.match(counts, re);
  const settled = await boxes();
  await page.clock.runFor(5000); // nothing else may fire
  assert.equal(await boxes(), settled);
  assert.equal(await page.locator('#w-pipeline .counts').innerText(), counts);
  assert.deepEqual(errors, []);
  await context.close();
});

test('tracing call buttons and log use each example\'s real arity', async () => {
  const { page, context, errors } = await open();
  const buttons = () => page.locator('#w-tracing .callbtns .btn').allInnerTexts();
  assert.deepEqual((await buttons()).slice(0, 3), ['f(ones(3), ones(3))', 'f(ones(4), ones(4))', 'f(ones(3), ones(3)) again']);
  await page.locator('#w-tracing .callbtns .btn').nth(0).click();
  let log = await page.locator('#w-tracing .log').innerText();
  assert.match(log, /f\(ones\(3\), ones\(3\)\)/);
  assert.match(log, /\(f32\[3\], f32\[3\]\)/);
  assert.match(log, /cache miss/);
  await page.locator('#w-tracing .callbtns .btn').nth(2).click();
  log = await page.locator('#w-tracing .log').innerText();
  assert.match(log, /cache hit/);
  await page.locator('#w-tracing .stage-head .seg button', { hasText: 'Side effect' }).click();
  assert.deepEqual((await buttons()).slice(0, 2), ['f(ones(3))', 'f(ones(4))']);
  await page.locator('#w-tracing .callbtns .btn').nth(0).click();
  assert.match(await page.locator('#w-tracing .log').innerText(), /stdout: tracing: Traced/);
  await page.locator('#w-tracing .stage-head .seg button', { hasText: 'Python if' }).click();
  await page.locator('#w-tracing .callbtns .btn').nth(0).click();
  assert.match(await page.locator('#w-tracing .log').innerText(), /TracerBoolConversionError/);
  assert.deepEqual(errors, []);
  await context.close();
});

test('every composer preset resolves as its label promises', async () => {
  const { page, context, errors } = await open();
  const presets = page.locator('#w-transforms .presets .chip');
  const expected = [
    { label: /per-example gradients/, bad: false, sig: /\{w: f32\[B,…\], b: f32\[B,…\]\}/, torch: /torch\.func\.vmap\(torch\.func\.grad\(loss\)/ },
    { label: /compiled training gradient/, bad: false, sig: /compiled/, torch: /torch\.compile\(torch\.func\.grad\(loss\)\)/ },
    { label: /Hessian/, bad: false, sig: /\{w: \{w: f32/, expl: /Hessian/, torch: /torch\.func\.jacfwd\(torch\.func\.grad\(loss\)\)/ },
    { label: /the wrong order/, bad: true, expl: /vmap\(grad\(loss\)\)/, torch: /torch\.func\.grad\(torch\.func\.vmap\(loss/ },
  ];
  assert.equal(await presets.count(), expected.length);
  for (let i = 0; i < expected.length; i++) {
    const want = expected[i];
    await presets.nth(i).click();
    const label = await presets.nth(i).innerText();
    assert.match(label, want.label);
    const bad = await page.locator('#w-transforms .expl').evaluate((el) => el.classList.contains('bad'));
    assert.equal(bad, want.bad, `${label}: error state`);
    const sig = await page.locator('#w-transforms .sig').innerText();
    const expl = await page.locator('#w-transforms .expl').innerText();
    const torch = await page.locator('#w-transforms .eq pre').innerText();
    if (want.sig) assert.match(sig, want.sig, label);
    if (want.expl) assert.match(expl, want.expl, label);
    assert.match(torch, want.torch, label);
  }
  // grad twice is invalid on a pytree-valued gradient and the message must point at jacfwd
  await page.locator('#w-transforms .chips .btn', { hasText: 'clear' }).click();
  const gradChip = page.locator('#w-transforms .chips .chip', { hasText: 'grad(' });
  await gradChip.click(); await gradChip.click();
  assert.equal(await page.locator('#w-transforms .expl').evaluate((el) => el.classList.contains('bad')), true);
  assert.match(await page.locator('#w-transforms .expl').innerText(), /jacfwd\(grad\(loss\)\)/);
  assert.deepEqual(errors, []);
  await context.close();
});
