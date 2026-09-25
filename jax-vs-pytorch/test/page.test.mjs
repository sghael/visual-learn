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
const WIDGETS = ['tracing', 'fusion', 'autodiff', 'vmap', 'composer', 'purity', 'systolic', 'gpu', 'sharding'];
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
  await page.waitForSelector('#w-sharding svg');
  return { page, context, errors };
}
const readout = (page, id) => page.locator(`#${id} .stepper .readout`).innerText();
const stepBtn = (page, id, role) => page.locator(`#${id} .stepper [data-role="${role}"]`);

for (const width of [1440, 390]) {
  test(`page renders every figure legibly, without errors or horizontal overflow, at ${width}px`, async () => {
    const { page, context, errors } = await open({ width });
    const widgets = await page.$$eval('[data-widget]', (els) => els.map((el) => [el.dataset.widget, el.children.length]));
    assert.deepEqual(widgets.map((w) => w[0]), WIDGETS);
    for (const [name, children] of widgets) assert.ok(children > 0, `${name} rendered nothing`);
    await page.evaluate(async () => { for (let y = 0; y < document.body.scrollHeight; y += 600) { window.scrollTo(0, y); await new Promise((r) => setTimeout(r, 20)); } });
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
    assert.equal(overflow, 0, 'page scrolls horizontally');
    // chart text must render at 11 CSS px or more
    const small = await page.evaluate(() => Array.from(document.querySelectorAll('svg text')).filter((t) => t.textContent.trim() && t.getBoundingClientRect().width).map((t) => {
      const svg = t.ownerSVGElement, vb = svg.viewBox.baseVal;
      const scale = vb && vb.width ? svg.getBoundingClientRect().width / vb.width : 1;
      return [t.textContent, parseFloat(getComputedStyle(t).fontSize) * scale];
    }).filter(([, px]) => px < 10.95));
    assert.deepEqual(small, [], 'svg text smaller than 11px');
    assert.deepEqual(errors, []);
    await context.close();
  });
}

test('nothing moves on load or on scroll: steppers rest where they were put', async () => {
  const { page, context, errors } = await open({ reduced: false });
  const before = { t: await readout(page, 'w-tracing'), a: await readout(page, 'w-autodiff'), s: await readout(page, 'w-systolic') };
  assert.deepEqual(before, { t: 'step 5 of 5', a: 'step 9 of 9', s: 'cycle 4 of 10' });
  for (const id of ['w-tracing', 'w-autodiff', 'w-systolic']) await page.locator('#' + id).scrollIntoViewIfNeeded();
  await page.waitForTimeout(2000);
  assert.deepEqual({ t: await readout(page, 'w-tracing'), a: await readout(page, 'w-autodiff'), s: await readout(page, 'w-systolic') }, before);
  assert.deepEqual(errors, []);
  await context.close();
});

test('switching tracing examples during playback neither throws nor keeps the old stepper alive', async () => {
  const { page, context, errors } = await open({ reduced: false });
  await stepBtn(page, 'w-tracing', 'play').click(); // restarts from the top
  await page.waitForFunction(() => /step [1-4] of 5/.test(document.querySelector('#w-tracing .stepper .readout').textContent), null, { timeout: 5000 });
  await page.locator('#w-tracing .seg button', { hasText: 'Python if' }).click();
  await page.waitForTimeout(3500); // longer than two ticks of the old stepper
  assert.equal(await readout(page, 'w-tracing'), 'step 3 of 3');
  assert.match(await page.locator('#w-tracing .note').innerText(), /TracerBoolConversionError/);
  assert.equal(await stepBtn(page, 'w-tracing', 'play').innerText(), 'Replay');
  assert.deepEqual(errors, []);
  await context.close();
});

test('a stepper reset mid-run cannot be resumed by its old timer (fake clock)', async () => {
  // Fake time makes the race deterministic: park a run between ticks, reset, then
  // let far more than one interval pass. Nothing may advance.
  const { page, context, errors } = await open({ reduced: false, fakeClock: true });
  await stepBtn(page, 'w-autodiff', 'play').click(); // from the rest state this replays from step 1
  assert.equal(await readout(page, 'w-autodiff'), 'step 1 of 9');
  await page.clock.runFor(1400 * 2 + 100);            // two more ticks
  assert.equal(await readout(page, 'w-autodiff'), 'step 3 of 9');
  await stepBtn(page, 'w-autodiff', 'reset').click();
  assert.equal(await readout(page, 'w-autodiff'), 'step 0 of 9');
  await page.clock.runFor(20000);
  assert.equal(await readout(page, 'w-autodiff'), 'step 0 of 9');
  assert.match(await page.locator('#w-autodiff .result').first().innerText(), /w\.grad = None/);
  // Play again runs to the end and stops there.
  await stepBtn(page, 'w-autodiff', 'play').click();
  await page.clock.runFor(1400 * 12);
  assert.equal(await readout(page, 'w-autodiff'), 'step 9 of 9');
  assert.equal(await stepBtn(page, 'w-autodiff', 'play').innerText(), 'Replay');
  // Pausing the systolic array mid-run also freezes it.
  await stepBtn(page, 'w-systolic', 'play').click();
  await page.clock.runFor(900);
  await stepBtn(page, 'w-systolic', 'play').click(); // Pause
  const frozen = await readout(page, 'w-systolic');
  await page.clock.runFor(10000);
  assert.equal(await readout(page, 'w-systolic'), frozen);
  assert.deepEqual(errors, []);
  await context.close();
});

test('tracing call buttons and log use each example\'s real arity', async () => {
  const { page, context, errors } = await open();
  const buttons = () => page.locator('#w-tracing .grid2 .controls .btn').allInnerTexts();
  const log = () => page.locator('#w-tracing .log').innerText();
  assert.deepEqual(await buttons(), ['f(ones(3), ones(3))', 'f(ones(4), ones(4))', 'Clear log']);
  // resting log: miss, hit, miss
  assert.match(await log(), /cache miss[\s\S]*cache hit[\s\S]*cache miss/);
  await page.locator('#w-tracing .btn', { hasText: 'Clear log' }).click();
  assert.match(await log(), /No calls yet/);
  await page.locator('#w-tracing .grid2 .controls .btn').nth(0).click();
  assert.match(await log(), /f\(ones\(3\), ones\(3\)\), signature \(f32\[3\], f32\[3\]\): cache miss/);
  await page.locator('#w-tracing .grid2 .controls .btn').nth(0).click();
  assert.match(await log(), /cache hit/);
  assert.match(await page.locator('#w-tracing .cache').innerText(), /\(f32\[3\], f32\[3\]\)/);
  await page.locator('#w-tracing .seg button', { hasText: 'Side effect' }).click();
  assert.deepEqual((await buttons()).slice(0, 2), ['f(ones(3))', 'f(ones(4))']);
  assert.match(await log(), /printed: tracing: JitTracer\(float32\[3\]\)/);
  await page.locator('#w-tracing .seg button', { hasText: 'Python if' }).click();
  assert.match(await log(), /TracerBoolConversionError/);
  assert.match(await page.locator('#w-tracing .cache').innerText(), /none/);
  await page.locator('#w-tracing .seg button', { hasText: 'with lax.cond' }).click();
  assert.match(await log(), /cache hit/);
  assert.match(await page.locator('#w-tracing .grid2').innerText(), /cond\[/);
  assert.deepEqual(errors, []);
  await context.close();
});

test('fusion readouts match the model, and toggling an op updates both multiples', async () => {
  const { page, context, errors } = await open();
  const lines = () => page.locator('#w-fusion .readline').allInnerTexts();
  let [eager, compiled] = await lines();
  assert.match(eager, /^5 kernel launches · 4 intermediates written to HBM · 704 MB of HBM traffic · at least 0\.22 ms/);
  assert.match(compiled, /^2 kernel launches · 1 intermediate written to HBM · 320 MB of HBM traffic · at least 0\.10 ms/);
  await page.locator('#w-fusion button[data-op="softmax"]').click();
  [eager, compiled] = await lines();
  assert.match(eager, /^4 kernel launches · 3 intermediates/);
  assert.match(compiled, /^1 kernel launch · 0 intermediates written to HBM · 192 MB of HBM traffic/);
  // the last enabled op cannot be switched off
  for (const op of ['matmul', 'bias', 'relu']) await page.locator(`#w-fusion button[data-op="${op}"]`).click();
  assert.equal(await page.locator('#w-fusion button[data-op="scale"]').isDisabled(), true);
  assert.equal(await page.locator('#w-fusion [data-mode="compiled"] svg').count(), 1);
  assert.deepEqual(errors, []);
  await context.close();
});

test('autodiff: both frameworks report the analytic gradient, and w changes numbers but not equations', async () => {
  const { page, context, errors } = await open();
  const expected = (w) => { const a = 1.5 * w; return (2 * (Math.sin(a) - 0.5) * Math.cos(a) * 1.5).toFixed(3); };
  const results = () => page.locator('#w-autodiff .result').allInnerTexts();
  let [torch, jax] = await results();
  assert.equal(torch, `w.grad = tensor(${expected(0.8)})`);
  assert.equal(jax, `dL(0.800) = ${expected(0.8)}`);
  const jaxpr = async () => (await page.locator('#w-autodiff .cols > div').nth(1).locator('pre').nth(1).innerText());
  const eqs = await jaxpr();
  assert.match(eqs, /d:f32\[\] = cos b/);
  assert.match(eqs, /i:f32\[\] = mul h 1\.5/);
  await page.locator('#w-autodiff input[type="range"]').fill('-1.2');
  [torch, jax] = await results();
  assert.equal(torch, `w.grad = tensor(${expected(-1.2)})`);
  assert.equal(jax, `dL(-1.200) = ${expected(-1.2)}`);
  assert.equal(await jaxpr(), eqs, 'the gradient jaxpr must not depend on w');
  assert.deepEqual(errors, []);
  await context.close();
});

test('every composer preset resolves as its label promises', async () => {
  const { page, context, errors } = await open();
  const presets = page.locator('#w-composer .presets button');
  const expected = [
    { label: /per-example gradients/, bad: false, sig: /\{w: f32\[B,…\], b: f32\[B,…\]\}/, torch: /torch\.func\.vmap\(torch\.func\.grad\(loss\)/ },
    { label: /compiled gradient/, bad: false, sig: /compiled/, torch: /torch\.compile\(torch\.func\.grad\(loss\)\)/ },
    { label: /Hessian/, bad: false, sig: /\{w: \{w: f32/, expl: /Hessian/, torch: /torch\.func\.jacfwd\(torch\.func\.grad\(loss\)\)/ },
    { label: /the wrong order/, bad: true, expl: /vmap\(grad\(loss\)\)/, torch: /torch\.func\.grad\(torch\.func\.vmap\(loss/ },
  ];
  assert.equal(await presets.count(), expected.length);
  assert.equal(await presets.nth(0).getAttribute('aria-pressed'), 'true', 'rests on per-example gradients');
  for (let i = 0; i < expected.length; i++) {
    const want = expected[i];
    await presets.nth(i).click();
    const label = await presets.nth(i).innerText();
    assert.match(label, want.label);
    assert.equal(await presets.nth(i).getAttribute('aria-pressed'), 'true');
    const bad = await page.locator('#w-composer .expl').evaluate((el) => el.classList.contains('bad'));
    assert.equal(bad, want.bad, `${label}: error state`);
    const sig = await page.locator('#w-composer .sig').innerText();
    const expl = await page.locator('#w-composer .expl').innerText();
    const torch = await page.locator('#w-composer .eq pre').innerText();
    if (want.sig) assert.match(sig, want.sig, label);
    if (want.expl) assert.match(expl, want.expl, label);
    assert.match(torch, want.torch, label);
  }
  // grad twice is invalid on a pytree-valued gradient and the message must point at jacfwd
  await page.locator('#w-composer [data-role="clear"]').click();
  const gradBtn = page.locator('#w-composer [data-wrap="grad"]');
  await gradBtn.click(); await gradBtn.click();
  assert.equal(await page.locator('#w-composer .expl').evaluate((el) => el.classList.contains('bad')), true);
  assert.match(await page.locator('#w-composer .expl').innerText(), /jacfwd\(grad\(loss\)\)/);
  assert.deepEqual(errors, []);
  await context.close();
});

test('vmap: the batch-axis control changes the jaxpr dimension numbers', async () => {
  const { page, context, errors } = await open();
  const vjaxpr = () => page.locator('#w-vmap [data-mode="vmap"] pre').nth(1).innerText();
  assert.match(await vjaxpr(), /b:f32\[5,3\][\s\S]*\(\(\[0\], \[1\]\), \(\[\], \[\]\)\)/);
  await page.locator('#w-vmap .seg button', { hasText: 'axis 1' }).click();
  assert.match(await vjaxpr(), /b:f32\[3,5\][\s\S]*\(\(\[0\], \[0\]\), \(\[\], \[\]\)\)/);
  assert.match(await page.locator('#w-vmap [data-mode="loop"] pre').first().innerText(), /X\[:, i\]/);
  assert.deepEqual(errors, []);
  await context.close();
});

test('state: picking a concept highlights lines in both listings', async () => {
  const { page, context, errors } = await open();
  const hl = (n) => page.locator('#w-purity pre').nth(n).locator('.ln.hl').count();
  assert.equal(await hl(0), 7); // parameters, at rest
  await page.locator('#w-purity button', { hasText: 'gradients' }).click();
  assert.equal(await hl(0), 1);
  assert.equal(await hl(1), 2);
  assert.match(await page.locator('#w-purity .expl').nth(1).innerText(), /zero_grad/);
  assert.deepEqual(errors, []);
  await context.close();
});

test('sharding: presets set the selects, and the collectives match the rule engine', async () => {
  const { page, context, errors } = await open();
  const coll = () => page.locator('#w-sharding .coll').innerText();
  assert.match(await coll(), /None/);
  await page.locator('#w-sharding button[data-preset="rowtp"]').click();
  assert.equal(await page.locator('#w-sharding select').nth(0).inputValue(), '-,model');
  assert.equal(await page.locator('#w-sharding select').nth(1).inputValue(), 'model,-');
  assert.match(await coll(), /all-reduce over 'model'/);
  assert.match(await page.locator('#w-sharding .torch-eq').innerText(), /row-parallel/);
  await page.locator('#w-sharding select').nth(1).selectOption('data,-');
  assert.match(await coll(), /all-gather w over 'data'/);
  assert.equal(await page.locator('#w-sharding button[aria-pressed="true"]').count(), 0, 'custom layout: no preset pressed');
  assert.match(await page.locator('#w-sharding .torch-eq').innerText(), /custom DTensor/);
  assert.deepEqual(errors, []);
  await context.close();
});

test('the top bar marks the section being read', async () => {
  const { page, context, errors } = await open();
  await page.locator('#fusion h2').scrollIntoViewIfNeeded();
  await page.evaluate(() => window.scrollTo(0, document.getElementById('fusion').offsetTop + 200));
  await page.waitForFunction(() => document.querySelector('.topbar nav a[aria-current="true"]')?.getAttribute('href') === '#fusion');
  assert.equal(await page.locator('.topbar nav a[aria-current="true"]').count(), 1);
  assert.deepEqual(errors, []);
  await context.close();
});
