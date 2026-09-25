// Browser regression tests. Drives index.html in headless Chrome (the installed
// Google Chrome via Playwright's "chrome" channel, falling back to Playwright's
// own Chromium if that is unavailable). Run with `pnpm test`.
// The page has no script dependencies; web fonts are stubbed so the tests run offline.
import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { chromium } from 'playwright';

const here = path.dirname(fileURLToPath(import.meta.url));
const PAGE = pathToFileURL(path.join(here, '..', 'index.html')).href;
const MINUS = '−';
let browser;

before(async () => {
  try { browser = await chromium.launch({ channel: 'chrome', headless: true }); }
  catch { browser = await chromium.launch({ headless: true }); }
});
after(async () => { if (browser) await browser.close(); });

/** Open the page; collect page errors and console errors. */
async function open({ width = 1440 } = {}) {
  const context = await browser.newContext({ viewport: { width, height: 900 } });
  await context.route(/fonts\.(googleapis|gstatic)\.com/, (route) => route.fulfill({ status: 200, contentType: 'text/css', body: '' }));
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', (e) => errors.push('pageerror: ' + e.message));
  page.on('console', (m) => { if (m.type() === 'error') errors.push('console: ' + m.text()); });
  await page.goto(PAGE);
  await page.waitForSelector('#grpo-samples .g-row');
  return { page, context, errors };
}
const advs = (page) => page.$$eval('#grpo-samples .adv', (els) => els.map((e) => e.innerText.trim()));

for (const width of [1440, 390]) {
  test(`every figure renders, with no errors, no horizontal overflow, and legible chart text at ${width}px`, async () => {
    const { page, context, errors } = await open({ width });
    const widgets = await page.$$eval('[data-widget]', (els) => els.map((el) => [el.dataset.widget, el.querySelectorAll('svg, .g-row, .turn, .resp, tr').length]));
    assert.equal(widgets.length, 11);
    for (const [name, n] of widgets) assert.ok(n > 0, `${name} rendered nothing`);
    await page.evaluate(async () => { for (let y = 0; y < document.body.scrollHeight; y += 600) { window.scrollTo(0, y); await new Promise((r) => setTimeout(r, 10)); } });
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
    assert.equal(overflow, 0, 'page scrolls horizontally');
    // Charts are drawn at CSS-pixel width, so no label may render smaller than 11px.
    const small = await page.$$eval('svg text', (els) => els.filter((t) => t.textContent.trim() && t.getBoundingClientRect().width > 0)
      .map((t) => { const svg = t.ownerSVGElement; const s = svg.getBoundingClientRect().width / svg.viewBox.baseVal.width; return [t.textContent.slice(0, 24), parseFloat(getComputedStyle(t).fontSize) * s]; })
      .filter(([, px]) => px < 11));
    assert.deepEqual(small, []);
    assert.deepEqual(errors, []);
    await context.close();
  });
}

test('top bar: the section links fit at 1440px, and the current section is marked while scrolling', async () => {
  const { page, context, errors } = await open({ width: 1440 });
  const fits = await page.$eval('#nav', (n) => n.scrollWidth <= n.clientWidth + 1);
  assert.ok(fits, 'nav overflows its bar');
  await page.$eval('#grpo', (el) => el.scrollIntoView());
  await page.waitForTimeout(100);
  assert.equal(await page.$eval('#nav a[aria-current="true"]', (a) => a.getAttribute('href')), '#grpo');
  assert.deepEqual(errors, []);
  await context.close();
});

test('GRPO: a group with no variance yields zero advantage everywhere, and re-toggling restores signal', async () => {
  const { page, context, errors } = await open();
  await page.uncheck('#grpo-format'); // the format bonus alone would keep the group from being zero-variance
  const wrong = page.locator('#grpo-samples .tog.bad');
  while (await wrong.count()) await wrong.first().click(); // make every rollout correct
  assert.equal(await page.locator('#grpo-signal').innerText(), 'none');
  const a = await advs(page);
  assert.deepEqual(a, a.map(() => '+0.00'));
  await page.locator('#grpo-samples .tog').first().click();
  assert.notEqual(await page.locator('#grpo-signal').innerText(), 'none');
  assert.ok((await advs(page))[0].startsWith(MINUS), 'the lone wrong rollout gets a negative advantage');
  assert.equal((await advs(page))[0], `${MINUS}2.24`, 'one wrong of six: −√5');
  assert.deepEqual(errors, []);
  await context.close();
});

test('GRPO: three correct of six with no format bonus gives advantages of exactly ±1 (population std)', async () => {
  const { page, context, errors } = await open();
  await page.uncheck('#grpo-format');
  const togs = page.locator('#grpo-samples .tog');
  for (let i = 0; i < 6; i++) { // want rollouts 0..2 correct, 3..5 wrong
    const isOk = (await togs.nth(i).getAttribute('class')).includes('ok');
    if (isOk !== i < 3) await togs.nth(i).click();
  }
  assert.deepEqual(await advs(page), ['+1.00', '+1.00', '+1.00', `${MINUS}1.00`, `${MINUS}1.00`, `${MINUS}1.00`]);
  assert.equal(await page.locator('#grpo-std').innerText(), '0.50');
  assert.deepEqual(errors, []);
  await context.close();
});

test('GRPO: every verdict matches its text, and every reward matches the verdict and format rule', async () => {
  const { page, context, errors } = await open();
  const check = async (formatOn) => {
    const rows = await page.$$eval('#grpo-samples .g-row', (els) => els.map((r) => ({ text: r.querySelector('.g-txt').textContent, ok: r.querySelector('.tog').classList.contains('ok'), r: +r.querySelector('.g-r').textContent })));
    for (const { text, ok, r } of rows) {
      const final = (text.match(/\\boxed\{(\d+)\}/) || text.match(/(\d+)\s*$/))[1];
      assert.equal(ok, final === '408', `verdict for "${text}"`);
      const want = (ok ? 1 : 0) + (formatOn && text.includes('\\boxed') ? 0.2 : 0);
      assert.equal(r, Math.round(want * 10) / 10, `reward for "${text}"`);
    }
  };
  await check(true);
  for (let i = 0; i < 6; i++) await page.locator('#grpo-samples .tog').nth(i).click(); // flip every rollout
  await check(true);
  await page.uncheck('#grpo-format');
  await check(false);
  await page.click('#grpo-reset');
  assert.ok(await page.isChecked('#grpo-format'), 'reset restores the format reward');
  assert.deepEqual(await advs(page), ['+0.70', '+0.70', `${MINUS}1.60`, '+0.70', `${MINUS}1.21`, '+0.70']);
  assert.deepEqual(errors, []);
  await context.close();
});

test('GRPO: verdict buttons are reachable and operable from the keyboard', async () => {
  const { page, context, errors } = await open();
  await page.focus('#grpo-reset');
  await page.keyboard.press('Tab');
  await page.keyboard.press('Enter');
  assert.ok((await page.locator('#grpo-samples .tog').first().getAttribute('class')).includes('bad'), 'Enter flipped the first verdict');
  assert.equal(await page.evaluate(() => document.activeElement.dataset.i), '0', 'focus stays on the flipped button');
  assert.deepEqual(errors, []);
  await context.close();
});

test('policy gradient: a positive-advantage update raises the sampled answer, and repeating it moves the policy less', async () => {
  const { page, context, errors } = await open();
  const probs = () => page.$$eval('#pg-viz .val', (els) => els.map((e) => +e.dataset.p));
  await page.selectOption('#pg-action', '2');
  await page.click('#pg-step');
  const p1 = await probs();
  assert.ok(p1[2] > 0.3 && p1[0] < 0.2 && p1[4] < 0.2, `unexpected distribution ${p1}`);
  assert.ok(Math.abs(p1.reduce((a, b) => a + b, 0) - 1) < 1e-3, 'probabilities sum to 1');
  assert.equal(await page.locator('#pg-n').innerText(), '1');
  assert.equal(await page.locator('#pg-adv').innerText(), '+0.50', 'baseline moved halfway to the reward of 1');
  await page.click('#pg-step');
  const p2 = await probs();
  assert.ok(p2[2] - p1[2] < p1[2] - 0.2, 'second update is smaller than the first');
  await page.click('#pg-reset');
  assert.equal(await page.locator('#pg-n').innerText(), '0');
  assert.equal(await page.locator('#pg-adv').innerText(), '+1.00');
  assert.deepEqual(errors, []);
  await context.close();
});

test('trajectory: flipping the outcome flips the sign of every policy-turn advantage; observations stay masked', async () => {
  const { page, context, errors } = await open();
  const creds = () => page.$$eval('#traj .turn.policy .cred', (els) => els.map((e) => e.innerText.trim()));
  assert.ok((await creds()).every((c) => c === 'A = +0.55'));
  await page.locator('#traj-outcome button', { hasText: 'fail' }).click();
  assert.equal(await page.locator('#traj-outcome button[aria-pressed="true"]').innerText(), 'Hidden tests fail');
  assert.ok((await creds()).every((c) => c === `A = ${MINUS}0.45`));
  assert.equal(await page.locator('#traj .turn.env .cred', { hasText: 'masked' }).count(), 3, 'the three tool outputs are masked; the grader turn shows R');
  assert.equal(await page.locator('#traj .turn.env .cred', { hasText: 'R = 0' }).count(), 1);
  await page.locator('#traj-credit button', { hasText: 'shaping' }).click();
  assert.match(await page.locator('#traj-adv').innerText(), /\+0\.10 per valid tool call/);
  assert.ok((await creds()).includes(`A = ${MINUS}0.35`), 'shaping adds 0.10 to tool-calling turns');
  assert.deepEqual(errors, []);
  await context.close();
});

test('reward model: preferring A moves P(A ≻ B) up by less each time, and reset returns it to 0.50', async () => {
  const { page, context, errors } = await open();
  const P = async () => +(await page.locator('#rlhf-p').textContent());
  const preferA = () => page.locator('#rlhf-pair button', { hasText: 'Prefer A' }).click();
  await preferA(); const p1 = await P();
  await preferA(); const p2 = await P();
  assert.ok(p2 > 0.6);
  assert.ok(p2 - p1 < p1 - 0.5, `steps should shrink: ${p1}, ${p2}`);
  assert.equal(await page.locator('#rlhf-count').innerText(), '2 comparisons');
  assert.equal(await page.locator('#rlhf-curve circle').count(), 3, 'two earlier positions and the current one');
  await page.click('#rlhf-next');
  assert.equal(await P(), 0.5, 'the next pair starts untrained');
  await page.click('#rlhf-reset');
  assert.equal((await page.locator('#rlhf-p').textContent()), '0.50');
  assert.deepEqual(errors, []);
  await context.close();
});

test('KL figure: readouts match the exact solution, and β controls how far the policy moves', async () => {
  const { page, context, errors } = await open();
  const state = () => page.evaluate(() => ({ beta: +document.querySelector('#kl-beta-v').textContent, kl: +document.querySelector('#kl-val').textContent, bars: [...document.querySelectorAll('#kl-viz .pb')].map((b) => +b.dataset.p) }));
  let s = await state();
  assert.equal(s.beta, 0.5);
  const exact = await page.evaluate((b) => window.__klSolve(b), s.beta);
  assert.equal(s.kl, +exact.kl.toFixed(3));
  s.bars.forEach((p, i) => assert.ok(Math.abs(p - exact.pol[i]) < 1e-4));
  await page.$eval('#kl-beta', (el) => { el.value = 100; el.dispatchEvent(new Event('input')); });
  s = await state();
  assert.equal(s.beta, 5);
  assert.ok(s.kl < 0.03, `large β keeps the policy near the reference (KL ${s.kl})`);
  await page.$eval('#kl-beta', (el) => { el.value = 0; el.dispatchEvent(new Event('input')); });
  s = await state();
  assert.equal(s.beta, 0.05);
  assert.ok(s.bars[7] > 0.99, 'small β piles the policy onto the top-rewarded token');
  await page.focus('#kl-beta');
  for (let i = 0; i < 10; i++) await page.keyboard.press('ArrowRight');
  assert.equal((await state()).beta, 0.08, 'the slider responds to the keyboard (10 steps of 0.02 decades)');
  assert.deepEqual(errors, []);
  await context.close();
});

test('published methods: every organization links a primary source, and no source from the previous version was dropped', async () => {
  const { page, context, errors } = await open();
  const groups = await page.$$eval('#labs-table tbody', (tbs) => tbs.map((tb) => [tb.querySelector('tr.grp th').textContent, tb.querySelectorAll('a[href^="https://"]').length]));
  assert.equal(groups.length, 9);
  for (const [name, n] of groups) assert.ok(n >= 1, `${name} has no source`);
  const hrefs = new Set(await page.$$eval('a[href^="https://"]', (as) => as.map((a) => a.href)));
  const kept = [
    'https://ai.meta.com/blog/llama-4-multimodal-intelligence/', 'https://arxiv.org/abs/1707.06347', 'https://arxiv.org/abs/2203.02155',
    'https://arxiv.org/abs/2212.08073', 'https://arxiv.org/abs/2305.18290', 'https://arxiv.org/abs/2402.03300', 'https://arxiv.org/abs/2407.21783',
    'https://arxiv.org/abs/2501.12599', 'https://arxiv.org/abs/2501.12948', 'https://arxiv.org/abs/2503.14476', 'https://arxiv.org/abs/2505.24298',
    'https://arxiv.org/abs/2506.13585', 'https://arxiv.org/abs/2507.18071', 'https://arxiv.org/abs/2507.20534', 'https://blog.google/products/gemini/gemini-3/',
    'https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/',
    'https://deepmind.google/discover/blog/advanced-version-of-gemini-with-deep-think-officially-achieves-gold-medal-standard-at-the-international-mathematical-olympiad/',
    'https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/',
    'https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/',
    'https://github.com/volcengine/verl', 'https://mistral.ai/news/magistral', 'https://openai.com/index/chain-of-thought-monitoring/',
    'https://openai.com/index/healthbench/', 'https://openai.com/index/introducing-codex/', 'https://openai.com/index/introducing-deep-research/',
    'https://openai.com/index/introducing-gpt-5/', 'https://openai.com/index/introducing-o3-and-o4-mini/', 'https://openai.com/index/learning-to-reason-with-llms/',
    'https://poloclub.github.io/', 'https://poloclub.github.io/transformer-explainer/', 'https://qwenlm.github.io/blog/qwen3-coder/', 'https://qwenlm.github.io/blog/qwen3/',
    'https://rlhfbook.com/', 'https://www.anthropic.com/news/claude-3-7-sonnet', 'https://www.anthropic.com/news/claude-4',
    'https://www.anthropic.com/news/claude-sonnet-4-5', 'https://www.anthropic.com/research/emergent-misalignment-reward-hacking',
    'https://www.nature.com/articles/s41586-025-09422-z', 'https://x.ai/news/grok-4',
  ];
  const missing = kept.filter((u) => !hrefs.has(u));
  assert.deepEqual(missing, []);
  assert.deepEqual(errors, []);
  await context.close();
});

test('the page is self-contained: no external scripts, and every in-page link resolves', async () => {
  const { page, context, errors } = await open();
  assert.equal(await page.locator('script[src]').count(), 0);
  const broken = await page.$$eval('a[href^="#"]', (as) => as.map((a) => a.getAttribute('href')).filter((h) => !document.querySelector(h)));
  assert.deepEqual(broken, []);
  assert.deepEqual(errors, []);
  await context.close();
});
