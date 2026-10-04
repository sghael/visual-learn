import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { chromium } from 'playwright';
import { cp, mkdtemp, readFile, rm, stat } from 'node:fs/promises';
import { createServer } from 'node:http';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import {
  ROOT, REPO, manifest, open, assertLayout, assertLocalLinks, capturePage,
  near, number, readout, click, slide, withWidget,
} from './browser-helpers.mjs';

let browser;
before(async () => {
  try { browser = await chromium.launch({ channel: 'chrome', headless: true }); }
  catch { browser = await chromium.launch({ headless: true }); }
});
after(async () => { await browser?.close(); });

for (const width of [1440, 390]) {
  for (const entry of [{ slug: '', model: 'Collection', figures: [] }, ...manifest]) {
    test(`${entry.model}: offline layout, labels and links at ${width}px`, async () => {
      const session = await open(browser, { slug: entry.slug, width });
      try {
        assert.deepEqual(await session.page.locator('[data-widget]').evaluateAll(nodes => nodes.map(node => node.dataset.widget)), entry.figures);
        await assertLayout(session.page);
        if (width === 1440) {
          const proseWidth = await session.page.locator('main section > p').first().evaluate(node => node.getBoundingClientRect().width);
          assert.ok(proseWidth >= 800, `${entry.model}: desktop prose should use the wider book column`);
        }
        if (width === 1440) await assertLocalLinks(session.page);
        await capturePage(session.page, entry.slug || 'collection', width);
        assert.deepEqual(session.errors, []);
      } finally { await session.context.close(); }
    });
  }
}

test('the manifest covers every lesson directory and every interactive kind has behavioral tests', async () => {
  const { readdir } = await import('node:fs/promises');
  const folders = (await readdir(ROOT, { withFileTypes: true })).filter(item => item.isDirectory() && /^\d{4}-\d{2}-\d{2}-/.test(item.name)).map(item => item.name).sort();
  assert.deepEqual(manifest.map(item => item.slug).sort(), folders, 'regenerate the manifest before testing');
  const covered = new Set(['flow', 'bars', 'kv', 'moe', 'attention', 'sparse', 'grpo', 'precision', 'scaling', 'latent', 'distill', 'parallel', 'state', 'sink', 'verifier', 'mixture', 'cache', 'patches', 'diffusion', 'ternary', 'recurrence']);
  assert.deepEqual([...new Set(manifest.flatMap(item => item.figures))].filter(kind => !covered.has(kind)), [], 'new interactive kinds need behavioral tests');
});

test('collection search, mechanism and order compose; clearing restores all lessons and focus', async () => {
  const { page, context, errors } = await open(browser, { width: 390 });
  try {
    const cards = page.locator('.catalog > li');
    const initial = await cards.evaluateAll(nodes => nodes.map(node => ({ text: node.textContent.toLowerCase(), category: node.dataset.category, href: node.querySelector('a').getAttribute('href') })));
    const visibleLinks = () => page.locator('.catalog > li:visible h3 a').evaluateAll(nodes => nodes.map(node => node.getAttribute('href')));
    const expected = initial.filter(item => item.text.includes('qwen') && item.category === 'Memory').map(item => item.href);
    assert.ok(expected.length >= 2, 'fixture should exercise ordering within a combined filter');
    await page.getByLabel('Find a model or idea').fill('  QwEn  ');
    await page.getByLabel('Mechanism', { exact: true }).selectOption('Memory');
    assert.deepEqual(await visibleLinks(), expected);
    await page.getByLabel('Order', { exact: true }).selectOption('newest');
    assert.deepEqual(await visibleLinks(), [...expected].reverse());
    assert.equal(await page.locator('#match-count').innerText(), `${expected.length} of ${manifest.length} tutorials`);
    await page.getByLabel('Find a model or idea').fill('no-model-has-this-identifier-4931');
    assert.equal(await page.locator('.catalog > li:visible').count(), 0);
    assert.equal(await page.locator('#empty').isVisible(), true);
    await page.getByRole('button', { name: 'Clear filters' }).click();
    assert.deepEqual(await visibleLinks(), initial.map(item => item.href));
    assert.equal(await page.locator('#empty').isVisible(), false);
    assert.equal(await page.locator('#category').inputValue(), 'all');
    assert.equal(await page.locator('#order').inputValue(), 'oldest');
    assert.equal(await page.locator('#search').evaluate(node => node === document.activeElement), true);
    await assertLayout(page);
    assert.deepEqual(errors, []);
  } finally { await context.close(); }
});

test('KV context and MHA/GQA/MQA presets preserve the advertised memory ratios', async () => {
  await withWidget(browser, 'kv', async (widget, config) => {
    const memory = async () => number((await readout(widget)).match(/= ([\d,.]+) GiB/)[1]);
    await slide(widget, 65536);
    await click(widget, 'MQA'); const one = await memory();
    await click(widget, 'GQA'); near(await memory(), one * config.kvHeads, 0.001 * config.kvHeads);
    await click(widget, 'MHA'); const all = await memory(); near(all, one * config.heads, 0.001 * config.heads);
    await slide(widget, 131072); near(await memory(), all * 2, 0.002);
    await click(widget, 'MQA'); near(await memory(), one * 2, 0.002);
    assert.match(await readout(widget), /KV cache only, one sequence in bf16/);
  });
});

test('MoE token examples reroute the selected experts without changing their count', async () => {
  await withWidget(browser, 'moe', async (widget, config) => {
    assert.equal(Number(await widget.locator('input').inputValue()), config.active);
    assert.equal(await widget.locator('svg text.selected').count(), config.active);
    if (config.shared) assert.match(await widget.locator('svg').textContent(), /shared expert.*always active/);
    await slide(widget, 2);
    const selected = () => widget.locator('svg text.selected').evaluateAll(nodes => nodes.map(node => `${node.getAttribute('x')},${node.getAttribute('y')}`));
    await click(widget, 'Token A'); const first = await selected(); assert.equal(first.length, 2);
    await click(widget, 'Token B'); assert.notDeepEqual(await selected(), first); assert.equal((await selected()).length, 2);
    await slide(widget, 4); assert.equal((await selected()).length, 4);
    await click(widget, 'Token C'); assert.equal((await selected()).length, 4);
    await click(widget, 'Token A'); await slide(widget, 2); assert.deepEqual(await selected(), first);
    assert.match(await readout(widget), /not a speedup/);
  });
});

test('causal attention excludes the future and caps a local query at the window size', async () => {
  await withWidget(browser, 'attention', async (widget, config) => {
    const count = async () => number((await readout(widget)).match(/can read (\d+)/)[1]);
    await slide(widget, 1); await click(widget, 'Full'); assert.equal(await count(), 1);
    await click(widget, 'Local'); assert.equal(await count(), 1);
    await slide(widget, config.tokens); assert.equal(await count(), Math.min(config.window, config.tokens));
    await click(widget, 'Full'); assert.equal(await count(), config.tokens);
    await click(widget, 'Toy hybrid'); assert.ok(await count() >= Math.min(config.window, config.tokens)); assert.ok(await count() <= config.tokens);
    await slide(widget, 1); assert.equal(await count(), 1);
  });
});

test('sparse attention changes the selected set and obeys its key budget', async () => {
  await withWidget(browser, 'sparse', async (widget, config) => {
    const selected = () => widget.locator('svg text').evaluateAll(nodes => nodes.filter(node => /· selected$/.test(node.textContent)).map(node => node.getAttribute('y')));
    await slide(widget, 1); assert.equal((await selected()).length, 1);
    await click(widget, 'Query A'); const first = await selected();
    await click(widget, 'Query B');
    if (!config.scores || new Set(config.scores).size > 1) assert.notDeepEqual(await selected(), first);
    await slide(widget, config.tokens || 16); assert.equal((await selected()).length, config.tokens || 16);
    await click(widget, 'Query C'); assert.equal((await selected()).length, config.tokens || 16);
    await slide(widget, 1); await click(widget, 'Query A'); assert.deepEqual(await selected(), first);
  });
});

test('group-relative rewards remain finite for all-equal groups and preserve signed contrast', async () => {
  await withWidget(browser, 'grpo', async widget => {
    const advantages = () => widget.locator('svg text[text-anchor="end"]').allTextContents().then(values => values.map(number));
    await click(widget, 'All equal'); assert.deepEqual(await advantages(), [0, 0, 0, 0]);
    assert.match(await readout(widget), /no within-group learning signal/);
    assert.doesNotMatch(await widget.innerText(), /NaN|Infinity/);
    await click(widget, 'One succeeds'); const values = await advantages();
    assert.ok(values.slice(0, 3).every(value => value < 0)); assert.ok(values[3] > 0);
    near(values.reduce((a, b) => a + b, 0), 0, 0.003);
    near(values[3], -3 * values[0], 0.003);
    await click(widget, 'Mixed results'); assert.ok((await advantages()).every(Number.isFinite));
  });
});

test('precision changes storage in proportion to bit width, without suggesting a full runtime footprint', async () => {
  await withWidget(browser, 'precision', async widget => {
    const size = async () => number((await readout(widget)).match(/= ([\d,.]+) decimal GB/)[1]);
    await click(widget, '4 bits'); const four = await size();
    await click(widget, '8 bits'); near(await size(), four * 2, 0.02);
    await click(widget, '16 bits'); near(await size(), four * 4, 0.04);
    await click(widget, '4 bits'); near(await size(), four);
    assert.match(await readout(widget), /no scales, KV cache, activations or runtime/);
  });
});

test('fixed training compute trades model size for training tokens', async () => {
  await withWidget(browser, 'scaling', async (widget, config) => {
    const tokens = async () => number((await readout(widget)).match(/parameters permit ([\d,.]+)T training tokens/)[1]);
    const compute = async () => (await readout(widget)).match(/^6ND = .*?approximate training FLOPs\./)[0];
    await slide(widget, config.baselineParams); const baseline = await tokens(); const budget = await compute();
    near(baseline, config.baselineTokens, 0.001);
    await slide(widget, config.baselineParams * 2); near(await tokens(), baseline / 2, 0.001); assert.equal(await compute(), budget);
    await slide(widget, config.baselineParams / 2); near(await tokens(), baseline * 2, 0.001); assert.equal(await compute(), budget);
    await slide(widget, config.baselineParams); near(await tokens(), baseline, 0.001);
  });
});

test('latent width reaches the original scalar count at its uncompressed endpoint', async () => {
  await withWidget(browser, 'latent', async (widget, config) => {
    await slide(widget, config.original); assert.match(await readout(widget), /100% as many cached scalars/);
    await slide(widget, config.latent); assert.match(await readout(widget), new RegExp(`^${config.latent} / ${config.original}`));
    assert.match(await readout(widget), /reconstruction costs are excluded/);
  });
});

test('distillation temperature changes concentration while probabilities remain normalized', async () => {
  await withWidget(browser, 'distill', async widget => {
    const probabilities = () => widget.locator('svg text[text-anchor="end"]').allTextContents().then(values => values.map(value => number(value.replace('%', '').trim())));
    await slide(widget, 0.2); const cold = await probabilities(); near(cold.reduce((a, b) => a + b, 0), 100, 0.03);
    await slide(widget, 4); const warm = await probabilities(); near(warm.reduce((a, b) => a + b, 0), 100, 0.03);
    assert.ok(Math.max(...cold) > Math.max(...warm)); assert.ok(warm.every(value => value > 0));
    await slide(widget, 0.2); assert.deepEqual(await probabilities(), cold);
  });
});

test('parallel scheduling cannot beat the longest task or exceed sequential time', async () => {
  await withWidget(browser, 'parallel', async (widget, config) => {
    const time = async () => number((await readout(widget)).match(/Completion time: ([\d,.]+)/)[1]);
    const total = config.tasks.reduce((a, b) => a + b, 0);
    await slide(widget, 1); assert.equal(await time(), total);
    await slide(widget, 4); const parallel = await time();
    assert.ok(parallel >= Math.max(...config.tasks)); assert.ok(parallel >= total / 4); assert.ok(parallel <= total);
    assert.equal(await widget.locator('svg .task-label').count(), config.tasks.length);
    await slide(widget, 1); assert.equal(await time(), total);
  });
});

test('recurrent state stays fixed while explicit history grows', async () => {
  await withWidget(browser, 'state', async widget => {
    const counts = async () => {
      const text = await readout(widget);
      return [number(text.match(/= ([\d,.]+) scalars;/)[1]), number(text.match(/recurrent state = ([\d,.]+) scalars/)[1])];
    };
    await slide(widget, 128); const first = await counts();
    await slide(widget, 256); const second = await counts();
    assert.equal(second[0], first[0] * 2); assert.equal(second[1], first[1]);
    assert.match(await readout(widget), /Hybrid models also retain/);
  });
});

test('an attention sink reserves normalized mass without contributing a value', async () => {
  await withWidget(browser, 'sink', async (widget, config) => {
    const output = () => widget.evaluate(node => Number(node.dataset.value));
    const probabilities = () => widget.locator('svg text[text-anchor="end"]').allTextContents().then(values => values.map(value => number(value.replace('%','').trim())));
    await click(widget, 'Without sink');
    const mean = config.values.reduce((a,b)=>a+b,0)/config.values.length;
    near(await output(), mean); near((await probabilities()).at(-1), 0);
    await click(widget, 'With sink'); await slide(widget, 0);
    const equal = await probabilities(); equal.forEach(value=>near(value, 25));
    near(equal.reduce((a,b)=>a+b,0),100); near(await output(),mean*0.75);
    await slide(widget, 4); assert.ok(await output() < mean*0.1);
    await click(widget, 'Without sink'); near(await output(),mean);
  });
});

test('the wide reading column and margin captions fit around their breakpoint', async () => {
  for (const width of [768,1024,1279,1280,1920]) {
    for (const slug of ['2024-05-06-deepseek-v2','2025-08-05-gpt-oss']) {
      const {page,context,errors}=await open(browser,{slug,width});
      try {
        await assertLayout(page);
        const boxes=await page.locator('.fig').first().evaluate(node=>{
          const graphic=node.firstElementChild.getBoundingClientRect(),caption=node.querySelector('figcaption').getBoundingClientRect();
          return {graphic:{x:graphic.x,right:graphic.right,bottom:graphic.bottom},caption:{x:caption.x,y:caption.y,right:caption.right}};
        });
        assert.ok(boxes.caption.right <= width);
        if(width>=1280)assert.ok(boxes.caption.x >= boxes.graphic.right,'caption must sit beside the wider figure');
        else assert.ok(boxes.caption.y >= boxes.graphic.bottom,'caption must move below the figure before the columns stop fitting');
        assert.deepEqual(errors,[]);
      }finally{await context.close();}
    }
  }
});

test('research notes open as a rendered page with relative local assets', async () => {
  const {page,context,errors}=await open(browser,{width:390});
  try {
    await page.getByRole('link',{name:'Research method and coverage notes'}).click();
    assert.equal(await page.locator('h1').innerText(),'Research method and coverage');
    await assertLayout(page); await assertLocalLinks(page);
    assert.deepEqual(errors,[]);
  }finally{await context.close();}
});

test('verifier presets expose formatting rejection and an unchecked reasoning error', async () => {
  await withWidget(browser, 'verifier', async widget => {
    assert.equal(await widget.getByRole('group', { name: 'Response example', exact: true }).count(), 1);
    assert.equal(await widget.getByRole('group', { name: 'Checking rule', exact: true }).count(), 1);
    await click(widget, 'Formatted'); await click(widget, 'Exact string'); assert.match(await readout(widget), /^Rejected/);
    await click(widget, 'Final number'); assert.match(await readout(widget), /^Accepted/);
    await click(widget, 'Flawed reasoning'); assert.match(await readout(widget), /Accepted.*misses the incorrect intermediate arithmetic/);
    await click(widget, 'Wrong'); assert.match(await readout(widget), /^Rejected/);
    await click(widget, 'Correct'); await click(widget, 'Exact string'); assert.match(await readout(widget), /^Accepted/);
  });
});

test('data mixture endpoints preserve the total token budget', async () => {
  await withWidget(browser, 'mixture', async (widget, config) => {
    const portions = async () => (await readout(widget)).match(/([\d,.]+)B targeted \+ ([\d,.]+)B replay/).slice(1).map(number);
    await slide(widget, 0); assert.deepEqual(await portions(), [0, config.tokens]);
    await slide(widget, 100); assert.deepEqual(await portions(), [config.tokens, 0]);
    await slide(widget, 50); const halves = await portions(); near(halves[0], halves[1]); near(halves[0] + halves[1], config.tokens, 0.02);
  });
});

test('cache reuse and bit width compose without losing their separate effects', async () => {
  await withWidget(browser, 'cache', async widget => {
    const memory = async () => number((await readout(widget)).match(/= ([\d,.]+) MiB/)[1]);
    await slide(widget, 1); await click(widget, '16 bits'); const full = await memory();
    await click(widget, '8 bits'); near(await memory(), full / 2, 0.02);
    await click(widget, '4 bits'); near(await memory(), full / 4, 0.02);
    const before = await memory(); await slide(widget, 2); assert.ok(await memory() < before);
    await slide(widget, 1); near(await memory(), before);
    await click(widget, '16 bits'); near(await memory(), full);
  });
});

test('larger image patches reduce token count and squared attention pairs', async () => {
  await withWidget(browser, 'patches', async (widget, config) => {
    const counts = async () => (await readout(widget)).match(/= (\d+) patches; dense image self-attention has ([\d,]+) query-key pairs/).slice(1).map(number);
    await slide(widget, 1); const fine = await counts(); assert.equal(fine[0], config.width * config.height); assert.equal(fine[1], fine[0] ** 2);
    await slide(widget, 4); const coarse = await counts(); assert.equal(coarse[0], Math.ceil(config.width / 4) * Math.ceil(config.height / 4)); assert.equal(coarse[1], coarse[0] ** 2); assert.ok(coarse[0] < fine[0]);
    await slide(widget, 1); assert.deepEqual(await counts(), fine);
  });
});

test('ternary weights round, clip and return to the zero bin', async () => {
  await withWidget(browser, 'ternary', async widget => {
    const forward = async () => number((await widget.locator('svg').textContent()).match(/Forward: (−?[-\d.]+) ×/)[1].replace('−', '-'));
    await slide(widget, 0.2); assert.equal(await forward(), 0);
    await slide(widget, 0.3); assert.equal(await forward(), 1);
    await slide(widget, 1); assert.equal(await forward(), 1);
    await slide(widget, -1); assert.equal(await forward(), -1);
    await slide(widget, -0.2); assert.equal(await forward(), 0);
    await slide(widget, 0.25); assert.equal(await forward(), 1);
    await slide(widget, -0.25); assert.equal(await forward(), -1);
    assert.match(await readout(widget), /ties away from zero/);
  });
});

test('diffusion advances only on request; changing order and reset discard the old state', async () => {
  await withWidget(browser, 'diffusion', async (widget, _config, page) => {
    const unmasked = () => widget.locator('svg text[text-anchor="middle"]').allTextContents().then(values => values.map((value, index) => value === '[MASK]' ? null : index).filter(value => value !== null));
    assert.deepEqual(await unmasked(), []);
    await page.clock.runFor(60000); assert.deepEqual(await unmasked(), []);
    await click(widget, 'Denoise one round'); assert.deepEqual(await unmasked(), [2, 5]);
    await click(widget, 'Left-to-right order'); assert.deepEqual(await unmasked(), []);
    await click(widget, 'Denoise one round'); assert.deepEqual(await unmasked(), [0, 1]);
    await click(widget, 'Reset'); await page.clock.runFor(60000); assert.deepEqual(await unmasked(), []);
    for (let i = 0; i < 4; i++) await click(widget, 'Denoise one round');
    assert.equal((await unmasked()).length, 8); assert.equal(await widget.getByRole('button', { name: 'Denoise one round' }).isDisabled(), true);
    await click(widget, 'Scattered order'); assert.equal(await widget.getByRole('button', { name: 'Denoise one round' }).isEnabled(), true); assert.deepEqual(await unmasked(), []);
  }, { clock: true, reduced: false });
});

test('recurrence can retain a signal, decay it, and reset during an unfinished sequence', async () => {
  await withWidget(browser, 'recurrence', async (widget, _config, page) => {
    const state = async () => number((await readout(widget)).match(/State = ([\d,.]+)\./)[1]);
    await click(widget, 'Step'); assert.equal(await state(), 1);
    await click(widget, 'Step'); assert.equal(await state(), 1);
    await click(widget, 'Fixed gate'); assert.equal(await state(), 0);
    await click(widget, 'Step'); assert.equal(await state(), 0.5);
    await click(widget, 'Step'); assert.equal(await state(), 0.25);
    await click(widget, 'Reset'); await page.clock.runFor(60000); assert.equal(await state(), 0);
    for (let i = 0; i < 8; i++) await click(widget, 'Step');
    assert.equal(await widget.getByRole('button', { name: 'Step', exact: true }).isDisabled(), true);
    await click(widget, 'Selective gate'); assert.equal(await state(), 0); assert.equal(await widget.getByRole('button', { name: 'Step', exact: true }).isEnabled(), true);
  }, { clock: true, reduced: false });
});

test('sliders are labeled keyboard controls and update their visible value', async () => {
  await withWidget(browser, 'state', async (widget, _config, page) => {
    const input = widget.getByLabel('History tokens');
    await input.focus(); await page.keyboard.press('ArrowRight');
    const current = await input.inputValue();
    assert.equal(await widget.locator('output').innerText(), current);
    assert.match(await readout(widget), new RegExp(`^${current} tokens:`));
    const visibleFocus = await input.evaluate(node => {
      const style = getComputedStyle(node);
      return style.outlineStyle !== 'none' && parseFloat(style.outlineWidth) > 0 || style.boxShadow !== 'none';
    });
    assert.equal(visibleFocus, true, 'focused range control must remain visible to keyboard users');
  });
});

test('an individual lesson works when copied away from the collection', async () => {
  const folder = await mkdtemp(path.join(tmpdir(), 'open-weight-isolated-'));
  const entry = manifest.find(item => item.figures.includes('kv'));
  try {
    for (const file of ['index.html', 'styles.css', 'tutorial.js']) await cp(path.join(ROOT, entry.slug, file), path.join(folder, file));
    const { page, context, errors } = await open(browser, { url: pathToFileURL(path.join(folder, 'index.html')).href, width: 390 });
    try {
      const widget = page.locator('[data-widget="kv"]').first();
      await click(widget, 'MQA'); assert.match(await readout(widget), /× 1 KV heads/);
      await assertLayout(page); assert.deepEqual(errors, []);
    } finally { await context.close(); }
  } finally { await rm(folder, { recursive: true, force: true }); }
});

test('static assets and navigation work under a nested HTTP deployment prefix', async () => {
  const prefix = '/preview/visual-learn/';
  const failures = [];
  const server = createServer(async (request, response) => {
    try {
      const pathname = decodeURIComponent(new URL(request.url, 'http://test.invalid').pathname);
      if (!pathname.startsWith(prefix)) throw new Error('missing deployment prefix');
      let filename = path.resolve(REPO, pathname.slice(prefix.length));
      if (filename !== REPO && !filename.startsWith(REPO + path.sep)) throw new Error('outside repository');
      if ((await stat(filename)).isDirectory()) filename = path.join(filename, 'index.html');
      const mime = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json' }[path.extname(filename)] || 'text/plain';
      response.writeHead(200, { 'Content-Type': mime }); response.end(await readFile(filename));
    } catch (error) { failures.push(`${request.url}: ${error.message}`); response.writeHead(404); response.end('Not found'); }
  });
  await new Promise(resolve => server.listen(0, '0.0.0.0', resolve));
  const origin = `http://127.0.0.1:${server.address().port}`;
  const { page, context, errors } = await open(browser, { url: `${origin}${prefix}open-weight-llms/index.html` });
  try {
    await page.locator('.catalog h3 a').last().click();
    await page.locator('[data-widget] .widget-graphic').first().waitFor();
    assert.ok(page.url().startsWith(`${origin}${prefix}open-weight-llms/`));
    await assertLayout(page);
    await page.getByRole('link', { name: 'All open-weight tutorials', exact: true }).click();
    assert.equal(await page.locator('.catalog > li').count(), manifest.length);
    assert.deepEqual(errors, []); assert.deepEqual(failures, []);
  } finally { await context.close(); await new Promise(resolve => server.close(resolve)); }
});
