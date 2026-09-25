// Browser regression tests. Drives index.html in headless Chrome (the installed
// Google Chrome via Playwright's "chrome" channel, falling back to Playwright's
// own Chromium if that is unavailable). Run with `pnpm test`.
//
// The page is static: every comparison is drawn as small multiples, so there are
// no toggles or sliders to exercise. The tests check that each figure renders,
// that the simulation says what the prose promises, that computed captions match
// the drawn data, and that the two tables match the paper.
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

/** Open the page with web fonts stubbed so the tests run offline; collect page and console errors. */
async function open({ width = 1440 } = {}) {
  const context = await browser.newContext({ viewport: { width, height: 900 } });
  await context.route(/fonts\.(googleapis|gstatic)\.com/, (route) => route.fulfill({ status: 200, contentType: 'text/css', body: '' }));
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', (e) => errors.push('pageerror: ' + e.message));
  page.on('console', (m) => { if (m.type() === 'error') errors.push('console: ' + m.text()); });
  await page.goto(PAGE);
  await page.waitForSelector('#absorb-svg circle');
  return { page, context, errors };
}

const count = (page, sel) => page.$$eval(sel, (els) => els.length);

for (const width of [1440, 390]) {
  test(`every figure renders, without errors or horizontal overflow, with legible chart text at ${width}px`, async () => {
    const { page, context, errors } = await open({ width });
    for (const map of ['#map-input', '#map-attn-0', '#map-attn-4', '#map-norm']) {
      assert.equal(await count(page, `${map} rect[data-i]`), 196, `${map} has 14×14 cells`);
    }
    assert.equal(await count(page, '#depth-grid .map svg'), 12, 'three model sizes × four depths');
    assert.equal(await count(page, '#tokens-svg rect.tok'), 1 + 4 * 14, '[CLS] plus four rows of the strip');
    assert.equal(await count(page, '#dots-norm circle'), 196, 'one dot per patch token');
    assert.equal(await count(page, '#absorb-svg g.panel'), 3, 'panels for 0, 1 and 4 registers');
    assert.ok(await count(page, '#sequence-svg rect') > 0, 'sequence diagram drawn');

    // measured SVGs are drawn at their container width, so their text is not scaled down
    const svgs = await page.$$eval('#tokens-svg, #dots-norm, #sequence-svg, #absorb-svg', (els) => els.map((svg) => ({
      id: svg.id, drawn: +svg.getAttribute('width'), shown: Math.round(svg.getBoundingClientRect().width),
    })));
    for (const s of svgs) assert.ok(Math.abs(s.drawn - s.shown) <= 1, `${s.id} is drawn at ${s.drawn}px but shown at ${s.shown}px`);
    const smallest = await page.$$eval('svg text', (els) => Math.min(...els.filter((t) => t.getBoundingClientRect().width > 0).map((t) => {
      const svg = t.ownerSVGElement; const vb = svg.viewBox.baseVal;
      const scale = vb && vb.width ? svg.getBoundingClientRect().width / vb.width : 1;
      return parseFloat(getComputedStyle(t).fontSize) * scale;
    })));
    assert.ok(smallest >= 11, `smallest chart text is ${smallest.toFixed(1)}px`);

    await page.evaluate(async () => { for (let y = 0; y < document.body.scrollHeight; y += 600) { window.scrollTo(0, y); await new Promise((r) => setTimeout(r, 20)); } });
    await page.waitForTimeout(200);
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
    assert.equal(overflow, 0, 'page scrolls horizontally');
    assert.deepEqual(errors, []);
    await context.close();
  });
}

test('figures redraw at the new width when the viewport shrinks', async () => {
  const { page, context, errors } = await open({ width: 1440 });
  const wide = await page.$eval('#absorb-svg', (el) => +el.getAttribute('width'));
  await page.setViewportSize({ width: 390, height: 900 });
  await page.waitForFunction((w) => +document.getElementById('absorb-svg').getAttribute('width') !== w, wide);
  const narrow = await page.$eval('#absorb-svg', (el) => ({ drawn: +el.getAttribute('width'), shown: Math.round(el.getBoundingClientRect().width) }));
  assert.ok(narrow.drawn < wide);
  assert.ok(Math.abs(narrow.drawn - narrow.shown) <= 1);
  assert.deepEqual(errors, []);
  await context.close();
});

test('Figure 1: artifacts only without registers, and registers move attention onto the bird', async () => {
  const { page, context, errors } = await open();
  assert.ok(await count(page, '#map-attn-0 rect[data-artifact]') > 0, 'no-register map shows artifacts');
  assert.equal(await count(page, '#map-attn-4 rect[data-artifact]'), 0, 'four-register map has none');
  assert.equal(await count(page, '#map-attn-0 .artifact-mark'), await count(page, '#map-attn-0 rect[data-artifact]'), 'every artifact is marked');
  const d = await page.$eval('#attn-takeaway', (el) => ({ ...el.dataset, text: el.textContent }));
  assert.ok(+d.bird4 > +d.bird0, 'bird share rises with registers');
  assert.match(d.text, new RegExp(`${d.artifacts} sky patches take ${Math.round(d.artShare * 100)}%`));
  assert.match(d.text, new RegExp(`from ${Math.round(d.bird0 * 100)}% to ${Math.round(d.bird4 * 100)}%`));
  // the artifact tokens hold the largest attention shares in the no-register map
  const top = await page.$$eval('#map-attn-0 rect[data-i]', (els) => els.map((e) => ({ v: +e.dataset.v, a: e.hasAttribute('data-artifact') })).sort((a, b) => b.v - a.v));
  const nArt = top.filter((t) => t.a).length;
  assert.ok(top.slice(0, nArt).every((t) => t.a));
  assert.deepEqual(errors, []);
  await context.close();
});

test('Figure 2: position labels match row-major flattening with [CLS] at position 0', async () => {
  const { page, context, errors } = await open();
  const labels = await page.$$eval('#tokens-svg text[data-pos]', (els) => els.map((t) => ({ ...t.dataset, text: t.textContent })));
  assert.equal(labels.length, 8, 'four callouts, labeled in the grid and in the strip');
  for (const l of labels) {
    assert.equal(+l.pos, 1 + 14 * +l.row + +l.col);
    assert.equal(l.text, l.pos);
  }
  assert.deepEqual(labels.filter((l) => l.where === 'strip').map((l) => l.text), ['1', '15', '92', '196']);
  const first = await page.$eval('#tokens-svg rect.tok', (el) => ({ pos: el.dataset.pos, cls: el.classList.contains('cls') }));
  assert.deepEqual(first, { pos: '0', cls: true });
  // strip cells take the color of their patch
  const colors = await page.evaluate(() => {
    const strip = [...document.querySelectorAll('#tokens-svg rect.tok:not(.cls)')].map((r) => [+r.dataset.pos, r.getAttribute('fill')]);
    const grid = [...document.querySelectorAll('#map-input rect[data-i]')].map((r) => r.getAttribute('fill'));
    return strip.map(([pos, fill]) => fill === grid[pos - 1]);
  });
  assert.ok(colors.every(Boolean));
  assert.deepEqual(errors, []);
  await context.close();
});

test('Figure 3: artifacts are the tokens above the 150 cutoff, in the sky, and the caption matches the data', async () => {
  const { page, context, errors } = await open();
  const cells = await page.$$eval('#map-norm rect[data-i]', (els) => els.map((e) => ({ i: +e.dataset.i, v: +e.dataset.v, a: e.hasAttribute('data-artifact') })));
  const art = cells.filter((c) => c.a);
  assert.ok(art.length >= 3 && art.length <= 8, 'a few artifacts, about 2–4% of 196 tokens');
  assert.ok(cells.every((c) => c.a === c.v > 150), 'artifact means norm above 150');
  const content = await page.evaluate((idx) => idx.map((i) => window.__registerSim.scene[i]), art.map((c) => c.i));
  assert.ok(content.every((l) => l === 0), 'artifacts sit on redundant sky patches, never on the bird');
  assert.equal(await count(page, '#dots-norm circle.art'), art.length);
  const d = await page.$eval('#where-takeaway', (el) => ({ ...el.dataset, text: el.textContent }));
  assert.equal(+d.artifacts, art.length);
  assert.equal(+d.normal, 196 - art.length);
  assert.ok(+d.normalMax < 150 && +d.artifactMin > 150);
  assert.match(d.text, /sky tokens/);
  assert.deepEqual(errors, []);
  await context.close();
});

test('Figure 4: no artifacts in DINOv2 ViT-S or ViT-B at any depth; ViT-L has them only from the middle layers (paper Fig. 4)', async () => {
  const { page, context, errors } = await open();
  const maps = await page.$$eval('#depth-grid .map', (els) => els.map((e) => ({ ...e.dataset, head: e.querySelector('h4').textContent, note: e.querySelector('.note').textContent })));
  assert.equal(maps.length, 12);
  for (const m of maps) {
    assert.equal(m.head, `layer ${m.layer}`, 'heading matches the layer simulated');
    assert.equal(m.note, +m.artifacts ? `${m.artifacts} above 150` : 'none above 150', 'note matches the count');
  }
  for (const m of maps.filter((x) => x.model !== 'L')) assert.equal(+m.artifacts, 0, `ViT-${m.model} layer ${m.layer}`);
  const L = Object.fromEntries(maps.filter((x) => x.model === 'L').map((x) => [x.layer, +x.artifacts]));
  assert.deepEqual(Object.keys(L), ['6', '12', '18', '24']);
  assert.equal(L[6], 0, 'early layers are clean');
  assert.ok(L[12] > 0 && L[18] >= L[12] && L[24] === L[18], 'artifacts appear mid-network and persist');
  assert.deepEqual(errors, []);
  await context.close();
});

test('Figures 6 and 7: registers follow [CLS] and precede the patches; one register removes every artifact and holds the high norm', async () => {
  const { page, context, errors } = await open();
  const order = await page.$$eval('#sequence-svg rect.in', (els) => els.map((e) => e.dataset.kind));
  assert.deepEqual(order, ['cls', 'reg', 'reg', 'reg', 'reg', 'pat', 'pat', 'pat'], 'DINOv2 order: [CLS], registers, patches');
  assert.equal(await count(page, '#sequence-svg rect.out[data-kind="reg"][stroke-dasharray="3 3"]'), 4, 'register outputs drawn as discarded');

  const panels = await page.$$eval('#absorb-svg g.panel', (gs) => gs.map((g) => ({
    n: +g.dataset.registers, reg: g.querySelectorAll('circle.reg').length, art: g.querySelectorAll('circle.art').length, cls: g.querySelectorAll('circle.cls').length,
  })));
  assert.deepEqual(panels.map((p) => p.n), [0, 1, 4]);
  assert.ok(panels[0].art > 0 && panels[0].reg === 0);
  assert.deepEqual(panels.slice(1).map((p) => [p.reg, p.art, p.cls]), [[1, 0, 1], [4, 0, 1]]);
  const sim = await page.evaluate(() => { const s = window.__registerSim; return [0, 1, 4].map((n) => { const r = s.simulate('L', 24, n); return { art: r.artifacts.size, maxPatch: Math.max(...r.patch), minReg: r.regs.length ? Math.min(...r.regs) : null }; }); });
  assert.ok(sim[1].maxPatch < 150 && sim[2].maxPatch < 150, 'no patch token crosses the cutoff with registers');
  assert.ok(sim[1].minReg > 150 && sim[2].minReg > 150, 'the registers carry the high norms');
  const t = await page.$eval('#absorb-takeaway', (el) => ({ ...el.dataset, text: el.textContent }));
  assert.equal(+t.artifacts0, panels[0].art);
  assert.match(t.text, new RegExp(`highest patch norm is ${Math.round(+t.maxPatch)}`));
  assert.deepEqual(errors, []);
  await context.close();
});

test('probe table matches the paper (Fig. 5b and Table 1, DINOv2 ViT-g)', async () => {
  const { page, context, errors } = await open();
  const rows = await page.$$eval('#probe-table tbody tr', (trs) => trs.map((tr) => [...tr.children].slice(1).map((td) => td.innerText.trim())));
  assert.deepEqual(rows, [
    ['41.7', '22.8', '–'],
    ['0.79', '5.09', '–'],
    ['18.38', '25.23', '–'],
    ['65.8', '69.0', '86.0'],
    ['17.1', '79.1', '87.3'],
  ]);
  assert.deepEqual(errors, []);
  await context.close();
});

test('results table matches the paper (ICLR 2024, Tables 2a and 3), and regressions are marked as such', async () => {
  const { page, context, errors } = await open();
  const rows = await page.$$eval('#results-table tbody tr', (trs) => trs.map((tr) => [...tr.children].map((td) => (td.querySelector('.v') || td).textContent.replace(/\s+/g, ' ').trim())));
  assert.deepEqual(rows, [
    ['DeiT-III (ViT-B)', '0', '84.7', '38.9', '0.511', '11.7'],
    ['', '4', '84.7', '39.1', '0.512', '27.1'],
    ['OpenCLIP (ViT-B)', '0', '78.2', '26.6', '0.702', '38.8'],
    ['', '4', '78.1', '26.7', '0.661', '37.1'],
    ['DINOv2 (ViT-L)', '0', '84.3', '46.6', '0.378', '35.3'],
    ['', '4', '84.8', '47.9', '0.366', '55.4'],
  ]);
  const marks = (n) => page.$$eval(`#results-table tbody tr:nth-child(${n}) td`, (tds) => tds.slice(2).map((td) => `${td.className.replace('n', '').trim()}:${td.querySelector('.d').textContent}`));
  assert.deepEqual(await marks(2), ['same:=', 'up:▲', 'down:▼', 'up:▲'], 'DeiT-III: NYUd RMSE rose, and lower is better');
  assert.deepEqual(await marks(4), ['down:▼', 'up:▲', 'up:▲', 'down:▼']);
  assert.deepEqual(await marks(6), ['up:▲', 'up:▲', 'up:▲', 'up:▲']);
  assert.deepEqual(errors, []);
  await context.close();
});

test('top bar marks the section being read and tracks reading progress', async () => {
  const { page, context, errors } = await open();
  assert.equal(await count(page, '.topbar nav a[aria-current="true"]'), 0, 'nothing current above the first section');
  await page.$eval('#registers', (el) => el.scrollIntoView());
  await page.waitForFunction(() => document.querySelector('.topbar nav a[aria-current="true"]')?.getAttribute('href') === '#registers');
  const w = await page.$eval('#progress', (el) => parseFloat(el.style.width));
  assert.ok(w > 20 && w < 100, `progress ${w}%`);
  assert.equal(await page.$eval('.topbar .home', (el) => el.getAttribute('href')), '../');
  assert.deepEqual(errors, []);
  await context.close();
});

test('every source link from the previous version is still on the page', async () => {
  const { page, context, errors } = await open();
  const hrefs = await page.$$eval('a[href]', (as) => as.map((a) => a.getAttribute('href')));
  for (const id of ['2309.16588', '2304.07193', '2010.11929', '2309.17453', '2402.17762', '2506.08010']) {
    assert.ok(hrefs.includes(`https://arxiv.org/abs/${id}`), id);
  }
  assert.ok(hrefs.every((h) => !h.startsWith('/')), 'relative or absolute-URL links only, never root-relative');
  assert.deepEqual(errors, []);
  await context.close();
});
