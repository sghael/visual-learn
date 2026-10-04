import assert from 'node:assert/strict';
import { readFile, mkdir, stat } from 'node:fs/promises';
import path from 'node:path';
import { tmpdir } from 'node:os';
import { fileURLToPath, pathToFileURL } from 'node:url';

export const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
export const REPO = path.dirname(ROOT);
export const CAPTURE_DIR = process.env.CAPTURE_DIR || path.join(process.platform === 'darwin' ? '/private/tmp' : tmpdir(), 'open-weight-qa');
export const manifest = JSON.parse(await readFile(path.join(ROOT, 'manifest.json'), 'utf8'));
export const lessons = new Map(await Promise.all(manifest.map(async item => [
  item.slug, JSON.parse(await readFile(path.join(ROOT, item.slug, 'lesson.json'), 'utf8')),
])));

export const localURL = slug => pathToFileURL(path.join(ROOT, slug || '', 'index.html')).href;

export async function open(browser, { slug = '', width = 1440, url = localURL(slug), clock = false, reduced = true } = {}) {
  const context = await browser.newContext({
    viewport: { width, height: 900 },
    reducedMotion: reduced ? 'reduce' : 'no-preference',
  });
  // Fonts must not make offline tests depend on Google's availability or pixel fit.
  await context.route(/https:\/\/fonts\.(googleapis|gstatic)\.com\//, route => route.fulfill({
    status: 200, contentType: 'text/css', body: '',
  }));
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', error => errors.push(`pageerror: ${error.message}`));
  page.on('console', message => { if (message.type() === 'error') errors.push(`console: ${message.text()}`); });
  if (clock) await page.clock.install();
  await page.goto(url, { waitUntil: 'load' });
  await page.locator('main h1').waitFor();
  return { page, context, errors };
}

export async function assertLayout(page) {
  const metrics = await page.evaluate(() => {
    const svgText = [...document.querySelectorAll('svg text')].filter(node => node.textContent.trim());
    const undersized = svgText.flatMap(node => {
      const matrix = node.getScreenCTM();
      if (!matrix) return [{ text: node.textContent, reason: 'missing transform' }];
      // The screen CTM includes viewBox scaling and nested SVG/CSS transforms.
      const scale = Math.min(Math.hypot(matrix.a, matrix.b), Math.hypot(matrix.c, matrix.d));
      const px = parseFloat(getComputedStyle(node).fontSize) * scale;
      return Number.isFinite(px) && px >= 10.95 ? [] : [{ text: node.textContent, px }];
    });
    return {
      overflow: Math.max(document.documentElement.scrollWidth, document.body.scrollWidth) - document.documentElement.clientWidth,
      undersized,
      duplicateIDs: [...document.querySelectorAll('[id]')].map(node => node.id).filter((id, i, all) => all.indexOf(id) !== i),
      emptyGraphics: [...document.querySelectorAll('[data-widget]')].filter(node => !node.querySelector('.widget-graphic')?.textContent.trim()).map(node => node.dataset.widget),
    };
  });
  assert.ok(metrics.overflow <= 0, `horizontal document overflow: ${metrics.overflow}px`);
  assert.deepEqual(metrics.undersized, [], 'SVG labels must render at least 11 CSS px');
  assert.deepEqual(metrics.duplicateIDs, [], 'duplicate IDs break anchors and control labels');
  assert.deepEqual(metrics.emptyGraphics, [], 'every figure must have a visible resting state');
}

const targetCache = new Map();
async function targetHTML(filename) {
  if (!targetCache.has(filename)) targetCache.set(filename, readFile(filename, 'utf8'));
  return targetCache.get(filename);
}

export async function assertLocalLinks(page) {
  const refs = await page.locator('a[href], link[href], script[src]').evaluateAll(nodes => nodes.map(node => ({
    tag: node.tagName, raw: node.getAttribute('href') ?? node.getAttribute('src'),
  })));
  const base = page.url();
  assert.ok(base.startsWith('file:'), 'link validation expects the disk version');
  for (const { tag, raw } of refs) {
    if (/^(https?:|mailto:|data:)/i.test(raw)) continue;
    assert.ok(!raw.startsWith('/'), `${tag} must use a relative local URL: ${raw}`);
    assert.ok(!/^[a-z][a-z\d+.-]*:/i.test(raw), `unexpected local URL scheme: ${raw}`);
    const resolved = new URL(raw, base);
    let filename = fileURLToPath(resolved);
    let info;
    try { info = await stat(filename); }
    catch { assert.fail(`missing local ${tag} target: ${raw} from ${base}`); }
    if (info.isDirectory()) filename = path.join(filename, 'index.html');
    assert.ok((await stat(filename)).isFile(), `not a file: ${filename}`);
    if (resolved.hash && /\.html$/i.test(filename)) {
      const id = decodeURIComponent(resolved.hash.slice(1));
      const html = await targetHTML(filename);
      const ids = [...html.matchAll(/\bid\s*=\s*["']([^"']+)["']/g)].map(match => match[1]);
      assert.ok(ids.includes(id), `missing anchor ${id} in ${filename}, linked from ${base}`);
    }
  }
}

export async function capturePage(page, name, width) {
  if (process.env.CAPTURE !== '1') return;
  const folder = path.join(CAPTURE_DIR, `${name}-${width}`);
  await mkdir(folder, { recursive: true });
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({ path: path.join(folder, '00-header.png'), fullPage: false });
  const figures = page.locator('figure');
  for (let i = 0; i < await figures.count(); i++) {
    const box = await figures.nth(i).boundingBox();
    assert.ok(box, `figure ${i + 1} has no bounding box`);
    const absoluteTop = box.y + await page.evaluate(() => window.scrollY);
    // Viewport chunks avoid Chrome's corruption of exceptionally tall screenshots.
    const parts = Math.max(1, Math.ceil((box.height + 80) / 760));
    for (let part = 0; part < parts; part++) {
      await page.evaluate(y => window.scrollTo(0, y), Math.max(0, absoluteTop - 90 + part * 760));
      await page.screenshot({ path: path.join(folder, `${String(i + 1).padStart(2, '0')}-figure-${part + 1}.png`), fullPage: false });
    }
  }
}

export const near = (actual, expected, tolerance = 0.001) => assert.ok(
  Math.abs(actual - expected) <= tolerance,
  `expected ${actual} to be within ${tolerance} of ${expected}`,
);
export const number = text => Number(String(text).replaceAll(',', ''));
export const readout = widget => widget.locator('.widget-result').innerText();
export const click = (widget, name) => widget.getByRole('button', { name, exact: true }).click();
export const slide = (widget, value, index = 0) => widget.locator('input[type="range"]').nth(index).fill(String(value));

export async function withWidget(browser, kind, run, options = {}) {
  const entries = manifest.filter(item => item.figures.includes(kind));
  assert.ok(entries.length, `manifest has no ${kind} example`);
  for (const entry of entries) {
    const figures = lessons.get(entry.slug).sections.filter(section => section.figure?.kind === kind);
    for (const [index, section] of figures.entries()) {
      const session = await open(browser, { slug: entry.slug, ...options });
      try {
        const widget = session.page.locator(`[data-widget="${kind}"]`).nth(index);
        await widget.locator('.widget-graphic').waitFor();
        await run(widget, section.figure.config, session.page);
        assert.deepEqual(session.errors, [], `${entry.slug}: ${kind} emitted browser errors`);
      } catch (error) {
        error.message = `${entry.slug} (${kind}): ${error.message}`;
        throw error;
      } finally { await session.context.close(); }
    }
  }
}
