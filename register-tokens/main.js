// Register tokens explainer: a small deterministic simulation and the figures drawn from it.
// No neural network runs here. The model below is written to reproduce the paper's
// qualitative findings (Darcet et al., ICLR 2024); every number it produces is illustrative.
(() => {
'use strict';

const N = 14;            // patches per side
const P = N * N;         // 196 patch tokens
const CUTOFF = 150;      // the paper's hand-picked high-norm cutoff for DINOv2 (§2.1)
const NORM_MAX = 450;    // x-axis limit for the norm dot plots

// ---------- colors, read once from the stylesheet ----------
const css = getComputedStyle(document.documentElement);
const v = (name) => css.getPropertyValue(name).trim();
const COL = {
  cls: v('--c-purple'), reg: v('--c-green'), art: v('--c-orange'), patch: v('--c-grey'),
  ink: v('--ink'), ink2: v('--ink-2'), ink3: v('--ink-3'), rule: v('--rule'), rule2: v('--rule-2'), paper: v('--paper'),
  attnLo: v('--attn-lo'), attnHi: v('--attn-hi'), normLo: v('--norm-lo'), normHi: v('--norm-hi'),
  sky: v('--sky'), ground: v('--ground'), obj: v('--obj'), obj2: v('--obj-2'),
};
function hex2rgb(h) { h = h.replace('#', ''); return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16)); }
function mix(a, b, t) { const A = hex2rgb(a), B = hex2rgb(b); return `rgb(${A.map((x, i) => Math.round(x + (B[i] - x) * t)).join(',')})`; }

// ---------- deterministic random numbers, so the page is identical on every load ----------
function rng(seed) { let s = seed >>> 0; return () => { s = (s + 0x6D2B79F5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
const rand = (seed) => rng(seed)();

// ---------- the simulated image: 0 sky, 1 ground, 2 bird ----------
const scene = [];
for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
  const hill = y > 9.5 + 1.2 * Math.sin(x * 0.9);
  const dx = (x - 6.5) / 3.4, dy = (y - 6) / 2.2;
  const body = dx * dx + dy * dy < 1;
  const head = ((x - 9.6) / 1.3) ** 2 + ((y - 4.6) / 1.3) ** 2 < 1;
  const wing = Math.abs(x - 6) < 1.6 && y > 2.2 && y < 5;
  scene.push(body || head || wing ? 2 : hill ? 1 : 0);
}
const SCENE_NAME = ['sky', 'ground', 'bird'];
// Information in a patch: the fraction of its 4 neighbors with different content, with a floor for textured regions.
const info = scene.map((l, i) => {
  const x = i % N, y = (i / N) | 0; let diff = 0, cnt = 0;
  [[1, 0], [-1, 0], [0, 1], [0, -1]].forEach(([dx, dy]) => { const xx = x + dx, yy = y + dy; if (xx < 0 || yy < 0 || xx >= N || yy >= N) return; cnt++; if (scene[yy * N + xx] !== l) diff++; });
  let val = diff / cnt;
  if (l === 2) val = Math.max(val, 0.45);
  if (l === 1) val = Math.max(val, 0.18);
  return val;
});
const redundancy = info.map((x) => 1 - Math.min(1, x * 1.4));
// Candidate artifact positions: the most redundant non-bird patches, in a seeded order (paper Fig. 5a).
const r7 = rng(7);
const CANDIDATES = info.map((_, i) => ({ i, s: redundancy[i] + (r7() - 0.5) * 0.25 }))
  .filter((o) => scene[o.i] !== 2 && redundancy[o.i] > 0.85)
  .sort((a, b) => b.s - a.s).map((o) => o.i).slice(0, 5);
const ONSET = [9, 10, 11, 12, 13];                      // layer at which each candidate starts to grow (ViT-L, 24 layers)
const PEAK = CANDIDATES.map((_, k) => 300 + 100 * rand(k * 17 + 5));
const JITTER = Array.from({ length: P }, (_, i) => rand(i * 97 + 13));

// DINOv2 size study (paper Fig. 4c): outliers only from ViT-L up. ViT-S and ViT-B stay clean.
const MODELS = {
  S: { name: 'DINOv2 ViT-S', layers: 12, artifacts: false },
  B: { name: 'DINOv2 ViT-B', layers: 12, artifacts: false },
  L: { name: 'DINOv2 ViT-L', layers: 24, artifacts: true },
};

/** Output norms of every token of a simulated model at a given layer. An artifact is any patch token above CUTOFF. */
function simulate(model, layer, nreg) {
  const m = MODELS[model];
  const t = layer / m.layers;
  const base = 12 + 40 * t;
  const patch = info.map((x, i) => base * (0.8 + 0.35 * x) * (0.92 + 0.16 * JITTER[i]));
  if (m.artifacts && nreg === 0) {
    CANDIDATES.forEach((i, k) => {
      if (layer < ONSET[k]) return;
      const f = Math.min(1, (layer - ONSET[k] + 1) / 3);
      patch[i] += (PEAK[k] - patch[i]) * f;
    });
  }
  const artifacts = new Set();
  patch.forEach((x, i) => { if (x > CUTOFF) artifacts.add(i); });
  const regs = Array.from({ length: nreg }, (_, k) => 270 + 130 * rand(k * 29 + nreg * 7 + 11));
  const cls = 70 + 25 * t;
  return { patch, artifacts, regs, cls };
}

/** Share of the last-layer [CLS] attention that goes to each patch, simulated DINOv2 ViT-L. */
function attention(nreg) {
  const s = simulate('L', 24, nreg);
  const raw = scene.map((l, i) => {
    const j = rand(i * 13 + 5) * 0.03;
    return l === 2 ? 0.45 + 0.55 * info[i] + j : 0.05 + 0.12 * info[i] + j;
  });
  s.artifacts.forEach((i) => { raw[i] = 3 + rand(i + 101); });
  const sum = raw.reduce((a, b) => a + b, 0);
  return { share: raw.map((x) => x / sum), artifacts: s.artifacts };
}

// ---------- SVG helpers ----------
const NS = 'http://www.w3.org/2000/svg';
function el(tag, attrs, parent) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
}
function txt(parent, x, y, str, cls, attrs = {}) { const t = el('text', { x, y, class: cls, ...attrs }, parent); t.textContent = str; return t; }
function clear(svg) { while (svg.firstChild) svg.removeChild(svg.firstChild); }
const fmt = (x, d = 0) => x.toFixed(d);
const pct = (x) => `${(x * 100).toFixed(0)}%`;

const sceneColor = (i) => scene[i] === 2 ? (i % 3 === 0 ? COL.obj2 : COL.obj) : scene[i] === 1 ? COL.ground : COL.sky;
const normColor = (x) => x > CUTOFF ? COL.art : mix(COL.normLo, COL.normHi, Math.min(1, x / CUTOFF));

/** Draw a 14 x 14 map into an SVG with viewBox 0 0 14 14. The map has no text, so scaling it is safe. */
function drawMap(svg, fill, { artifacts = null, outline = false, values = null } = {}) {
  clear(svg);
  for (let i = 0; i < P; i++) {
    const r = el('rect', { x: i % N, y: (i / N) | 0, width: 1.02, height: 1.02, fill: fill(i), 'data-i': i }, svg);
    if (values) r.setAttribute('data-v', values[i].toFixed(5));
    if (artifacts && artifacts.has(i)) r.setAttribute('data-artifact', '');
  }
  if (outline) {
    let d = '';
    for (let i = 0; i < P; i++) {
      const x = i % N, y = (i / N) | 0;
      if (x < N - 1 && scene[i] !== scene[i + 1]) d += `M${x + 1} ${y}v1`;
      if (y < N - 1 && scene[i] !== scene[i + N]) d += `M${x} ${y + 1}h1`;
    }
    el('path', { d, fill: 'none', stroke: COL.ink, 'stroke-opacity': 0.45, 'stroke-width': 1, 'vector-effect': 'non-scaling-stroke', 'shape-rendering': 'geometricPrecision' }, svg);
  }
  if (artifacts) artifacts.forEach((i) => {
    el('rect', { x: (i % N) + 0.1, y: ((i / N) | 0) + 0.1, width: 0.8, height: 0.8, fill: 'none', stroke: COL.art, 'stroke-width': 2, 'vector-effect': 'non-scaling-stroke', class: 'artifact-mark' }, svg);
  });
}

// Redraw measured SVGs when their container changes width, so text stays at its CSS size.
function measured(container, draw) {
  let last = -1;
  const run = () => { const w = Math.round(container.clientWidth); if (w > 0 && w !== last) { last = w; draw(w); } };
  run();
  if ('ResizeObserver' in window) new ResizeObserver(run).observe(container);
  else window.addEventListener('resize', run);
}

// =============== Figure 1: attention without and with registers ===============
const A0 = attention(0), A4 = attention(4);
const ATTN_MAX = Math.max(...A0.share, ...A4.share);
const attnColor = (x) => mix(COL.attnLo, COL.attnHi, Math.pow(x / ATTN_MAX, 0.45));
{
  drawMap(document.getElementById('map-input'), sceneColor);
  drawMap(document.getElementById('map-attn-0'), (i) => attnColor(A0.share[i]), { artifacts: A0.artifacts, values: A0.share });
  drawMap(document.getElementById('map-attn-4'), (i) => attnColor(A4.share[i]), { artifacts: A4.artifacts, values: A4.share });

  // color key: eight steps of the shared scale, labeled at both ends, plus the artifact mark
  const key = document.getElementById('attn-key');
  const steps = 8, sw = 16;
  const ramp = `<svg width="${steps * sw}" height="26" aria-hidden="true">${Array.from({ length: steps }, (_, k) => `<rect x="${k * sw}" y="0" width="${sw}" height="10" fill="${attnColor(ATTN_MAX * ((k + 0.5) / steps) ** (1 / 0.45))}"/>`).join('')}
    <text x="0" y="23" style="font:11.5px var(--sans);fill:var(--ink-3)">0%</text><text x="${steps * sw}" y="23" text-anchor="end" style="font:11.5px var(--sans);fill:var(--ink-3)">${(ATTN_MAX * 100).toFixed(1)}%</text></svg>`;
  key.innerHTML = `<span class="item">${ramp}<span>share of <code>[CLS]</code>’s attention<br>to patches, per patch</span></span>
    <span class="item"><svg width="14" height="14" aria-hidden="true"><rect x="1.5" y="1.5" width="11" height="11" fill="none" stroke="${COL.art}" stroke-width="2"/></svg><span>artifact token</span></span>`;

  const sum = (share, pred) => share.reduce((a, x, i) => a + (pred(i) ? x : 0), 0);
  const artShare = sum(A0.share, (i) => A0.artifacts.has(i));
  const bird0 = sum(A0.share, (i) => scene[i] === 2), bird4 = sum(A4.share, (i) => scene[i] === 2);
  const t = document.getElementById('attn-takeaway');
  t.textContent = `Without registers, ${A0.artifacts.size} sky patches take ${pct(artShare)} of the attention; with four registers the bird’s share rises from ${pct(bird0)} to ${pct(bird4)}.`;
  t.dataset.artifacts = A0.artifacts.size; t.dataset.artShare = artShare; t.dataset.bird0 = bird0; t.dataset.bird4 = bird4;
}

// =============== Figure 2: patches to tokens ===============
const CALLOUTS = [[0, 0], [1, 0], [6, 7], [13, 13]];   // [row, column]
const ROWS_SHOWN = [0, 1, 6, 13];
function drawTokens(W) {
  const svg = document.getElementById('tokens-svg');
  clear(svg);
  const LW = 44;                                        // column for row labels
  const G = Math.min(224, Math.round((W - LW - 44) * 0.9)), cell = G / N, gy = 24;
  txt(svg, LW, 14, '14 × 14 patches', 'label muted');
  for (let i = 0; i < P; i++) el('rect', { x: LW + (i % N) * cell, y: gy + ((i / N) | 0) * cell, width: cell + 0.3, height: cell + 0.3, fill: sceneColor(i) }, svg);
  ROWS_SHOWN.forEach((row) => txt(svg, LW - 8, gy + row * cell + cell / 2 + 4, `row ${row}`, 'tick-t', { 'text-anchor': 'end' }));
  const px = LW + G + 10;                               // position labels to the right of the grid
  txt(svg, px, 14, 'position', 'label muted');
  if (W - px > 330) {
    const ax = px + 90;
    txt(svg, ax, gy + G * 0.42, 'position = 1 + 14 × row + column', 'annot');
    txt(svg, ax, gy + G * 0.42 + 20, '[CLS] takes position 0', 'annot');
  }

  // the strip: [CLS], then rows 0, 1, 6 and 13 with the others elided
  const gC = 20, gRow = 4, gE = 22;
  const cw = Math.min(12, (W - gC - gRow - 2 * gE) / 57);
  const sy = gy + G + 46, sh = 22;
  const segX = {};
  let x = 0;
  el('rect', { x, y: sy, width: cw, height: sh, fill: COL.cls, 'data-pos': 0, class: 'tok cls' }, svg);
  txt(svg, 0, sy - 7, '[CLS]', 'label');
  txt(svg, 0, sy + sh + 15, '0', 'tick-t strong');
  x += cw + gC;
  ROWS_SHOWN.forEach((row, k) => {
    if (k > 0) {
      const gap = ROWS_SHOWN[k] - ROWS_SHOWN[k - 1] > 1 ? gE : gRow;
      if (gap === gE) txt(svg, x + gE / 2, sy + sh / 2 + 4, '…', 'label muted', { 'text-anchor': 'middle' });
      x += gap;
    }
    segX[row] = x;
    for (let c = 0; c < N; c++) {
      const i = row * N + c;
      el('rect', { x: x + c * cw, y: sy, width: Math.max(1, cw - (cw > 5 ? 1 : 0)), height: sh, fill: sceneColor(i), 'data-pos': 1 + i, class: 'tok' }, svg);
    }
    txt(svg, x, sy + sh + 31, `row ${row}`, 'tick-t');
    x += N * cw;
  });

  // four patches, marked in the grid and in the strip with the same position number
  CALLOUTS.forEach(([row, col], k) => {
    const pos = 1 + row * N + col;
    const cx = LW + col * cell + cell / 2, cy = gy + row * cell + cell / 2;
    el('line', { x1: cx, y1: cy, x2: px - 3, y2: cy, stroke: COL.ink2, 'stroke-width': 0.9 }, svg);
    el('rect', { x: LW + col * cell, y: gy + row * cell, width: cell, height: cell, fill: 'none', stroke: COL.ink, 'stroke-width': 1.5 }, svg);
    el('circle', { cx, cy, r: 2, fill: COL.ink }, svg);
    const g = txt(svg, px, cy + 4, String(pos), 'tick-t strong');
    Object.assign(g.dataset, { row, col, pos, where: 'grid' });
    el('rect', { x: segX[row] + col * cw - 0.5, y: sy - 0.5, width: cw, height: sh + 1, fill: 'none', stroke: COL.ink, 'stroke-width': 1.5, class: 'callout', 'data-pos': pos }, svg);
    const last = k === CALLOUTS.length - 1;
    const s = txt(svg, last ? segX[row] + (col + 1) * cw : segX[row] + col * cw + cw / 2, sy + sh + 15, String(pos), 'tick-t strong', { 'text-anchor': last ? 'end' : 'middle' });
    Object.assign(s.dataset, { row, col, pos, where: 'strip' });
  });
  const H = sy + sh + 40;
  svg.setAttribute('width', W); svg.setAttribute('height', H); svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
}
measured(document.getElementById('tokens-graphic'), drawTokens);

// =============== Figure 3: where the high norms sit ===============
const FINAL = simulate('L', 24, 0);
drawMap(document.getElementById('map-norm'), (i) => normColor(FINAL.patch[i]), { outline: true, values: FINAL.patch, artifacts: null });
document.querySelectorAll('#map-norm rect[data-i]').forEach((r) => { if (FINAL.artifacts.has(+r.dataset.i)) r.setAttribute('data-artifact', ''); });

function drawAxis(svg, xs, y, left, right, title) {
  const g = el('g', { class: 'axis' }, svg);
  el('line', { x1: left, x2: right, y1: y, y2: y }, g);
  [0, 150, 300, 450].forEach((t) => {
    el('line', { x1: xs(t), x2: xs(t), y1: y, y2: y + 4 }, g);
    txt(g, xs(t), y + 16, t === CUTOFF ? '150 cutoff' : String(t), '', { 'text-anchor': t === 0 ? 'start' : t === 450 ? 'end' : 'middle' });
  });
  txt(svg, right, y + 32, title, 'axis-title', { 'text-anchor': 'end' });
}

function stats(list) { const s = [...list].sort((a, b) => a - b); return { min: s[0], max: s[s.length - 1], med: s[Math.floor(s.length / 2)], n: s.length }; }

function drawWhere(W) {
  const svg = document.getElementById('dots-norm');
  clear(svg);
  const L = 4, R = 10, H = 170, top = 44, band = 58, ay = top + band + 12;
  const xs = (x) => L + (Math.min(x, NORM_MAX) / NORM_MAX) * (W - L - R);
  el('line', { x1: xs(CUTOFF), x2: xs(CUTOFF), y1: top - 4, y2: ay, stroke: COL.rule2, 'stroke-width': 1, 'stroke-dasharray': '3 3' }, svg);
  const normal = [], art = [];
  FINAL.patch.forEach((x, i) => {
    const isArt = FINAL.artifacts.has(i);
    (isArt ? art : normal).push(x);
    el('circle', { cx: xs(x), cy: top + 4 + rand(i * 7 + 3) * (band - 8), r: isArt ? 3.2 : 2.5, fill: isArt ? COL.art : COL.patch, 'fill-opacity': isArt ? 1 : 0.75, class: isArt ? 'dot art' : 'dot' }, svg);
  });
  const n = stats(normal), a = stats(art);
  txt(svg, xs(n.min), 16, `${n.n} patch tokens`, 'label');
  txt(svg, xs(n.min), 31, `norm ${fmt(n.min)}–${fmt(n.max)}`, 'label muted');
  txt(svg, xs(a.max), 16, `${a.n} artifact tokens`, 'label', { 'text-anchor': 'end', fill: COL.art, style: `fill:${COL.art}` });
  txt(svg, xs(a.max), 31, `norm ${fmt(a.min)}–${fmt(a.max)}`, 'label muted', { 'text-anchor': 'end' });
  drawAxis(svg, xs, ay, L, W - R, 'output token norm (simulated)');
  svg.setAttribute('width', W); svg.setAttribute('height', H); svg.setAttribute('viewBox', `0 0 ${W} ${H}`);

  const t = document.getElementById('where-takeaway');
  const where = [...new Set([...FINAL.artifacts].map((i) => SCENE_NAME[scene[i]]))].join(' and ');
  t.textContent = `The norms fall into two groups: ${n.n} tokens between ${fmt(n.min)} and ${fmt(n.max)}, and ${a.n} ${where} tokens between ${fmt(a.min)} and ${fmt(a.max)}, about ${Math.round(a.med / n.med)} times the typical norm.`;
  Object.assign(t.dataset, { normal: n.n, artifacts: a.n, normalMax: n.max, artifactMin: a.min });
}
measured(document.querySelector('#fig-where .dots'), drawWhere);

// =============== Figure 4: depth x model size ===============
{
  const grid = document.getElementById('depth-grid');
  for (const key of ['S', 'B', 'L']) {
    const m = MODELS[key];
    const lab = document.createElement('div');
    lab.className = 'rowlab';
    lab.innerHTML = `${m.name}<span>${m.layers} layers</span>`;
    grid.appendChild(lab);
    for (let q = 1; q <= 4; q++) {
      const layer = Math.round((q / 4) * m.layers);
      const s = simulate(key, layer, 0);
      const cellDiv = document.createElement('div');
      cellDiv.className = 'map';
      cellDiv.dataset.model = key; cellDiv.dataset.layer = layer; cellDiv.dataset.artifacts = s.artifacts.size;
      cellDiv.innerHTML = `<h4>layer ${layer}</h4><svg viewBox="0 0 14 14" role="img" aria-label="${m.name}, layer ${layer}: ${s.artifacts.size ? s.artifacts.size + ' artifact tokens' : 'no artifact tokens'}"></svg><div class="note${s.artifacts.size ? ' has' : ''}">${s.artifacts.size ? `${s.artifacts.size} above 150` : 'none above 150'}</div>`;
      grid.appendChild(cellDiv);
      drawMap(cellDiv.querySelector('svg'), (i) => normColor(s.patch[i]), { outline: true });
    }
  }
}

// =============== Figure 5: the sequence with four registers ===============
function drawSequence(W) {
  const svg = document.getElementById('sequence-svg');
  clear(svg);
  const slots = ['CLS', 'R1', 'R2', 'R3', 'R4', 'p1', 'p2', '…', 'p196'];
  const kind = (s) => s === 'CLS' ? 'cls' : s[0] === 'R' ? 'reg' : s === '…' ? 'dots' : 'pat';
  const g = W < 480 ? 4 : 6;
  const bw = Math.min(54, (W - (slots.length - 1) * g) / slots.length);
  const X = (k) => k * (bw + g);
  const total = X(slots.length - 1) + bw;
  const inY = 30, bh = 28, blkY = inY + bh + 22, blkH = 40, outY = blkY + blkH + 22;
  const stroke = { cls: COL.cls, reg: COL.reg, pat: COL.rule2 };
  const bracket = (x1, x2, y, up, label, cls = 'label muted') => {
    const d = up ? 4 : -4;
    el('path', { d: `M${x1} ${y + d}V${y}H${x2}V${y + d}`, fill: 'none', stroke: COL.ink3, 'stroke-width': 1 }, svg);
    txt(svg, (x1 + x2) / 2, up ? y - 6 : y + 15, label, cls, { 'text-anchor': 'middle' });
  };
  bracket(X(0), X(0) + bw, inY - 8, true, '[CLS]');
  bracket(X(1), X(4) + bw, inY - 8, true, '4 registers');
  bracket(X(5), X(8) + bw, inY - 8, true, '196 patch tokens');

  const row = (y, out) => slots.forEach((s, k) => {
    const kd = kind(s);
    if (kd === 'dots') { txt(svg, X(k) + bw / 2, y + bh / 2 + 4, '…', 'label muted', { 'text-anchor': 'middle' }); return; }
    const discarded = out && kd === 'reg';
    el('rect', { x: X(k) + 0.5, y: y + 0.5, width: bw - 1, height: bh - 1, fill: 'none', stroke: discarded ? COL.ink3 : stroke[kd], 'stroke-width': 1.25, 'stroke-dasharray': discarded ? '3 3' : 'none', 'data-kind': kd, class: out ? 'out' : 'in' }, svg);
    txt(svg, X(k) + bw / 2, y + bh / 2 + 4, s, 'label', { 'text-anchor': 'middle', style: `font-size:${bw < 34 ? 11.5 : 12.5}px;fill:${discarded ? COL.ink3 : COL.ink}` });
    el('line', { x1: X(k) + bw / 2, x2: X(k) + bw / 2, y1: out ? blkY + blkH : y + bh, y2: out ? y : blkY, stroke: COL.ink3, 'stroke-width': 1 }, svg);
  });
  row(inY, false);
  el('rect', { x: 0.5, y: blkY + 0.5, width: total - 1, height: blkH - 1, fill: 'none', stroke: COL.ink2, 'stroke-width': 1 }, svg);
  txt(svg, total / 2, blkY + blkH / 2 + 4.5, W < 480 ? '24 transformer blocks' : '24 transformer blocks, unchanged', 'label', { 'text-anchor': 'middle' });
  row(outY, true);
  const by = outY + bh + 8;
  bracket(X(0), X(0) + bw, by, false, 'kept');
  bracket(X(1), X(4) + bw, by, false, 'discarded');
  bracket(X(5), X(8) + bw, by, false, 'kept');
  const H = by + 22;
  svg.setAttribute('width', W); svg.setAttribute('height', H); svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
}
measured(document.getElementById('sequence-graphic'), drawSequence);

// =============== Figure 6: registers absorb the high norms ===============
const ABSORB = [0, 1, 4].map((n) => ({ n, s: simulate('L', 24, n) }));
function drawAbsorb(W) {
  const svg = document.getElementById('absorb-svg');
  clear(svg);
  const labelW = Math.min(100, Math.round(W * 0.27)), R = 10, panelH = 108, top = 4;
  const xs = (x) => labelW + (Math.min(x, NORM_MAX) / NORM_MAX) * (W - labelW - R);
  const bottom = top + ABSORB.length * panelH;
  el('line', { x1: xs(CUTOFF), x2: xs(CUTOFF), y1: top + 20, y2: bottom, stroke: COL.rule2, 'stroke-width': 1, 'stroke-dasharray': '3 3' }, svg);
  ABSORB.forEach(({ n, s }, k) => {
    const y0 = top + k * panelH;
    const g = el('g', { class: 'panel', 'data-registers': n }, svg);
    txt(g, 0, y0 + 13, n === 0 ? 'No registers' : n === 1 ? '1 register' : `${n} registers`, 'label', { style: 'font-weight:600' });
    const yC = y0 + 34, yR = y0 + 54, yP = y0 + 72, band = 26;
    if (k > 0) el('line', { x1: 0, x2: W - R, y1: y0 - 2, y2: y0 - 2, stroke: COL.rule, 'stroke-width': 1 }, g);
    txt(g, 0, yC + 4, '[CLS]', 'label muted');
    txt(g, 0, yR + 4, 'registers', 'label muted');
    txt(g, 0, yP + band / 2 + 4, 'patch tokens', 'label muted');
    el('circle', { cx: xs(s.cls), cy: yC, r: 3.4, fill: COL.cls, class: 'dot cls' }, g);
    if (!s.regs.length) txt(g, labelW, yR + 4, 'none', 'label muted', { style: `fill:${COL.ink3}` });
    s.regs.forEach((x) => el('circle', { cx: xs(x), cy: yR, r: 3.4, fill: COL.reg, class: 'dot reg' }, g));
    const art = [];
    s.patch.forEach((x, i) => {
      const isArt = s.artifacts.has(i);
      if (isArt) art.push(x);
      el('circle', { cx: xs(x), cy: yP + 3 + rand(i * 7 + 3) * (band - 6), r: isArt ? 3 : 2.1, fill: isArt ? COL.art : COL.patch, 'fill-opacity': isArt ? 1 : 0.55, class: isArt ? 'dot art' : 'dot pat' }, g);
    });
    if (art.length) txt(g, xs(Math.min(...art)) - 8, yP + band / 2 + 4, `${art.length} artifacts`, 'annot', { 'text-anchor': 'end' });
    if (n === 1) txt(g, xs(s.regs[0]) - 8, yR + 4, 'the high norm moves here', 'annot', { 'text-anchor': 'end' });
  });
  drawAxis(svg, xs, bottom + 4, labelW, W - R, 'output token norm (simulated)');
  const H = bottom + 42;
  svg.setAttribute('width', W); svg.setAttribute('height', H); svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
}
measured(document.querySelector('#fig-absorb .graphic'), drawAbsorb);
{
  const [a0, a1, a4] = ABSORB.map((o) => o.s);
  const maxPatch = Math.max(...a1.patch, ...a4.patch);
  const regs = [...a1.regs, ...a4.regs];
  const t = document.getElementById('absorb-takeaway');
  t.textContent = `Without registers, ${a0.artifacts.size} patch tokens cross the cutoff. With one register or four, none do: the highest patch norm is ${fmt(maxPatch)}, and the registers sit between ${fmt(Math.min(...regs))} and ${fmt(Math.max(...regs))}.`;
  Object.assign(t.dataset, { artifacts0: a0.artifacts.size, maxPatch });
}

// =============== top bar: current section and reading progress ===============
{
  const links = [...document.querySelectorAll('.topbar nav a')];
  const sections = links.map((a) => document.querySelector(a.getAttribute('href')));
  const bar = document.getElementById('progress');
  let ticking = false;
  const update = () => {
    ticking = false;
    const h = document.documentElement;
    const p = h.scrollHeight > h.clientHeight ? h.scrollTop / (h.scrollHeight - h.clientHeight) : 0;
    bar.style.width = `${Math.max(0, Math.min(1, p)) * 100}%`;
    let cur = -1;
    sections.forEach((s, i) => { if (s && s.getBoundingClientRect().top < 120) cur = i; });
    links.forEach((a, i) => { if (i === cur) a.setAttribute('aria-current', 'true'); else a.removeAttribute('aria-current'); });
  };
  window.addEventListener('scroll', () => { if (!ticking) { ticking = true; requestAnimationFrame(update); } }, { passive: true });
  update();
}

// expose the simulation for the tests
window.__registerSim = { simulate, attention, scene, CANDIDATES, CUTOFF, N };
})();
