/* Quantization-aware training explainer.
   No dependencies. Every chart is drawn as SVG at the pixel width of its container and
   redrawn when that width changes, so chart text keeps its real size on a phone.
   window.QAT exposes the math and the lab state for the tests in test/. */
(function () {
'use strict';

/* ---------- small toolkit ---------- */
const NS = 'http://www.w3.org/2000/svg';
const $ = (s) => document.querySelector(s);
function E(parent, tag, attrs, text) {
  const n = document.createElementNS(NS, tag);
  if (attrs) for (const k in attrs) if (attrs[k] != null) n.setAttribute(k, attrs[k]);
  if (text != null) n.textContent = text;
  if (parent) parent.appendChild(n);
  return n;
}
function lin(d0, d1, r0, r1) {
  const f = (v) => r0 + (v - d0) / (d1 - d0) * (r1 - r0);
  f.invert = (p) => d0 + (p - r0) / (r1 - r0) * (d1 - d0);
  return f;
}
function logScale(d0, d1, r0, r1) {
  const a = Math.log10(d0), b = Math.log10(d1);
  return (v) => r0 + (Math.log10(v) - a) / (b - a) * (r1 - r0);
}
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const MINUS = '−';
const num = (v, d) => (v < 0 ? MINUS : '') + Math.abs(v).toFixed(d);
const signed = (v, d = 3) => (v >= 0 ? '+' : MINUS) + Math.abs(v).toFixed(d);
const SUP = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
const pow10 = (e) => '10' + String(e).split('').map((c) => SUP[c]).join('');
function pathD(pts) { return pts.map((p, i) => (i ? 'L' : 'M') + p[0].toFixed(1) + ',' + p[1].toFixed(1)).join(''); }
function mulberry(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
function gauss(r) { let u = 0, v = 0; while (u === 0) u = r(); while (v === 0) v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }

/** Replace host's contents with an SVG as wide as the host, h pixels tall. */
function sized(host, h, label) {
  host.textContent = '';
  const w = Math.max(80, Math.floor(host.clientWidth));
  const svg = E(host, 'svg', { width: w, height: h, viewBox: `0 0 ${w} ${h}`, role: 'img', 'aria-label': label });
  return { svg, w };
}
/** Keep a text element inside [lo, hi] horizontally by shifting it. */
function fitX(t, lo, hi) {
  const b = t.getBBox();
  if (b.x < lo) t.setAttribute('x', +t.getAttribute('x') + (lo - b.x));
  else if (b.x + b.width > hi) t.setAttribute('x', +t.getAttribute('x') - (b.x + b.width - hi));
}
/** Spread label y positions so they are at least gap apart, keeping them within [lo, hi]. */
function spread(items, gap, lo, hi) {
  items.sort((a, b) => a.y - b.y);
  for (let i = 1; i < items.length; i++) if (items[i].y - items[i - 1].y < gap) items[i].y = items[i - 1].y + gap;
  const over = items.length ? items[items.length - 1].y - hi : 0;
  if (over > 0) { items[items.length - 1].y -= over; for (let i = items.length - 2; i >= 0; i--) if (items[i + 1].y - items[i].y < gap) items[i].y = items[i + 1].y - gap; }
  if (items.length && items[0].y < lo) { const d = lo - items[0].y; items.forEach((it) => { it.y += d; }); }
  return items;
}
function arrowDefs(svg, id) {
  const m = E(E(svg, 'defs'), 'marker', { id, viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' });
  E(m, 'path', { d: 'M0,1 L9,5 L0,9 z', fill: 'var(--ink-2)' });
}

/** Redraw fn whenever host's width changes (and once when web fonts arrive). */
const redraws = [];
function responsive(host, fn) {
  let last = -1;
  const run = (force) => { const w = host.clientWidth; if (force === true || w !== last) { last = w; fn(); } };
  redraws.push(() => run(true));
  if ('ResizeObserver' in window) new ResizeObserver(() => run()).observe(host);
  else window.addEventListener('resize', () => run());
  run(true);
}
if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => redraws.forEach((f) => f()));

/* ---------- the quantizer used everywhere on the page ----------
   Symmetric, per tensor: q = clamp(round(x/s), -qmax, qmax), s = max|x| / qmax, qmax = 2^(b-1) - 1.
   The most negative signed code (-qmax-1) is deliberately unused so the grid is symmetric about zero. */
function quantizeArray(arr, bits) {
  const qmax = (1 << (bits - 1)) - 1;
  let mx = 0; for (const v of arr) mx = Math.max(mx, Math.abs(v));
  const s = (mx || 1e-8) / qmax; const out = [], mask = [];
  for (const v of arr) { const q = Math.round(v / s); const c = clamp(q, -qmax, qmax); mask.push(c === q ? 1 : 0); out.push(s * c); }
  return { out, mask, s, qmax };
}
/** One value through a quantizer with clipping range alpha. */
function quantizeOne(x, bits, alpha) {
  const qmax = (1 << (bits - 1)) - 1, s = alpha / qmax;
  const q = clamp(Math.round(x / s), -qmax, qmax);
  return { q, s, qmax, xh: s * q, clipped: Math.abs(x) > alpha };
}
/** Straight-through estimate of dx̂/dx used on this page: 1 inside the clipping range, 0 outside. */
const steGrad = (x, alpha) => (Math.abs(x) <= alpha ? 1 : 0);
window.QAT = { quantizeArray, quantizeOne, steGrad };

/* ---------- top bar: current section and reading progress ---------- */
(function () {
  const links = [...document.querySelectorAll('.topbar nav a')];
  const secs = links.map((a) => document.getElementById(a.getAttribute('href').slice(1)));
  const bar = $('#progress');
  let ticking = false;
  function update() {
    ticking = false;
    const doc = document.documentElement;
    const max = doc.scrollHeight - window.innerHeight;
    bar.style.width = (max > 0 ? clamp(window.scrollY / max, 0, 1) * 100 : 0) + '%';
    const mark = window.innerHeight * 0.35;
    let cur = -1;
    secs.forEach((s, i) => { if (s && s.getBoundingClientRect().top <= mark) cur = i; });
    links.forEach((a, i) => { if (i === cur) a.setAttribute('aria-current', 'true'); else a.removeAttribute('aria-current'); });
    if (cur >= 0 && links[cur].parentElement.scrollWidth > links[cur].parentElement.clientWidth) {
      const nav = links[cur].parentElement, a = links[cur];
      if (a.offsetLeft < nav.scrollLeft || a.offsetLeft + a.offsetWidth > nav.scrollLeft + nav.clientWidth) nav.scrollLeft = a.offsetLeft - 16;
    }
  }
  window.addEventListener('scroll', () => { if (!ticking) { ticking = true; requestAnimationFrame(update); } }, { passive: true });
  window.addEventListener('resize', update);
  update();
})();

/* ---------- Figure 1: weight memory for five model sizes ---------- */
(function () {
  const host = $('#memChart');
  const MODELS = [1, 8, 27, 70, 405];
  const FORMATS = [
    { k: 'int4', bytes: 0.5, fill: 'var(--quant)' },
    { k: 'int8', bytes: 1, fill: 'var(--c-grey)' },
    { k: 'bf16', bytes: 2, fill: 'var(--fp)' },
  ];
  const GPUS = [{ v: 16, n: '16 GB laptop GPU' }, { v: 24, n: '24 GB RTX 4090' }, { v: 80, n: '80 GB H100' }];
  const fmtGB = (v) => (v < 100 && v % 1 ? v.toFixed(1) : String(Math.round(v)));
  function draw() {
    const refRow = 16, top = GPUS.length * refRow + 10, head = 18, rowH = 36;
    const plotTop = top + head, plotBot = plotTop + MODELS.length * rowH;
    const H = plotBot + 44;
    const { svg, w } = sized(host, H, 'Dot plot of weight memory in bf16, int8 and int4 for 1B, 8B, 27B, 70B and 405B parameter models, on a log scale, with reference lines at 16, 24 and 80 GB.');
    const L = 42, R = w - 14;
    const x = logScale(0.3, 1000, L, R);
    // reference lines: smallest value on the top row, so no line crosses another line's label
    GPUS.forEach((g, i) => {
      const ty = 12 + i * refRow, gx = x(g.v);
      E(svg, 'line', { x1: gx, x2: gx, y1: ty - 9, y2: plotBot, stroke: 'var(--ink-3)', 'stroke-width': 1, 'stroke-dasharray': '2 3' });
      const t = E(svg, 'text', { x: gx + 4, y: ty, class: 'annot' }, g.n);
      if (gx + 4 + t.getComputedTextLength() > w) { t.setAttribute('x', gx - 4); t.setAttribute('text-anchor', 'end'); }
    });
    MODELS.forEach((p, i) => {
      const cy = plotTop + i * rowH + rowH / 2 + 4;
      E(svg, 'text', { x: 0, y: cy + 4, class: 'label' }, p + 'B');
      const xs = FORMATS.map((f) => x(p * f.bytes));
      E(svg, 'line', { x1: xs[0], x2: xs[2], y1: cy, y2: cy, stroke: 'var(--rule-2)', 'stroke-width': 1 });
      FORMATS.forEach((f, j) => {
        E(svg, 'circle', { cx: xs[j], cy, r: 4.5, fill: f.fill });
        E(svg, 'text', { x: xs[j], y: cy - 9, 'text-anchor': 'middle', class: 'val halo' }, fmtGB(p * f.bytes));
        if (i === 0) E(svg, 'text', { x: xs[j], y: cy - 24, 'text-anchor': 'middle', class: 'label muted' }, f.k);
      });
    });
    const ay = plotBot + 4;
    E(svg, 'line', { x1: L, x2: R, y1: ay, y2: ay, class: 'axis-line', stroke: 'var(--rule-2)' });
    [1, 10, 100, 1000].forEach((v) => {
      E(svg, 'line', { x1: x(v), x2: x(v), y1: ay, y2: ay + 4, stroke: 'var(--rule-2)' });
      E(svg, 'text', { x: x(v), y: ay + 17, 'text-anchor': v === 1000 ? 'end' : 'middle', class: 'tick' }, v === 1000 ? '1,000' : String(v));
    });
    E(svg, 'text', { x: L, y: ay + 34, class: 'axis-title' }, 'GB of weights (log scale)');
  }
  responsive(host, draw);
})();

/* ---------- Figure 2: one number on the grid, its error, and the choice of α ---------- */
(function () {
  const SIG = 0.35, XMAX = 1.6;
  const st = { bits: 4, alpha: 1, x: 0.437 };
  const el = {
    bits: $('#nlBits'), bitsV: $('#nlBitsV'), a: $('#nlAlpha'), aV: $('#nlAlphaV'), xin: $('#nlXin'),
    X: $('#nlX'), Q: $('#nlQ'), Xh: $('#nlXh'), Err: $('#nlErr'), Kind: $('#nlKind'), L: $('#nlLevels'), S: $('#nlS'),
  };
  const hostA = $('#numLine'), hostB = $('#errLine'), hostC = $('#alphaPlot');
  const density = (v) => Math.exp(-v * v / (2 * SIG * SIG));
  const errAt = (v) => Math.abs(v - quantizeOne(v, st.bits, st.alpha).xh);
  let xScale = null;

  /** Mean squared error of the quantizer over N(0, SIG²), split into the part from inside and outside [-α, α]. */
  function mse(alpha, bits) {
    const qmax = (1 << (bits - 1)) - 1, s = alpha / qmax, lo = -6 * SIG, n = 1600, dx = 12 * SIG / n;
    let r = 0, c = 0;
    for (let i = 0; i < n; i++) {
      const v = lo + (i + 0.5) * dx, p = density(v) / (SIG * Math.sqrt(2 * Math.PI)) * dx;
      const e = v - s * clamp(Math.round(v / s), -qmax, qmax);
      if (Math.abs(v) > alpha) c += e * e * p; else r += e * e * p;
    }
    return { r, c, t: r + c };
  }
  let curveCache = { bits: null, pts: null, best: null };
  function curves() {
    if (curveCache.bits === st.bits) return curveCache;
    const pts = [];
    for (let a = 0.2; a <= 1.5001; a += 0.01) pts.push({ a, ...mse(a, st.bits) });
    const best = pts.reduce((m, p) => (p.t < m.t ? p : m));
    curveCache = { bits: st.bits, pts, best };
    return curveCache;
  }
  window.QAT.quantMse = mse;

  function drawA() {
    const H = 150, ax = 70;
    const { svg, w } = sized(hostA, H, 'Number line showing the quantization levels, the clipping range, a value x and the level it rounds to.');
    const pad = 14, x = lin(-XMAX, XMAX, pad, w - pad); xScale = x;
    const { q, s, xh, qmax } = quantizeOne(st.x, st.bits, st.alpha);
    // weight distribution, context in grey
    const dpts = []; for (let i = 0; i <= 160; i++) { const v = -XMAX + 2 * XMAX * i / 160; dpts.push([x(v), ax - 3 - density(v) * 48]); }
    E(svg, 'path', { d: pathD(dpts), fill: 'none', stroke: 'var(--c-grey)', 'stroke-width': 1 });
    // axis: solid inside the clipping range, faint outside it
    E(svg, 'line', { x1: x(-XMAX), x2: x(XMAX), y1: ax, y2: ax, stroke: 'var(--rule-2)' });
    E(svg, 'line', { x1: x(-st.alpha), x2: x(st.alpha), y1: ax, y2: ax, stroke: 'var(--ink-2)' });
    // levels
    const dense = 2 * qmax + 1 > 40;
    for (let k = -qmax; k <= qmax; k++) E(svg, 'line', { x1: x(k * s), x2: x(k * s), y1: ax - 6, y2: ax + 6, stroke: 'var(--ink-2)', 'stroke-width': dense ? 0.5 : 1 });
    // clipping range markers and labels
    [-1, 1].forEach((sg) => {
      const px = x(sg * st.alpha);
      E(svg, 'line', { x1: px, x2: px, y1: ax - 14, y2: ax + 14, stroke: 'var(--ink)', 'stroke-width': 1 });
      E(svg, 'text', { x: px, y: ax + 28, 'text-anchor': 'middle', class: 'label' }, (sg < 0 ? MINUS : '+') + 'α');
      const outW = sg < 0 ? px - pad : w - pad - px;
      if (outW > 56) E(svg, 'text', { x: sg < 0 ? (pad + px) / 2 : (px + w - pad) / 2, y: ax + 28, 'text-anchor': 'middle', class: 'tick' }, 'clipped');
    });
    // value ticks
    [-1.5, -1, -0.5, 0, 0.5, 1, 1.5].forEach((t) => E(svg, 'text', { x: x(t), y: ax + 66, 'text-anchor': 'middle', class: 'tick' }, num(t, t % 1 ? 1 : 0)));
    // error, x-hat, x
    E(svg, 'line', { x1: x(st.x), x2: x(xh), y1: ax, y2: ax, stroke: 'var(--err)', 'stroke-width': 3 });
    E(svg, 'rect', { x: x(xh) - 5, y: ax - 5, width: 10, height: 10, fill: 'var(--quant)', stroke: 'var(--paper)', 'stroke-width': 1.5 });
    const tq = E(svg, 'text', { x: x(xh), y: ax + 46, 'text-anchor': 'middle', class: 'label halo', fill: 'var(--quant)', style: 'fill: var(--quant)' }, 'x̂ = ' + num(xh, 3));
    fitX(tq, 0, w);
    const dot = E(svg, 'circle', { cx: x(st.x), cy: ax, r: 7.5, fill: 'var(--fp)', stroke: 'var(--paper)', 'stroke-width': 2, class: 'drag', tabindex: -1, 'aria-hidden': 'true' });
    const tx = E(svg, 'text', { x: x(st.x), y: ax - 15, 'text-anchor': 'middle', class: 'label halo', style: 'fill: var(--fp)' }, 'x = ' + num(st.x, 3));
    fitX(tx, 0, w);
    dot.addEventListener('pointerdown', startDrag);
  }

  function drawB() {
    const H = 104, top = 22, base = H - 16;
    const { svg, w } = sized(hostB, H, 'Error of the quantizer at every x: a sawtooth no higher than s/2 inside the clipping range, growing linearly outside it.');
    const pad = 14, x = lin(-XMAX, XMAX, pad, w - pad), YMAX = 0.5, y = lin(0, YMAX, base, top);
    const { s, qmax } = quantizeOne(0, st.bits, st.alpha);
    // exact breakpoints: zeros on every level, peaks halfway between, straight lines outside the range
    const xs = [-XMAX, -st.alpha];
    for (let k = -qmax; k <= qmax; k++) { xs.push(k * s); if (k < qmax) xs.push((k + 0.5) * s); }
    xs.push(st.alpha, XMAX);
    const pts = xs.map((v) => [x(v), y(Math.min(errAt(v), YMAX + 0.05))]);
    const cid = 'errclip';
    E(E(E(svg, 'defs'), 'clipPath', { id: cid }), 'rect', { x: 0, y: top, width: w, height: base - top + 2 });
    E(svg, 'line', { x1: pad, x2: w - pad, y1: base, y2: base, stroke: 'var(--rule-2)' });
    E(svg, 'line', { x1: x(-st.alpha), x2: x(st.alpha), y1: y(s / 2), y2: y(s / 2), stroke: 'var(--ink-3)', 'stroke-dasharray': '2 3' });
    E(svg, 'path', { d: pathD(pts), fill: 'none', stroke: 'var(--err)', 'stroke-width': 1.25, 'clip-path': `url(#${cid})`, 'stroke-linejoin': 'round' });
    E(svg, 'text', { x: pad, y: 10, class: 'tick' }, '0.5');
    E(svg, 'line', { x1: pad, x2: w - pad, y1: top, y2: top, stroke: 'var(--rule)' });
    const tl = E(svg, 'text', { x: x(0), y: Math.min(y(s / 2) - 6, base - 12), 'text-anchor': 'middle', class: 'annot halo' }, 'rounding error ≤ s/2 = ' + (s / 2).toFixed(s / 2 < 0.01 ? 4 : 3));
    fitX(tl, 0, w);
    [-1, 1].forEach((sg) => {
      const a = x(sg * st.alpha), b = sg < 0 ? pad : w - pad;
      if (Math.abs(b - a) > 96) E(svg, 'text', { x: (a + b) / 2, y: top + 14, 'text-anchor': 'middle', class: 'annot halo' }, 'clipping error');
    });
    const e = errAt(st.x);
    E(svg, 'circle', { cx: x(st.x), cy: y(Math.min(e, YMAX)), r: 3.5, fill: 'var(--err)', stroke: 'var(--paper)', 'stroke-width': 1.5 });
  }

  function drawC() {
    const H = 196, T = 26, B = H - 40;
    const { svg, w } = sized(hostC, H, 'Mean squared quantization error of normally distributed weights against the clipping range alpha, split into rounding and clipping parts, with the current alpha and the best alpha marked.');
    const L = 40, R = w - 62;
    const x = lin(0.2, 1.5, L, R), YLO = 3e-6, YHI = 0.3, y = logScale(YLO, YHI, B, T);
    const { pts, best } = curves();
    // curves leave the plot through its floor rather than running along it
    const Y = (v) => y(Math.max(v, 1e-12));
    const cid = 'mseclip';
    E(E(E(svg, 'defs'), 'clipPath', { id: cid }), 'rect', { x: L, y: T - 6, width: R - L, height: B - T + 6 });
    for (let e = -5; e <= -1; e++) {
      E(svg, 'line', { x1: L, x2: R, y1: y(10 ** e), y2: y(10 ** e), stroke: 'var(--rule)' });
      E(svg, 'text', { x: L - 6, y: y(10 ** e) + 4, 'text-anchor': 'end', class: 'tick' }, pow10(e));
    }
    E(svg, 'text', { x: 0, y: 12, class: 'axis-title' }, 'mean squared error (log scale)');
    E(svg, 'line', { x1: L, x2: R, y1: B, y2: B, stroke: 'var(--rule-2)' });
    [0.25, 0.5, 0.75, 1, 1.25, 1.5].forEach((t) => E(svg, 'text', { x: x(t), y: B + 16, 'text-anchor': 'middle', class: 'tick' }, t.toFixed(2)));
    E(svg, 'text', { x: R, y: B + 33, 'text-anchor': 'end', class: 'axis-title' }, 'clipping range α');
    const clip = `url(#${cid})`;
    E(svg, 'path', { d: pathD(pts.map((p) => [x(p.a), Y(p.r)])), fill: 'none', stroke: 'var(--ink-3)', 'stroke-width': 1, 'clip-path': clip });
    E(svg, 'path', { d: pathD(pts.map((p) => [x(p.a), Y(p.c)])), fill: 'none', stroke: 'var(--ink-3)', 'stroke-width': 1, 'stroke-dasharray': '4 3', 'clip-path': clip });
    E(svg, 'path', { d: pathD(pts.map((p) => [x(p.a), Y(p.t)])), fill: 'none', stroke: 'var(--err)', 'stroke-width': 1.75, 'clip-path': clip });
    // direct labels at the ends where each part dominates; above the line when it runs near the floor
    const last = pts[pts.length - 1], first = pts[0];
    const ends = [{ y: Y(last.t) + 4, t: 'total', cls: 'label', fill: 'var(--err)' }];
    if (Y(last.r) < B) ends.push({ y: Y(last.r) + 4, t: 'rounding', cls: 'label muted' });
    if (Y(last.c) < B) ends.push({ y: Y(last.c) + 4, t: 'clipping', cls: 'label muted' });
    spread(ends, 13, T, B + 4).forEach((l) => E(svg, 'text', { x: R + 6, y: l.y, class: l.cls, style: l.fill ? `fill: ${l.fill}` : null }, l.t));
    if (Y(last.c) >= B) E(svg, 'text', { x: L + 4, y: Y(first.c) + 16 > B - 4 ? Y(first.c) - 6 : Y(first.c) + 16, class: 'label muted halo' }, 'clipping');
    // best alpha: label below the minimum, or above it when the minimum sits in the lower half
    const by = Y(best.t);
    E(svg, 'line', { x1: x(best.a), x2: x(best.a), y1: by + 4, y2: B, stroke: 'var(--err)', 'stroke-width': 1, 'stroke-dasharray': '1 2' });
    const lowHalf = by > (T + B) / 2;
    const tb = E(svg, 'text', { x: x(best.a), y: lowHalf ? by - 12 : by + 20, 'text-anchor': lowHalf ? 'end' : 'middle', class: 'annot halo' }, 'lowest at α = ' + best.a.toFixed(2));
    fitX(tb, L, w);
    // current alpha
    const cur = mse(st.alpha, st.bits);
    E(svg, 'line', { x1: x(st.alpha), x2: x(st.alpha), y1: T - 4, y2: B, stroke: 'var(--ink)', 'stroke-width': 1 });
    E(svg, 'circle', { cx: x(st.alpha), cy: Y(cur.t), r: 4, fill: 'var(--err)', stroke: 'var(--paper)', 'stroke-width': 1.5 });
    const tc = E(svg, 'text', { x: x(st.alpha) + 5, y: T + 6, class: 'label halo' }, 'your α');
    if (x(st.alpha) + 5 + tc.getComputedTextLength() > w) { tc.setAttribute('x', x(st.alpha) - 5); tc.setAttribute('text-anchor', 'end'); }
  }

  function readout() {
    const { q, s, xh, qmax, clipped } = quantizeOne(st.x, st.bits, st.alpha);
    el.X.textContent = num(st.x, 3); el.Q.textContent = num(q, 0); el.Xh.textContent = num(xh, 3);
    el.Err.textContent = signed(st.x - xh); el.Kind.textContent = clipped ? 'clipping' : 'rounding';
    el.L.textContent = 2 * qmax + 1; el.S.textContent = s.toFixed(s < 0.01 ? 4 : 3);
    el.xin.value = st.x;
  }
  function drawAll() { drawA(); drawB(); drawC(); readout(); }
  function drawX() { drawA(); drawB(); readout(); }

  // dragging the blue dot
  let dragging = false;
  function moveTo(clientX) {
    if (!xScale) return;
    const r = hostA.getBoundingClientRect();
    st.x = clamp(+xScale.invert(clientX - r.left).toFixed(3), -XMAX, XMAX);
    drawX();
  }
  function startDrag(e) { dragging = true; e.preventDefault(); moveTo(e.clientX); }
  window.addEventListener('pointermove', (e) => { if (dragging) moveTo(e.clientX); });
  window.addEventListener('pointerup', () => { dragging = false; });
  window.addEventListener('pointercancel', () => { dragging = false; });

  el.bits.addEventListener('input', () => { st.bits = +el.bits.value; el.bitsV.textContent = st.bits; drawAll(); });
  el.a.addEventListener('input', () => { st.alpha = +el.a.value; el.aV.textContent = st.alpha.toFixed(2); drawAll(); });
  el.xin.addEventListener('input', () => { st.x = +el.xin.value; drawX(); });
  responsive(hostA, drawAll);
})();

/* ---------- Figure 3: per-tensor versus per-row scales, as small multiples ---------- */
(function () {
  const n = 12, BITS = 4, OUT = [3, 7], OUTV = 2.6;
  const r = mulberry(11);
  const base = []; for (let i = 0; i < n; i++) { base.push([]); for (let j = 0; j < n; j++) base[i].push(gauss(r) * 0.3); }
  function compute(outlier, gran) {
    const W = base.map((row) => row.slice()); if (outlier) W[OUT[0]][OUT[1]] = OUTV;
    const qmax = (1 << (BITS - 1)) - 1; const scales = [];
    const mxAll = Math.max(...W.flat().map(Math.abs));
    for (let i = 0; i < n; i++) scales.push((gran === 'tensor' ? mxAll : Math.max(...W[i].map(Math.abs))) / qmax);
    const E_ = W.map((row, i) => row.map((v) => { const s = scales[i]; return Math.abs(v - s * clamp(Math.round(v / s), -qmax, qmax)); }));
    const num2 = E_.flat().reduce((a, v) => a + v * v, 0), den = W.flat().reduce((a, v) => a + v * v, 0);
    return { E: E_, rel: Math.sqrt(num2 / den), maxS: Math.max(...scales) };
  }
  const panels = [
    { host: $('#hmA'), rel: $('#hmRelA'), outlier: false, gran: 'tensor' },
    { host: $('#hmB'), rel: $('#hmRelB'), outlier: true, gran: 'tensor' },
    { host: $('#hmC'), rel: $('#hmRelC'), outlier: true, gran: 'row' },
  ];
  panels.forEach((p) => { p.res = compute(p.outlier, p.gran); });
  const emax = Math.max(...panels.flatMap((p) => p.res.E.flat()));
  const paper = [252, 251, 247], red = [184, 67, 58];
  const color = (e) => { const t = Math.sqrt(clamp(e / emax, 0, 1)); return 'rgb(' + paper.map((c, i) => Math.round(c + (red[i] - c) * t)).join(',') + ')'; };
  $('#hmMax').textContent = emax.toFixed(2);
  $('#hmRamp').style.background = 'linear-gradient(to right,' + [0, 0.25, 0.5, 0.75, 1].map((t) => color(t * t * emax)).join(',') + ')';
  window.QAT.heat = panels.map((p) => ({ outlier: p.outlier, gran: p.gran, rel: p.res.rel }));
  function draw() {
    panels.forEach((p) => {
      const w0 = Math.max(80, Math.floor(p.host.clientWidth));
      const { svg, w } = sized(p.host, w0, `Error map, ${p.gran === 'tensor' ? 'per-tensor scale' : 'per-row scales'}, ${p.outlier ? 'with' : 'without'} an outlier; relative error ${(p.res.rel * 100).toFixed(1)} percent.`);
      const c = w / n;
      p.res.E.forEach((row, i) => row.forEach((e, j) => E(svg, 'rect', { x: j * c + 0.5, y: i * c + 0.5, width: c - 1, height: c - 1, fill: color(e) })));
      if (p.outlier) E(svg, 'rect', { x: OUT[1] * c - 1, y: OUT[0] * c - 1, width: c + 2, height: c + 2, fill: 'none', stroke: 'var(--ink)', 'stroke-width': 1.5 });
      p.rel.textContent = (p.res.rel * 100).toFixed(1) + '%';
    });
  }
  responsive($('#hmA'), draw);
})();

/* ---------- Figure 4: one QAT step, forward down the left, backward up the right ---------- */
(function () {
  const host = $('#loopSvg');
  const nodes = [
    { id: 'W', col: 0, row: 0, t: 'Master weight', f: [{ s: 'W, kept in fp32', c: 'var(--fp)' }] },
    { id: 'FQ', col: 0, row: 1, t: 'Fake-quantize', f: [{ s: 'q = clamp(round(W/s))', c: 'var(--ink-2)' }, { s: 'Ŵ = s · q', c: 'var(--quant)' }] },
    { id: 'Y', col: 0, row: 2, t: 'Layer output', f: [{ s: 'y = Ŵ x', c: 'var(--ink-2)' }] },
    { id: 'L', col: 0.5, row: 3, t: 'Loss', f: [{ s: 'L(y, target)', c: 'var(--ink-2)' }] },
    { id: 'G', col: 1, row: 2, t: 'Backpropagate', f: [{ s: '∂L/∂Ŵ', c: 'var(--ink-2)' }] },
    { id: 'STE', col: 1, row: 1, t: 'Straight-through', f: [{ s: '∂L/∂W = ∂L/∂Ŵ', c: 'var(--qat)' }, { s: 'inside ±α, else 0', c: 'var(--ink-2)', sans: true }] },
    { id: 'U', col: 1, row: 0, t: 'Optimizer step', f: [{ s: 'W ← W − η ∂L/∂W', c: 'var(--fp)' }] },
  ];
  function draw() {
    const { svg, w } = sized(host, 400, 'Diagram of one quantization-aware training step. Forward: master weight W, fake-quantize to Ŵ, layer output, loss. Backward: gradient with respect to Ŵ, straight-through estimator gives the gradient with respect to W, optimizer updates W.');
    arrowDefs(svg, 'ah-loop');
    const gap = Math.min(90, Math.max(36, w * 0.12)), colW = (w - gap) / 2;
    const cx = (col) => (col === 0 ? colW / 2 : col === 1 ? w - colW / 2 : w / 2);
    E(svg, 'text', { x: cx(0), y: 12, 'text-anchor': 'middle', class: 'axis-title' }, 'forward pass ↓');
    E(svg, 'text', { x: cx(1), y: 12, 'text-anchor': 'middle', class: 'axis-title' }, 'backward pass ↑');
    const box = {};
    let y = 44;
    for (let row = 0; row <= 3; row++) {
      let hMax = 0;
      nodes.filter((n) => n.row === row).forEach((n) => {
        const g = E(svg, 'g');
        E(g, 'text', { x: cx(n.col), y, 'text-anchor': 'middle', class: 'node-t' }, n.t);
        n.f.forEach((f, i) => E(g, 'text', { x: cx(n.col), y: y + 18 + i * 16, 'text-anchor': 'middle', class: f.sans ? 'node-n' : 'node-f', style: `fill: ${f.c}` }, f.s));
        const b = g.getBBox(); box[n.id] = { l: b.x, r: b.x + b.width, t: b.y, b: b.y + b.height, cx: cx(n.col), cy: b.y + b.height / 2 };
        hMax = Math.max(hMax, b.height);
      });
      y += hMax + 36;
    }
    const H = y - 20; svg.setAttribute('height', H); svg.setAttribute('viewBox', `0 0 ${w} ${H}`);
    const arrow = (d) => E(svg, 'path', { d, fill: 'none', stroke: 'var(--ink-2)', 'stroke-width': 1, 'marker-end': 'url(#ah-loop)' });
    const down = (a, b) => arrow(`M${box[a].cx},${box[a].b + 5} L${box[a].cx},${box[b].t - 5}`);
    const up = (a, b) => arrow(`M${box[a].cx},${box[a].t - 5} L${box[a].cx},${box[b].b + 5}`);
    down('W', 'FQ'); down('FQ', 'Y');
    arrow(`M${box.Y.cx},${box.Y.b + 5} L${box.Y.cx},${box.L.cy} L${box.L.l - 8},${box.L.cy}`);
    arrow(`M${box.L.r + 8},${box.L.cy} L${box.G.cx},${box.L.cy} L${box.G.cx},${box.G.b + 5}`);
    up('G', 'STE'); up('STE', 'U');
    const ty = Math.min(box.U.cy, box.W.cy);
    arrow(`M${box.U.l - 8},${ty} L${box.W.r + 8},${ty}`);
    if (box.U.l - box.W.r > 70) E(svg, 'text', { x: (box.U.l + box.W.r) / 2, y: ty - 6, 'text-anchor': 'middle', class: 'annot' }, 'new W');
  }
  responsive(host, draw);
})();

/* ---------- Figure 5: the staircase and its two derivatives ---------- */
(function () {
  const S = 1, A = 2;
  const q = (v) => S * Math.round(clamp(v, -A, A) / S);
  function frame(host, H, label, yd) {
    const { svg, w } = sized(host, H, label);
    const L = 24, R = w - 6, T = 8, B = H - 22;
    const x = lin(-3, 3, L, R), y = lin(yd[0], yd[1], B, T);
    E(svg, 'line', { x1: L, x2: R, y1: y(0), y2: y(0), stroke: 'var(--rule-2)' });
    [-2, 0, 2].forEach((t) => E(svg, 'text', { x: x(t), y: B + 16, 'text-anchor': 'middle', class: 'tick' }, num(t, 0)));
    return { svg, w, x, y, L, R, T, B };
  }
  function drawFwd() {
    const f = frame($('#steFwd'), 170, 'Staircase: the quantized value against x, clipped at plus and minus 2, beside the identity line.', [-3, 3]);
    const { svg, x, y, L, R } = f;
    E(svg, 'line', { x1: x(0), x2: x(0), y1: y(-3), y2: y(3), stroke: 'var(--rule-2)' });
    [-2, 2].forEach((t) => E(svg, 'text', { x: L - 5, y: y(t) + 4, 'text-anchor': 'end', class: 'tick' }, num(t, 0)));
    E(svg, 'line', { x1: x(-3), x2: x(3), y1: y(-3), y2: y(3), stroke: 'var(--c-grey)', 'stroke-dasharray': '3 3' });
    const pts = []; for (let v = -3; v <= 3.0001; v += 0.005) pts.push([x(v), y(q(v))]);
    E(svg, 'path', { d: pathD(pts), fill: 'none', stroke: 'var(--quant)', 'stroke-width': 1.75 });
    E(svg, 'text', { x: x(2.9), y: y(2.9) + 16, 'text-anchor': 'end', class: 'label muted halo' }, 'x̂ = x');
    E(svg, 'text', { x: x(-2.9), y: y(-2) - 7, class: 'label halo', style: 'fill: var(--quant)' }, 'x̂');
    E(svg, 'text', { x: x(0.62), y: y(0) + 16, class: 'annot halo' }, 'flat: slope 0');
  }
  function drawDeriv(host, kind) {
    const f = frame(host, 112, kind === 'true' ? 'The true derivative of the quantizer: zero everywhere except at the jumps, where it is undefined.' : 'The straight-through estimate: 1 for x between minus 2 and 2, 0 outside.', [-0.2, 1.35]);
    const { svg, x, y, L } = f;
    [0, 1].forEach((t) => E(svg, 'text', { x: L - 5, y: y(t) + 4, 'text-anchor': 'end', class: 'tick' }, String(t)));
    if (kind === 'true') {
      E(svg, 'line', { x1: x(-3), x2: x(3), y1: y(0), y2: y(0), stroke: 'var(--ink-2)', 'stroke-width': 1.75 });
      [-1.5, -0.5, 0.5, 1.5].forEach((k) => E(svg, 'line', { x1: x(k), x2: x(k), y1: y(0), y2: y(1.3), stroke: 'var(--ink-3)', 'stroke-dasharray': '1 3' }));
      E(svg, 'text', { x: x(0), y: y(0) - 8, 'text-anchor': 'middle', class: 'annot halo' }, '0 almost everywhere');
    } else {
      const c = 'var(--qat)';
      const seg = (a, b, v) => E(svg, 'line', { x1: x(a), x2: x(b), y1: y(v), y2: y(v), stroke: c, 'stroke-width': 1.75 });
      seg(-3, -A, 0); seg(-A, A, 1); seg(A, 3, 0);
      [-A, A].forEach((k) => E(svg, 'line', { x1: x(k), x2: x(k), y1: y(0), y2: y(1), stroke: c, 'stroke-dasharray': '1 2' }));
      E(svg, 'text', { x: x(0), y: y(1) - 7, 'text-anchor': 'middle', class: 'annot halo' }, '1 inside ±α, 0 outside');
    }
  }
  const hosts = [$('#steFwd'), $('#steTrue'), $('#steSte')];
  responsive(hosts[0].closest('.multiples'), () => { drawFwd(); drawDeriv(hosts[1], 'true'); drawDeriv(hosts[2], 'ste'); });
})();

/* ---------- Figure 6: a master weight drifting across rounding boundaries ---------- */
(function () {
  const host = $('#driftSvg');
  const STEPS = 100, W0 = 1.3, SEED = 8;
  const r = mulberry(SEED);
  const Wt = [W0]; for (let t = 0; t < STEPS; t++) Wt.push(Wt[t] - 0.03 + 0.06 * gauss(r));
  const Q = Wt.map((v) => clamp(Math.round(v), -2, 2));
  const flips = []; for (let t = 1; t < Q.length; t++) if (Q[t] !== Q[t - 1]) flips.push({ t, from: Q[t - 1], to: Q[t] });
  window.QAT.drift = { Wt, Q, flips };
  // the first level change that is later reversed, and the first move from 0 to -1
  const osc = flips.find((f, i) => flips[i + 1] && flips[i + 1].to === f.from);
  const cross = flips.find((f) => f.from === 0 && f.to === -1);
  function draw() {
    const H = 230;
    const { svg, w } = sized(host, H, `Line chart over ${STEPS} steps: the master weight W drifts from 1.3 down to about ${num(Wt[STEPS], 1)}, while the rounded value Ŵ steps from 1 to 0, back to 1 briefly, then to 0, −1 and −2.`);
    const L = 26, R = w - 8, T = 10, B = H - 36;
    const x = lin(0, STEPS, L, R), y = lin(-2.4, 1.7, B, T);
    [-1.5, -0.5, 0.5, 1.5].forEach((b) => E(svg, 'line', { x1: L, x2: R, y1: y(b), y2: y(b), stroke: 'var(--rule-2)', 'stroke-dasharray': '3 3' }));
    [-2, -1, 0, 1].forEach((k) => E(svg, 'text', { x: L - 6, y: y(k) + 4, 'text-anchor': 'end', class: 'tick' }, num(k, 0)));
    E(svg, 'line', { x1: L, x2: R, y1: B + 6, y2: B + 6, stroke: 'var(--rule-2)' });
    [0, 25, 50, 75, 100].forEach((t) => E(svg, 'text', { x: x(t), y: B + 21, 'text-anchor': t === 100 ? 'end' : 'middle', class: 'tick' }, String(t)));
    E(svg, 'text', { x: R, y: B + 35, 'text-anchor': 'end', class: 'axis-title' }, 'training step');
    // Ŵ as a step function, W as a line
    let d = `M${x(0)},${y(Q[0])}`; for (let t = 1; t <= STEPS; t++) d += `H${x(t)}V${y(Q[t])}`;
    E(svg, 'path', { d, fill: 'none', stroke: 'var(--quant)', 'stroke-width': 2 });
    E(svg, 'path', { d: pathD(Wt.map((v, t) => [x(t), y(v)])), fill: 'none', stroke: 'var(--fp)', 'stroke-width': 1.25 });
    const roomy = w >= 520;
    E(svg, 'text', { x: x(1), y: y(Wt[0]) - 9, class: 'label halo', style: 'fill: var(--fp)' }, roomy ? 'W, master weight' : 'W');
    E(svg, 'text', { x: x(1), y: y(1) + 17, class: 'label halo', style: 'fill: var(--quant)' }, roomy ? 'Ŵ, used in the forward pass' : 'Ŵ');
    E(svg, 'text', { x: R, y: y(0.5) - 5, 'text-anchor': 'end', class: 'tick halo' }, 'rounding boundary');
    if (osc) {
      const end = flips[flips.indexOf(osc) + 2];
      const t = E(svg, 'text', { x: x(osc.t) - 4, y: y(1.45), class: 'annot halo' }, `steps ${osc.t}–${end ? end.t : osc.t + 10}: Ŵ flips back and forth`);
      fitX(t, L, w);
    }
    if (cross) {
      const ax = x(cross.t) - 6;
      const t1 = E(svg, 'text', { x: ax, y: y(-1.55) + 16, 'text-anchor': 'end', class: 'annot halo' }, `step ${cross.t}: W crosses −0.5,`);
      const t2 = E(svg, 'text', { x: ax, y: y(-1.55) + 32, 'text-anchor': 'end', class: 'annot halo' }, 'and Ŵ drops from 0 to −1');
      fitX(t1, L, w); fitX(t2, L, w);
    }
  }
  responsive(host, draw);
})();

/* ---------- Figures 7 and 8: the training lab ---------- */
(function () {
  const H = 32, N = 64, FP_STEPS = 1500, QAT_STEPS = 2000, LR = 0.01, EVERY = 20, BITS = [2, 3, 4];
  const xs = Array.from({ length: N }, (_, i) => -1 + 2 * i / (N - 1));
  const ys = xs.map((x) => Math.sin(3 * x) + 0.3 * Math.cos(9 * x));
  function init(seed) { const r = mulberry(seed); const g = () => r() * 2 - 1; return { w1: Array.from({ length: H }, () => g() * 2), b1: Array.from({ length: H }, () => g() * 1.5), w2: Array.from({ length: H }, () => g() * 0.5), b2: 0 }; }
  /** Weights the forward pass sees: the master weights, or their fake-quantized copies. */
  function eff(p, bits) {
    if (!bits) return { w1: p.w1, w2: p.w2, m1: null, m2: null };
    const a = quantizeArray(p.w1, bits), c = quantizeArray(p.w2, bits);
    return { w1: a.out, w2: c.out, m1: a.mask, m2: c.mask };
  }
  function predict(p, e, x) { let y = p.b2; for (let j = 0; j < H; j++) y += e.w2[j] * Math.tanh(e.w1[j] * x + p.b1[j]); return y; }
  function loss(p, bits) { const e = eff(p, bits); let s = 0; for (let i = 0; i < N; i++) s += (predict(p, e, xs[i]) - ys[i]) ** 2; return s / N; }
  function makeOpt() { const z = () => new Array(H).fill(0); return { m: { w1: z(), b1: z(), w2: z(), b2: 0 }, v: { w1: z(), b1: z(), w2: z(), b2: 0 }, t: 0 }; }
  const h = new Float64Array(H);
  /** One full-batch Adam step. With bits set, the forward pass uses fake-quantized weights and the
   *  gradient reaches the master weights through the straight-through estimator (mask = 1 unless clipped). */
  function step(p, o, bits, lr) {
    const B1 = 0.9, B2 = 0.999, eps = 1e-8; o.t++; const t = o.t;
    const e = eff(p, bits);
    const g = { w1: new Array(H).fill(0), b1: new Array(H).fill(0), w2: new Array(H).fill(0), b2: 0 }; let Ls = 0;
    for (let i = 0; i < N; i++) {
      const x = xs[i]; let y = p.b2;
      for (let j = 0; j < H; j++) { h[j] = Math.tanh(e.w1[j] * x + p.b1[j]); y += e.w2[j] * h[j]; }
      const er = y - ys[i]; Ls += er * er / N; const dy = 2 * er / N; g.b2 += dy;
      for (let j = 0; j < H; j++) {
        g.w2[j] += dy * h[j] * (e.m2 ? e.m2[j] : 1);
        const dh = dy * e.w2[j] * (1 - h[j] * h[j]); g.b1[j] += dh; g.w1[j] += dh * x * (e.m1 ? e.m1[j] : 1);
      }
    }
    const upd = (k) => { for (let j = 0; j < H; j++) { o.m[k][j] = B1 * o.m[k][j] + (1 - B1) * g[k][j]; o.v[k][j] = B2 * o.v[k][j] + (1 - B2) * g[k][j] ** 2; p[k][j] -= lr * (o.m[k][j] / (1 - B1 ** t)) / (Math.sqrt(o.v[k][j] / (1 - B2 ** t)) + eps); } };
    upd('w1'); upd('b1'); upd('w2');
    o.m.b2 = B1 * o.m.b2 + (1 - B1) * g.b2; o.v.b2 = B2 * o.v.b2 + (1 - B2) * g.b2 ** 2; p.b2 -= lr * (o.m.b2 / (1 - B1 ** t)) / (Math.sqrt(o.v.b2 / (1 - B2 ** t)) + eps);
    return Ls;
  }
  const cosLR = (t, T) => LR * 0.5 * (1 + Math.cos(Math.PI * t / T));
  const clone = (p) => ({ w1: p.w1.slice(), b1: p.b1.slice(), w2: p.w2.slice(), b2: p.b2 });
  /** The whole experiment for one seed: fp32 pretraining, then PTQ and QAT at 2, 3 and 4 bits. */
  function train(seed) {
    const fp = init(seed), o = makeOpt();
    for (let t = 0; t < FP_STEPS; t++) step(fp, o, 0, cosLR(t, FP_STEPS));
    const l0 = loss(fp, 0), res = { seed, fp, l0, bits: {} };
    for (const b of BITS) {
      const qp = clone(fp), oq = makeOpt(), l1 = loss(fp, b), hist = [l1];
      for (let k = 0; k < QAT_STEPS; k++) { const L = step(qp, oq, b, cosLR(k, QAT_STEPS)); if (k % EVERY === EVERY - 1) hist.push(L); }
      const l2 = loss(qp, b);
      res.bits[b] = { l1, l2, hist, qat: qp, ratio: l1 / l2 };
    }
    return res;
  }

  /* Figure 8's data: PTQ loss ÷ QAT loss for seeds 1–8, computed with train() above.
     Precomputed because all eight runs take a few seconds; a test recomputes them and checks these values. */
  const SEED_RATIOS = {
    2: [1.11, 5.307, 0.3651, 29.91, 18.64, 56.08, 11.55, 11.06],
    3: [7.528, 4.077, 3.686, 2.838, 12.11, 9.063, 6.359, 5.87],
    4: [3.141, 12.44, 39.73, 5.87, 4.728, 40.5, 9.392, 6.886],
  };

  const fmtL = (v) => (v >= 0.001 ? v.toPrecision(2) : v.toExponential(1).replace('e-', 'e−'));
  const ratioText = (r) => (r >= 1 ? `${r < 10 ? r.toFixed(1) : Math.round(r)}× lower` : `${(1 / r).toFixed(1)}× higher`);
  const lab = { seed: 2, result: null, train, loss, SEED_RATIOS, ratioText };
  window.QAT.lab = lab;
  const status = $('#labStatus');

  function drawFit(host, res, b) {
    const Hh = 150;
    const { svg, w } = sized(host, Hh, `Fitted functions at ${b} bits: fp32 model, PTQ and QAT, against the training points.`);
    const L = 24, R = w - 4, T = 6, B = Hh - 20, x = lin(-1, 1, L, R), y = lin(-1.7, 1.7, B, T);
    const cid = 'fitclip' + b;
    E(E(E(svg, 'defs'), 'clipPath', { id: cid }), 'rect', { x: L, y: T, width: R - L, height: B - T });
    E(svg, 'line', { x1: L, x2: R, y1: B + 4, y2: B + 4, stroke: 'var(--rule-2)' });
    [-1, 0, 1].forEach((t) => { E(svg, 'text', { x: x(t), y: B + 17, 'text-anchor': t === -1 ? 'start' : t === 1 ? 'end' : 'middle', class: 'tick' }, num(t, 0)); E(svg, 'text', { x: L - 5, y: y(t) + 4, 'text-anchor': 'end', class: 'tick' }, num(t, 0)); });
    xs.forEach((xv, i) => E(svg, 'circle', { cx: x(xv), cy: y(ys[i]), r: 1.8, fill: 'var(--c-grey)' }));
    const grid = []; for (let v = -1; v <= 1.0001; v += 0.01) grid.push(v);
    const line = (p, bits, stroke, sw, dash) => { const e = eff(p, bits); E(svg, 'path', { d: pathD(grid.map((v) => [x(v), y(clamp(predict(p, e, v), -2, 2))])), fill: 'none', stroke, 'stroke-width': sw, 'stroke-dasharray': dash, 'clip-path': `url(#${cid})` }); };
    line(res.fp, 0, 'var(--fp)', 1.25);
    line(res.fp, b, 'var(--quant)', 1.5, '5 3');
    line(res.bits[b].qat, b, 'var(--qat)', 1.75);
  }
  function drawLoss(host, res, b, dom) {
    const Hh = 150;
    const r = res.bits[b];
    const { svg, w } = sized(host, Hh, `Loss during QAT at ${b} bits on a log scale: fp32 ${fmtL(res.l0)}, PTQ ${fmtL(r.l1)}, QAT ends at ${fmtL(r.l2)}.`);
    const L = 34, R = w - 84, T = 8, B = Hh - 34;
    const x = lin(0, QAT_STEPS, L, R), y = logScale(dom[0], dom[1], B, T);
    for (let e = Math.ceil(Math.log10(dom[0])); e <= Math.floor(Math.log10(dom[1])); e++) {
      E(svg, 'line', { x1: L, x2: R, y1: y(10 ** e), y2: y(10 ** e), stroke: 'var(--rule)' });
      E(svg, 'text', { x: L - 5, y: y(10 ** e) + 4, 'text-anchor': 'end', class: 'tick' }, pow10(e));
    }
    E(svg, 'line', { x1: L, x2: R, y1: B + 4, y2: B + 4, stroke: 'var(--rule-2)' });
    [0, 1000, 2000].forEach((t) => E(svg, 'text', { x: x(t), y: B + 17, 'text-anchor': t === 0 ? 'start' : t === 2000 ? 'end' : 'middle', class: 'tick' }, t.toLocaleString('en-US')));
    E(svg, 'text', { x: R, y: B + 31, 'text-anchor': 'end', class: 'axis-title' }, 'QAT step');
    E(svg, 'line', { x1: L, x2: R, y1: y(res.l0), y2: y(res.l0), stroke: 'var(--fp)', 'stroke-width': 1.25, 'stroke-dasharray': '4 3' });
    E(svg, 'line', { x1: L, x2: R, y1: y(r.l1), y2: y(r.l1), stroke: 'var(--quant)', 'stroke-width': 1.5, 'stroke-dasharray': '5 3' });
    E(svg, 'path', { d: pathD(r.hist.map((v, i) => [x(i * EVERY), y(clamp(v, dom[0], dom[1]))])), fill: 'none', stroke: 'var(--qat)', 'stroke-width': 1.25, 'stroke-linejoin': 'round' });
    const labels = spread([
      { y: y(r.l1) + 4, t: 'PTQ ' + fmtL(r.l1), c: 'var(--quant)', k: 'ptq' },
      { y: y(r.l2) + 4, t: 'QAT ' + fmtL(r.l2), c: 'var(--qat)', k: 'qat' },
      { y: y(res.l0) + 4, t: 'fp32 ' + fmtL(res.l0), c: 'var(--fp)', k: 'fp' },
    ], 13, T + 8, B + 4);
    labels.forEach((l) => E(svg, 'text', { x: R + 6, y: l.y, class: 'label', style: `fill: ${l.c}`, 'data-k': l.k }, l.t));
  }
  function drawSeeds() {
    const host = $('#seedPlot');
    const rowH = 38, top = 28, Hh = top + BITS.length * rowH + 40;
    const { svg, w } = sized(host, Hh, 'Dot plot of how many times lower the QAT loss is than the PTQ loss, for eight seeds at 2, 3 and 4 bits, on a log scale.');
    const L = 50, R = w - 10, x = logScale(0.25, 80, L, R);
    E(svg, 'line', { x1: x(1), x2: x(1), y1: top - 12, y2: top + BITS.length * rowH, stroke: 'var(--ink-2)' });
    E(svg, 'text', { x: x(1) - 5, y: top - 14, 'text-anchor': 'end', class: 'annot' }, 'QAT worse');
    E(svg, 'text', { x: x(1) + 5, y: top - 14, class: 'annot' }, 'QAT better →');
    BITS.forEach((b, i) => {
      const cy = top + i * rowH + rowH / 2;
      E(svg, 'text', { x: 0, y: cy + 4, class: 'label' }, b + ' bits');
      E(svg, 'line', { x1: L, x2: R, y1: cy, y2: cy, stroke: 'var(--rule)' });
      SEED_RATIOS[b].forEach((v, k) => { if (k + 1 !== lab.seed) E(svg, 'circle', { cx: x(v), cy, r: 4, fill: 'none', stroke: 'var(--ink-3)', 'stroke-width': 1.25 }); });
      const v = SEED_RATIOS[b][lab.seed - 1];
      E(svg, 'circle', { cx: x(v), cy, r: 4.5, fill: 'var(--ink)' });
      const t = E(svg, 'text', { x: x(v), y: cy - 9, 'text-anchor': 'middle', class: 'label halo' }, 'seed ' + lab.seed);
      fitX(t, L, w);
    });
    const ay = top + BITS.length * rowH + 6;
    [0.5, 1, 2, 5, 10, 20, 50].forEach((v) => E(svg, 'text', { x: x(v), y: ay + 12, 'text-anchor': 'middle', class: 'tick' }, v + '×'));
    E(svg, 'text', { x: R, y: ay + 30, 'text-anchor': 'end', class: 'axis-title' }, 'PTQ loss ÷ QAT loss (log scale)');
  }
  function lossDomain(res) {
    const all = [res.l0]; BITS.forEach((b) => { all.push(res.bits[b].l1, res.bits[b].l2, ...res.bits[b].hist); });
    return [Math.min(...all) / 1.6, Math.max(...all) * 1.6];
  }
  function drawLab() {
    const res = lab.result; if (!res) return;
    const dom = lossDomain(res);
    BITS.forEach((b) => {
      drawFit($('#labFit' + b), res, b);
      drawLoss($('#labLoss' + b), res, b, dom);
      $('#labGain' + b).innerHTML = `QAT loss <b>${ratioText(res.bits[b].ratio)}</b> than PTQ`;
    });
    const part = (b) => `${ratioText(res.bits[b].ratio)} at ${b} bits`;
    const lose = BITS.filter((b) => res.bits[b].ratio < 1);
    let s = `Seed ${res.seed}: compared with PTQ, QAT's final loss is ${part(2)}, ${part(3)} and ${part(4)}.`;
    if (lose.length) s += ` At ${lose.join(' and ')} bits this QAT run ends worse than rounding once; the grid is too coarse for it to recover.`;
    else s += ' Seed 3 shows a run where QAT ends worse at 2 bits.';
    $('#labTakeaway').textContent = s;
    drawSeeds();
  }
  function run(seed) {
    lab.seed = seed;
    document.querySelectorAll('#labSeed button').forEach((b) => b.setAttribute('aria-pressed', String(+b.dataset.seed === seed)));
    const t0 = performance.now();
    lab.result = train(seed);
    status.textContent = `Trained in your browser in ${Math.round(performance.now() - t0)} ms.`;
    drawLab();
  }
  lab.run = run;
  document.querySelectorAll('#labSeed button').forEach((b) => b.addEventListener('click', () => {
    const seed = +b.dataset.seed; if (seed === lab.seed && lab.result) return;
    status.textContent = 'Training…';
    // let the status paint before the synchronous training blocks the main thread
    requestAnimationFrame(() => setTimeout(() => run(seed), 0));
  }));
  run(lab.seed);
  responsive($('#fig-lab .lab-grid'), drawLab);
  responsive($('#seedPlot'), () => { if (lab.result) drawSeeds(); });
})();
})();
