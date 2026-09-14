/* ------------------------------------------------------------------
   Sharding widget: annotate the two operands of y = x @ w with
   PartitionSpecs on a 2 x 4 device mesh and watch which collectives a
   (deliberately simplified) SPMD partitioner has to insert.
   Registers JT.widget('sharding').
   ------------------------------------------------------------------ */
(function () {
  'use strict';
  const JT = window.JT;
  const C = JT.colors;

  /* @model-start */
  /* Mesh: 2 x 4, axes ('data', 'model'); device id = 4 * data + model. */
  const AXES = { data: 2, model: 4 };
  const DEVICES = [0, 1, 2, 3, 4, 5, 6, 7];
  const coord = { data: (d) => d >> 2, model: (d) => d & 3 };
  /* Validated colorblind-safe categorical set: hue = 'model' index. */
  const HUES = ['#2a78d6', '#eb6834', '#1baf7a', '#4a3aa7'];
  const GRAY = '#c9c6bb';

  function mixWhite(hex, t) {
    const n = parseInt(hex.slice(1), 16);
    const ch = (v) => Math.round(v + (255 - v) * t).toString(16).padStart(2, '0');
    return '#' + ch(n >> 16) + ch((n >> 8) & 255) + ch(n & 255);
  }
  /* Lightness = 'data' index: data 0 keeps the hue, data 1 is a tint. */
  const deviceColor = (d) => (coord.data(d) === 0 ? HUES[coord.model(d)] : mixWhite(HUES[coord.model(d)], 0.45));

  const specStr = (s) => (!s[0] && !s[1] ? 'P()' : 'P(' + s.map((a) => (a ? "'" + a + "'" : 'None')).join(', ') + ')');
  const key = (s) => s.map((a) => a || '-').join(',');
  const unkey = (k) => k.split(',').map((a) => (a === '-' ? null : a));
  /* Denominator of the per-device fraction: product of the sharded mesh axes. */
  const fracDen = (s) => s.reduce((n, a) => n * (a ? AXES[a] : 1), 1);

  /**
   * Simplified SPMD partitioner for y = x @ w.
   * xSpec = [bx, dx], wSpec = [dw, fw]; each entry is a mesh-axis name or null.
   * Returns { steps: [{type, over, of}], y: [by, fy], mem: {x, w, y}, notes }.
   */
  function plan(xSpec, wSpec) {
    let bx = xSpec[0], dx = xSpec[1], dw = wSpec[0], fw = wSpec[1];
    const steps = [], notes = [];
    /* Rule 3 (applied first): y = (bx, fw) cannot carry the same axis on both dims. */
    if (bx && fw && bx === fw) {
      steps.push({ type: 'all-gather', over: fw, of: 'w' });
      fw = null;
      notes.push("y cannot use '" + bx + "' on both of its dims, so w is gathered along F first");
    }
    /* Rule 1: the contracting dim D must be split the same way on both operands. */
    if (dx !== dw) {
      if (dx && dw) {
        steps.push({ type: 'all-gather', over: dw, of: 'w' }); dw = null;
        steps.push({ type: 'all-gather', over: dx, of: 'x' }); dx = null;
        notes.push('both operands gathered');
      } else if (dw) {
        steps.push({ type: 'all-gather', over: dw, of: 'w' }); dw = null;
      } else if (dx) {
        steps.push({ type: 'all-gather', over: dx, of: 'x' }); dx = null;
      }
    }
    /* Rule 2: D split identically on both -> every device holds a partial sum. */
    if (dx && dw && dx === dw) steps.push({ type: 'all-reduce', over: dx, of: 'y' });
    const y = [bx, fw];
    return { steps, y, notes, mem: { x: fracDen(xSpec), w: fracDen(wSpec), y: fracDen(y) } };
  }
  /* @plan-end */

  const X_OPTS = [
    { value: 'data,-', label: "P('data', None)" },
    { value: '-,model', label: "P(None, 'model')" },
    { value: 'data,model', label: "P('data', 'model')" },
    { value: '-,-', label: 'P()' },
  ];
  const W_OPTS = [
    { value: '-,-', label: 'P()' },
    { value: 'model,-', label: "P('model', None)" },
    { value: '-,model', label: "P(None, 'model')" },
    { value: 'data,-', label: "P('data', None)" },
  ];
  const PRESETS = [
    { id: 'ddp', label: 'Data parallel (DDP-like)', x: ['data', null], w: [null, null], torch: 'DistributedDataParallel: replicate w on every rank, split the batch, all-reduce gradients in the backward pass' },
    { id: 'coltp', label: 'Column tensor parallel', x: [null, null], w: [null, 'model'], torch: 'column-parallel Linear (Megatron-style): each rank owns a slice of the output features' },
    { id: 'rowtp', label: 'Row tensor parallel', x: [null, 'model'], w: ['model', null], torch: 'row-parallel Linear + all-reduce (Megatron-style): partial sums are summed with NCCL' },
    { id: 'fsdp', label: 'FSDP-like', x: ['data', null], w: ['data', null], torch: 'FullyShardedDataParallel: parameters are sharded and all-gathered right before use' },
    { id: '2d', label: '2-D: data × model', x: ['data', null], w: [null, 'model'], torch: '2-D parallel with DTensor: Shard(0) on the data mesh dim for x, Shard(1) on the model mesh dim for w' },
  ];
  const stepStr = (s) => (s.type === 'all-gather' ? "all-gather " + s.of + " over '" + s.over + "'" : "all-reduce over '" + s.over + "'");

  const CSS = `
#w-sharding .sh-ctl { display: grid; grid-template-columns: auto minmax(0, 1fr); gap: 0.5rem 0.75rem; align-items: center; margin-bottom: 0.85rem; }
#w-sharding .sh-ctl .lbl { font-family: var(--font-mono); font-size: 0.8125rem; color: var(--ink-3); white-space: nowrap; }
#w-sharding .sh-ctl .lbl b { color: var(--ink); font-weight: 600; }
#w-sharding .seg { flex-wrap: wrap; }
#w-sharding .seg button { font-family: var(--font-mono); font-size: 0.75rem; padding: 0.3rem 0.6rem; white-space: nowrap; }
#w-sharding .sh-presets { display: flex; flex-wrap: wrap; align-items: center; gap: 0.4rem; padding-bottom: 0.9rem; margin-bottom: 1rem; border-bottom: 1px solid var(--line); }
#w-sharding .sh-presets .lbl { font-size: 0.8125rem; color: var(--ink-3); margin-right: 0.25rem; }
#w-sharding .sh-main { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 1.25rem; align-items: start; }
#w-sharding .sh-left { min-width: 0; }
#w-sharding .sh-mesh-row { display: flex; flex-wrap: wrap; align-items: center; gap: 0.75rem 1.25rem; }
#w-sharding .sh-mesh-row .sh-mesh { flex: 1 1 220px; max-width: 300px; }
#w-sharding .sh-legend { flex: 1 1 180px; display: grid; gap: 0.3rem; font-size: 0.78rem; color: var(--ink-2); line-height: 1.4; }
#w-sharding .sh-legend .sw { display: inline-block; width: 12px; height: 12px; border-radius: 3px; vertical-align: -1px; margin-right: 0.4rem; }
#w-sharding .sh-legend .sw.striped { background: repeating-linear-gradient(45deg, #2a78d6 0 3px, #eb6834 3px 6px, #1baf7a 6px 9px, #4a3aa7 9px 12px); }
#w-sharding .sh-legend .dim { color: var(--ink-3); }
#w-sharding .sh-arrays { display: grid; grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr) auto minmax(0, 1fr); gap: 0.5rem 0.6rem; align-items: start; margin: 1.1rem 0 0.4rem; }
#w-sharding .sh-arr { min-width: 0; text-align: center; }
#w-sharding .sh-arr svg { width: 100%; max-width: 176px; margin: 0 auto; overflow: visible; }
#w-sharding .sh-cap { font-size: 0.8125rem; color: var(--ink-2); line-height: 1.3; margin-bottom: 0.4rem; min-height: 2.4rem; }
#w-sharding .sh-cap b { color: var(--ink); font-family: var(--font-mono); font-weight: 600; }
#w-sharding .sh-cap .spec { display: block; font-family: var(--font-mono); font-size: 0.75rem; color: var(--jax-deep); }
#w-sharding .sh-op { font-family: var(--font-mono); font-size: 1.3rem; color: var(--ink-3); text-align: center; padding-top: calc(2.4rem + 0.4rem + 3.2rem); line-height: 1; }
#w-sharding .sh-readout { margin: 0.5rem 0 1rem; display: flex; flex-wrap: wrap; gap: 0.25rem 1rem; }
#w-sharding .sh-coll { border: 1px solid var(--line); border-radius: var(--radius-sm); background: var(--surface-2); padding: 0.7rem 0.85rem; }
#w-sharding .sh-coll h5 { margin: 0 0 0.5rem; font-size: 0.72rem; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase; color: var(--ink-3); }
#w-sharding .sh-coll .none { color: var(--live-deep); font-weight: 500; font-size: 0.875rem; }
#w-sharding .sh-cards { display: grid; gap: 0.55rem; }
#w-sharding .sh-card { display: flex; flex-wrap: wrap; align-items: center; gap: 0.4rem 0.85rem; background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius-sm); padding: 0.55rem 0.7rem; }
#w-sharding .sh-card svg { width: 150px; flex: 0 0 auto; }
#w-sharding .sh-card .txt { flex: 1 1 160px; min-width: 0; font-size: 0.8125rem; color: var(--ink-2); line-height: 1.45; }
#w-sharding .sh-card .txt b { display: block; color: var(--live-deep); font-family: var(--font-mono); font-weight: 600; font-size: 0.8125rem; margin-bottom: 0.15rem; }
#w-sharding .sh-card .when { display: inline-block; font-size: 0.68rem; font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase; color: var(--ink-3); background: var(--surface-2); border-radius: 999px; padding: 0.1rem 0.5rem; margin-left: 0.4rem; vertical-align: 1px; font-family: var(--font-body); }
#w-sharding .sh-coll .note { margin: 0.55rem 0 0; font-size: 0.78rem; color: var(--ink-3); }
#w-sharding .sh-code { min-width: 0; }
#w-sharding .sh-code pre.code { font-size: 0.78rem; }
#w-sharding .sh-peq { display: flex; flex-wrap: wrap; gap: 0.2rem 0.5rem; padding: 0.6rem 0.9rem; border-top: 1px solid var(--line); background: var(--torch-soft); font-size: 0.8125rem; color: var(--ink-2); line-height: 1.45; }
#w-sharding .sh-peq .lbl { color: var(--torch-deep); font-weight: 600; white-space: nowrap; }
#w-sharding .sh-peq .val { flex: 1 1 200px; min-width: 0; }
#w-sharding svg .cell { stroke: #fff; stroke-width: 0.8; transition: fill 250ms var(--ease-out); }
#w-sharding.sh-static svg .cell { transition: none; }
#w-sharding svg .blk-outline { fill: none; stroke: #fff; stroke-width: 1.6; pointer-events: none; }
#w-sharding svg text.blk { font-family: var(--font-mono); font-size: 7.5px; fill: var(--ink); pointer-events: none; }
#w-sharding svg .pill { fill: #fff; fill-opacity: 0.92; pointer-events: none; }
#w-sharding svg .hit { fill: transparent; }
#w-sharding svg text.did { font-family: var(--font-body); font-size: 15px; font-weight: 600; fill: var(--ink); paint-order: stroke; stroke: #fff; stroke-width: 3px; stroke-linejoin: round; }
#w-sharding svg text.did.small { font-size: 10px; stroke-width: 2.5px; }
#w-sharding svg text.dco { font-family: var(--font-mono); font-size: 7px; fill: var(--ink); paint-order: stroke; stroke: #fff; stroke-width: 1.4px; stroke-linejoin: round; }
#w-sharding svg text.ax { font-size: 9.5px; fill: var(--ink-3); }
#w-sharding svg text.tick { font-family: var(--font-mono); font-size: 9px; fill: var(--ink-3); }
@media (max-width: 880px) {
  #w-sharding .sh-main { grid-template-columns: minmax(0, 1fr); }
}
@media (max-width: 600px) {
  #w-sharding .sh-ctl { grid-template-columns: minmax(0, 1fr); gap: 0.3rem; }
  #w-sharding .sh-ctl .seg { border-radius: 12px; margin-bottom: 0.35rem; }
  #w-sharding .sh-arrays { grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr); }
  #w-sharding .sh-op.eq { grid-column: 2; grid-row: 2; }
  #w-sharding .sh-arr.y { grid-column: 3; grid-row: 2; }
}
`;

  /* ---------------------------------------------------------------- */
  /* Drawing helpers                                                    */
  /* ---------------------------------------------------------------- */

  /** 2 x 4 mesh. o: {cell, gapX, gapY, ml, mt, mr, mb, axes, coords, groups, uid, label, cls} */
  function drawMesh(o) {
    const cell = o.cell, gx = o.gapX, gy = o.gapY;
    const gridW = 4 * cell + 3 * gx, gridH = 2 * cell + gy;
    const W = o.ml + gridW + o.mr, H = o.mt + gridH + o.mb;
    const svg = JT.svg('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': o.label || '2 by 4 device mesh', class: o.cls || '' });
    const at = (d) => ({ x: o.ml + coord.model(d) * (cell + gx), y: o.mt + coord.data(d) * (cell + gy) });
    const groups = o.groups ? (o.groups === 'model' ? [[0, 1, 2, 3], [4, 5, 6, 7]] : [[0, 4], [1, 5], [2, 6], [3, 7]]) : null;
    const mid = 'sh-arrow-' + o.uid;
    if (groups) {
      svg.appendChild(JT.svg('defs', null, JT.svg('marker', { id: mid, viewBox: '0 0 8 8', refX: 7, refY: 4, markerWidth: 5, markerHeight: 5, orient: 'auto-start-reverse' }, JT.svg('path', { d: 'M0 0 L8 4 L0 8 Z', fill: C.liveDeep }))));
      const pad = 4;
      for (const g of groups) {
        const ps = g.map(at);
        const x0 = Math.min.apply(null, ps.map((p) => p.x)) - pad, y0 = Math.min.apply(null, ps.map((p) => p.y)) - pad;
        const x1 = Math.max.apply(null, ps.map((p) => p.x)) + cell + pad, y1 = Math.max.apply(null, ps.map((p) => p.y)) + cell + pad;
        svg.appendChild(JT.svg('rect', { x: x0, y: y0, width: x1 - x0, height: y1 - y0, rx: 6, fill: C.liveSoft, stroke: C.live, 'stroke-width': 1.25 }));
      }
    }
    for (const d of DEVICES) {
      const p = at(d), cx = p.x + cell / 2;
      svg.appendChild(JT.svg('rect', { x: p.x, y: p.y, width: cell, height: cell, rx: Math.round(cell * 0.12), fill: deviceColor(d), class: 'dev' }));
      if (o.coords) {
        svg.appendChild(JT.svg('text', { x: cx, y: p.y + cell / 2 + 2, 'text-anchor': 'middle', class: 'did', text: String(d) }));
        svg.appendChild(JT.svg('text', { x: cx, y: p.y + cell - 5, 'text-anchor': 'middle', class: 'dco', text: '(' + coord.data(d) + ', ' + coord.model(d) + ')' }));
      } else {
        svg.appendChild(JT.svg('text', { x: cx, y: p.y + cell / 2 + 3.5, 'text-anchor': 'middle', class: 'did small', text: String(d) }));
      }
    }
    if (groups) {
      const horiz = o.groups === 'model';
      for (const g of groups) {
        for (let i = 0; i + 1 < g.length; i++) {
          const a = at(g[i]), b = at(g[i + 1]);
          const line = horiz
            ? { x1: a.x + cell + 1.5, y1: a.y + cell / 2, x2: b.x - 1.5, y2: b.y + cell / 2 }
            : { x1: a.x + cell / 2, y1: a.y + cell + 1.5, x2: b.x + cell / 2, y2: b.y - 1.5 };
          svg.appendChild(JT.svg('line', Object.assign({ stroke: C.liveDeep, 'stroke-width': 1.6, 'marker-start': 'url(#' + mid + ')', 'marker-end': 'url(#' + mid + ')' }, line)));
        }
      }
    }
    if (o.axes) {
      svg.appendChild(JT.svg('text', { x: o.ml + gridW / 2, y: 10, 'text-anchor': 'middle', class: 'ax', text: "'model' axis, size 4" }));
      for (let m = 0; m < 4; m++) svg.appendChild(JT.svg('text', { x: o.ml + m * (cell + gx) + cell / 2, y: o.mt - 6, 'text-anchor': 'middle', class: 'tick', text: String(m) }));
      const ly = o.mt + gridH / 2;
      svg.appendChild(JT.svg('text', { x: 11, y: ly, 'text-anchor': 'middle', class: 'ax', transform: 'rotate(-90 11 ' + ly + ')', text: "'data' axis, size 2" }));
      for (let r = 0; r < 2; r++) svg.appendChild(JT.svg('text', { x: o.ml - 7, y: o.mt + r * (cell + gy) + cell / 2 + 3.5, 'text-anchor': 'end', class: 'tick', text: String(r) }));
    }
    return svg;
  }

  /** An 8 x 8 array grid whose cells persist (so fills can transition). */
  function makeGrid(name, dims, extraCls) {
    const svg = JT.svg('svg', { viewBox: '0 0 100 100', role: 'img', 'aria-label': name + ' array, 8 by 8, colored by the device that holds each block' });
    const defs = JT.svg('defs');
    const cells = JT.svg('g');
    const rects = [];
    for (let r = 0; r < 8; r++) for (let c = 0; c < 8; c++) {
      const rect = JT.svg('rect', { x: 2 + c * 12, y: 2 + r * 12, width: 12, height: 12, class: 'cell' });
      rects.push(rect); cells.appendChild(rect);
    }
    const blocks = JT.svg('g');
    svg.append(defs, cells, blocks);
    const cap = JT.el('div', { class: 'sh-cap' });
    const wrap = JT.el('div', { class: 'sh-arr ' + (extraCls || '') }, [cap, svg]);
    return { name, dims, svg, defs, rects, blocks, cap, wrap };
  }

  function fillFor(devs, defs, uid) {
    if (devs.length === 8) return GRAY;
    if (devs.length === 1) return deviceColor(devs[0]);
    const id = 'sh-pat-' + uid + '-' + devs.join('-');
    if (!defs.querySelector('#' + id)) {
      const n = devs.length, sw = 3;
      const pat = JT.svg('pattern', { id, patternUnits: 'userSpaceOnUse', width: n * sw, height: n * sw, patternTransform: 'rotate(45)' });
      devs.forEach((d, k) => pat.appendChild(JT.svg('rect', { x: k * sw, y: 0, width: sw, height: n * sw, fill: deviceColor(d) })));
      defs.appendChild(pat);
    }
    return 'url(#' + id + ')';
  }

  function paintGrid(g, spec, uid) {
    const nR = spec[0] ? AXES[spec[0]] : 1, nC = spec[1] ? AXES[spec[1]] : 1;
    const rh = 8 / nR, cw = 8 / nC;
    g.defs.replaceChildren();
    g.blocks.replaceChildren();
    for (let i = 0; i < nR; i++) for (let j = 0; j < nC; j++) {
      const devs = DEVICES.filter((d) => (!spec[0] || coord[spec[0]](d) === i) && (!spec[1] || coord[spec[1]](d) === j));
      const fill = fillFor(devs, g.defs, uid + g.name);
      for (let r = i * rh; r < (i + 1) * rh; r++) for (let c = j * cw; c < (j + 1) * cw; c++) g.rects[r * 8 + c].style.fill = fill;
      const x = 2 + j * cw * 12, y = 2 + i * rh * 12, w = cw * 12, h = rh * 12;
      g.blocks.appendChild(JT.svg('rect', { x, y, width: w, height: h, rx: 1, class: 'blk-outline' }));
      const label = devs.length === 8 ? 'all' : devs.join(' ');
      const tw = label.length * 4.6 + 5, th = 10.5;
      g.blocks.appendChild(JT.svg('rect', { x: x + w / 2 - tw / 2, y: y + h / 2 - th / 2, width: tw, height: th, rx: 2.5, class: 'pill' }));
      g.blocks.appendChild(JT.svg('text', { x: x + w / 2, y: y + h / 2 + 2.7, 'text-anchor': 'middle', class: 'blk', text: label }));
      const who = devs.length === 8 ? 'all 8 devices (replicated)' : (devs.length > 1 ? 'devices ' + devs.join(', ') + ' (shared)' : 'device ' + devs[0] + ' only');
      g.blocks.appendChild(JT.svg('rect', { x, y, width: w, height: h, class: 'hit', 'data-tip': '<b>' + g.name + '[' + (i * rh) + ':' + ((i + 1) * rh) + ', ' + (j * cw) + ':' + ((j + 1) * cw) + ']</b> lives on ' + who }));
    }
  }

  /* ---------------------------------------------------------------- */
  /* Widget                                                             */
  /* ---------------------------------------------------------------- */
  JT.widget('sharding', (container) => {
    JT.style('sharding', CSS);
    const uid = 'sh' + Math.random().toString(36).slice(2, 7);
    const S = JT.stage(container, { title: 'Shard a matmul across a 2 × 4 device mesh', hint: 'y = x @ w on 8 devices, simplified GSPMD / Shardy rules' });
    if (JT.reducedMotion) container.classList.add('sh-static');

    let xSpec = ['data', null], wSpec = [null, null];

    /* controls */
    const xSeg = JT.seg(X_OPTS, (v) => { xSpec = unkey(v); render(); }, key(xSpec), 'jax');
    const wSeg = JT.seg(W_OPTS, (v) => { wSpec = unkey(v); render(); }, key(wSpec), 'jax');
    xSeg.setAttribute('aria-label', 'PartitionSpec for x');
    wSeg.setAttribute('aria-label', 'PartitionSpec for w');
    const ctl = JT.el('div', { class: 'sh-ctl' }, [
      JT.el('span', { class: 'lbl', html: '<b>x</b> [B=8, D=8]' }), xSeg,
      JT.el('span', { class: 'lbl', html: '<b>w</b> [D=8, F=8]' }), wSeg,
    ]);
    const chips = PRESETS.map((p) => JT.el('button', {
      type: 'button', class: 'chip', text: p.label, 'aria-pressed': 'false', dataset: { preset: p.id },
      onClick: () => { xSpec = p.x.slice(); wSpec = p.w.slice(); xSeg.setValue(key(xSpec)); wSeg.setValue(key(wSpec)); render(); },
    }));
    const presets = JT.el('div', { class: 'sh-presets' }, [JT.el('span', { class: 'lbl', text: 'Presets' })].concat(chips));

    /* mesh + legend */
    const mesh = drawMesh({ cell: 44, gapX: 8, gapY: 8, ml: 36, mt: 30, mr: 4, mb: 4, axes: true, coords: true, uid: uid + 'm', cls: 'sh-mesh', label: 'Device mesh: 2 rows (data axis) by 4 columns (model axis); each square is one device, labeled with its id and (data, model) coordinate' });
    const legend = JT.el('div', { class: 'sh-legend' }, [
      JT.el('div', null, [JT.el('span', { class: 'sw', style: { background: HUES[0] } }), 'solid: this block lives on exactly one device']),
      JT.el('div', null, [JT.el('span', { class: 'sw striped' }), 'striped: shared by the striped devices (replicated along one mesh axis)']),
      JT.el('div', null, [JT.el('span', { class: 'sw', style: { background: GRAY } }), 'gray, "all": replicated on every device']),
      JT.el('div', { class: 'dim', text: "hue = 'model' index, tint = 'data' index (lighter is data 1); device id = 4 × data + model" }),
    ]);
    const meshRow = JT.el('div', { class: 'sh-mesh-row' }, [mesh, legend]);

    /* arrays */
    const gx = makeGrid('x', '[B=8, D=8]'), gw = makeGrid('w', '[D=8, F=8]'), gy = makeGrid('y', '[B=8, F=8]', 'y');
    const arrays = JT.el('div', { class: 'sh-arrays' }, [
      gx.wrap, JT.el('div', { class: 'sh-op', text: '@', 'aria-hidden': 'true' }), gw.wrap, JT.el('div', { class: 'sh-op eq', text: '=', 'aria-hidden': 'true' }), gy.wrap,
    ]);
    const readout = JT.el('div', { class: 'readout sh-readout' });

    /* collectives strip */
    const collBody = JT.el('div');
    const coll = JT.el('div', { class: 'sh-coll', 'aria-live': 'polite' }, [JT.el('h5', { text: 'Collectives inserted by the compiler' }), collBody]);

    /* code pane */
    const codeHost = JT.el('div');
    const peqVal = JT.el('span', { class: 'val' });
    const pane = JT.el('div', { class: 'pane jax sh-code' }, [
      JT.el('div', { class: 'pane-head' }, ['JAX', JT.el('span', { class: 'sub', text: 'annotate the data, jit the math' })]),
      codeHost,
      JT.el('div', { class: 'sh-peq' }, [JT.el('span', { class: 'lbl', text: 'PyTorch equivalent' }), peqVal]),
    ]);

    const main = JT.el('div', { class: 'sh-main' }, [JT.el('div', { class: 'sh-left' }, [meshRow, arrays, readout, coll]), pane]);
    S.body.append(ctl, presets, main);
    S.foot.textContent = 'Simplified: the real partitioner propagates shardings through the full program and uses cost models to choose placements, collectives, and resharding.';

    const shardWord = (s) => (s[0] || s[1] ? 'sharded ' + specStr(s) : 'replicated ' + specStr(s));
    const fracStr = (den) => '1/' + den;

    function render() {
      const p = plan(xSpec, wSpec);

      paintGrid(gx, xSpec, uid); paintGrid(gw, wSpec, uid); paintGrid(gy, p.y, uid);
      gx.cap.innerHTML = '<b>x</b> ' + gx.dims + '<span class="spec">' + JT.escape(specStr(xSpec)) + '</span>';
      gw.cap.innerHTML = '<b>w</b> ' + gw.dims + '<span class="spec">' + JT.escape(specStr(wSpec)) + '</span>';
      gy.cap.innerHTML = '<b>y</b> ' + gy.dims + '<span class="spec">' + JT.escape(specStr(p.y)) + '</span>';

      readout.innerHTML = '<span>per-device memory: x <b>' + fracStr(p.mem.x) + '</b> · w <b>' + fracStr(p.mem.w) + '</b> · y <b>' + fracStr(p.mem.y) + '</b></span>'
        + '<span>y comes out <b>' + JT.escape(shardWord(p.y)) + '</b></span>';

      /* collectives */
      collBody.replaceChildren();
      if (!p.steps.length) {
        collBody.appendChild(JT.el('div', { class: 'none', text: 'No communication needed: every device already has what it needs.' }));
      } else {
        const cards = JT.el('div', { class: 'sh-cards' });
        p.steps.forEach((s, i) => {
          const n = AXES[s.over], groups = 8 / n;
          const mini = drawMesh({ cell: 22, gapX: s.over === 'model' ? 16 : 12, gapY: s.over === 'model' ? 12 : 16, ml: 6, mt: 6, mr: 6, mb: 6, groups: s.over, uid: uid + 'c' + i, label: (s.type + " over '" + s.over + "': " + groups + ' groups of ' + n + ' devices') });
          const when = s.type === 'all-gather' ? 'before matmul' : 'after matmul';
          const what = s.type === 'all-gather'
            ? n + ' devices per group, ' + groups + ' groups. Each group concatenates its slices so every member holds ' + s.of + ' whole along that axis.'
            : n + ' devices per group, ' + groups + ' groups. Every device computed a partial y from its slice of D; each group sums them. XLA may pick reduce-scatter instead when y should stay sharded.';
          const txt = JT.el('div', { class: 'txt' }, [
            JT.el('b', null, [stepStr(s) + ' (' + n + ' devices)', JT.el('span', { class: 'when', text: when })]),
            what,
          ]);
          cards.appendChild(JT.el('div', { class: 'sh-card' }, [mini, txt]));
        });
        collBody.appendChild(cards);
        if (p.notes.length) collBody.appendChild(JT.el('p', { class: 'note', text: 'Note: ' + p.notes.join('; ') + '.' }));
      }

      /* code */
      const lines = [
        'from jax.sharding import NamedSharding, PartitionSpec as P',
        "mesh = jax.make_mesh((2, 4), ('data', 'model'))",
        'x = jax.device_put(x, NamedSharding(mesh, ' + specStr(xSpec) + '))',
        'w = jax.device_put(w, NamedSharding(mesh, ' + specStr(wSpec) + '))',
        'y = jax.jit(lambda x, w: x @ w)(x, w)',
      ];
      lines.push('print(y.sharding.spec)   # ' + specStr(p.y) + (p.y[0] || p.y[1] ? '' : ', replicated'));
      const firstComment = lines.length + 1;
      if (!p.steps.length) {
        lines.push('# XLA inserts: nothing — no communication needed');
      } else {
        lines.push('# XLA inserts: ' + stepStr(p.steps[0]) + (p.steps.length > 1 ? ',' : ''));
        p.steps.slice(1).forEach((s, i) => lines.push('#   then ' + stepStr(s) + (i === p.steps.length - 2 && p.notes.length ? ' (' + p.notes[0] + ')' : '')));
      }
      const pre = JT.code(lines.join('\n'), { lang: 'python' });
      JT.markLines(pre, lines.map((_, i) => i + 1).filter((n) => n >= firstComment), 'now');
      codeHost.replaceChildren(pre);

      /* presets + torch mapping */
      const active = PRESETS.find((q) => key(q.x) === key(xSpec) && key(q.w) === key(wSpec));
      chips.forEach((c) => c.setAttribute('aria-pressed', String(!!active && c.dataset.preset === active.id)));
      peqVal.textContent = active ? active.torch : 'custom DTensor placement (no standard PyTorch wrapper; spell the Shard/Replicate placements out by hand)';
    }

    render();
    JT.bindTips(S.root);
  });
})();
