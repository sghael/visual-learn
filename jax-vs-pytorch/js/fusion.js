/* ------------------------------------------------------------------
   Fusion widget: why a compiled program (XLA via jax.jit, Inductor via
   torch.compile) moves less memory than eager op-by-op execution.
   A chain of ops on a [4096, 4096] float32 tensor is grouped into
   kernels; the timeline shows every HBM read/write each kernel makes.
   ------------------------------------------------------------------ */
(function () {
  'use strict';
  const JT = window.JT;

  /* ---------------- model (pure, no DOM) ---------------- */
  const TENSOR_MB = 64;                        // 4096 * 4096 * 4 bytes
  const TENSOR_BYTES = 4096 * 4096 * 4;
  const HBM_BW = 3.35e12;                      // H100-class HBM3, bytes/s
  const MATMUL_FLOPS = 2 * 4096 * 4096 * 4096; // 2 * N^3
  const TF32_RATE = 495e12;                    // H100-class TF32 tensor-core peak, dense

  const OPS = [
    { id: 'matmul', kind: 'matmul', chip: 'x @ W', label: 'matmul', short: 'mm', expr: (v) => `${v} @ W`,
      tip: '<b>Matrix multiply</b>, a library kernel (cuBLAS or tensor cores on GPU, the MXU on TPU). Reads x and W, compute-bound. Click to drop it from the chain.' },
    { id: 'bias', kind: 'pointwise', chip: '+ b', label: 'bias', short: 'b', expr: (v) => `${v} + b`,
      tip: '<b>Pointwise add</b>: one flop per byte, memory-bound. b is a 4096-vector (16 KB), counted as 0. Click to drop it from the chain.' },
    { id: 'relu', kind: 'pointwise', chip: 'relu', label: 'relu', short: 'relu', expr: (v) => `relu(${v})`,
      tip: '<b>Pointwise relu</b>: one compare per element, memory-bound. Click to drop it from the chain.' },
    { id: 'scale', kind: 'pointwise', chip: '* scale', label: 'scale', short: 's', expr: (v) => `${v} * scale`,
      tip: '<b>Pointwise multiply</b> by a scalar: memory-bound. Click to drop it from the chain.' },
    { id: 'softmax', kind: 'reduce', chip: 'softmax', label: 'softmax', short: 'smax', expr: (v) => `softmax(${v}, axis=-1)`,
      tip: '<b>Row softmax</b>: a reduction over the last axis, a natural fusion boundary. Counted as one read and one write (its internal two passes are ignored). Click to drop it from the chain.' },
  ];

  function kindLabel(ops) {
    if (ops.length === 1) return { matmul: 'matmul kernel', pointwise: 'pointwise kernel', reduce: 'reduction kernel' }[ops[0].kind];
    if (ops.some((o) => o.kind === 'matmul')) return 'matmul with fused epilogue';
    if (ops.some((o) => o.kind === 'reduce')) return 'reduction with fused producers';
    return 'fused pointwise kernel';
  }

  /**
   * plan(enabledIds, mode) -> execution plan.
   * Fusion rules (typical; real compilers differ by backend and version):
   *  - eager: every op is its own kernel.
   *  - compiled: consecutive pointwise ops fuse into one kernel; pointwise ops right after
   *    the matmul fuse into it as an epilogue; pointwise ops right before the softmax fuse
   *    into the reduction. A pointwise run between matmul and softmax goes to the matmul.
   * Traffic model: each kernel reads every 64 MB HBM input once and writes its output once;
   * vectors count as 0; intermediates inside a fused kernel cost 0.
   */
  function plan(enabledIds, mode) {
    const ops = OPS.filter((o) => enabledIds.includes(o.id));
    const n = ops.length;
    const nameOf = (i) => (i < 0 ? 'x' : i === n - 1 ? 'y' : 'h' + (i + 1));
    const groups = [];
    ops.forEach((o, i) => {
      const g = groups[groups.length - 1];
      if (mode === 'compiled' && g) {
        const hasMatmul = g.ops.some((p) => p.kind === 'matmul');
        const hasReduce = g.ops.some((p) => p.kind === 'reduce');
        const fuse = (o.kind === 'pointwise' && !hasReduce) || (o.kind === 'reduce' && !hasMatmul && !hasReduce);
        if (fuse) { g.ops.push(o); g.last = i; return; }
      }
      groups.push({ ops: [o], first: i, last: i });
    });
    const kernels = groups.map((g, k) => {
      const inputs = g.ops[0].kind === 'matmul' ? ['x', 'W'] : [nameOf(g.first - 1)];
      const output = nameOf(g.last);
      const internal = [];
      for (let i = g.first; i < g.last; i++) internal.push(nameOf(i));
      const tensors = inputs.length + 1;
      return {
        index: k, ops: g.ops, inputs, output, internal,
        reads: inputs.length, writes: 1,
        mb: tensors * TENSOR_MB, bytes: tensors * TENSOR_BYTES,
        isIntermediate: k < groups.length - 1,
        kind: kindLabel(g.ops),
      };
    });
    const bytes = kernels.reduce((s, k) => s + k.bytes, 0);
    return {
      mode, ops, kernels,
      launches: kernels.length,
      intermediates: Math.max(0, kernels.length - 1),
      mb: kernels.reduce((s, k) => s + k.mb, 0),
      bytes,
      ms: (bytes / HBM_BW) * 1e3,
      hasMatmul: ops.some((o) => o.kind === 'matmul'),
      matmulMs: (MATMUL_FLOPS / TF32_RATE) * 1e3,
    };
  }

  function codeSrc(ops) {
    const lines = ['# h: [4096, 4096] f32, 64 MB per tensor'];
    let v = 'x';
    ops.forEach((o, i) => { lines.push(`${i === ops.length - 1 ? 'y' : 'h'} = ${o.expr(v)}`); v = 'h'; });
    return lines.join('\n');
  }

  /* ---------------- styles ---------------- */
  const CSS = `
#w-fusion .f-top { display: flex; flex-wrap: wrap; align-items: center; gap: 0.5rem 1rem; margin-bottom: 1rem; }
#w-fusion .f-chips { display: flex; flex-wrap: wrap; align-items: center; gap: 0.4rem; }
#w-fusion .f-chips .f-lbl { font-size: 0.72rem; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase; color: var(--ink-3); margin-right: 0.25rem; }
#w-fusion .chip { font-family: var(--font-mono); font-size: 0.78rem; }
#w-fusion .chip[aria-pressed="false"] { color: var(--ink-3); border-style: dashed; }
#w-fusion .chip:disabled { cursor: default; }
#w-fusion .chip:disabled:hover { border-color: var(--jax-mid); color: var(--jax-deep); }
#w-fusion .f-top .seg { margin-left: auto; }
#w-fusion .seg .f-short { display: none; }
@media (max-width: 640px) { #w-fusion .seg .f-long { display: none; } #w-fusion .seg .f-short { display: inline; } }
#w-fusion .f-main { display: grid; grid-template-columns: minmax(0, 16rem) minmax(0, 1fr); gap: 1.25rem; align-items: start; }
@media (max-width: 760px) { #w-fusion .f-main { grid-template-columns: minmax(0, 1fr); gap: 0.9rem; } }
#w-fusion .f-code { border: 1px solid var(--line); border-radius: var(--radius-sm); overflow: hidden; min-width: 0; }
#w-fusion .f-code-head { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 0.25rem 0.75rem; padding: 0.45rem 0.8rem; font-size: 0.72rem; color: var(--ink-3); border-bottom: 1px solid var(--line); background: var(--surface-2); }
#w-fusion .f-code-head b { font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase; color: var(--ink-3); }
#w-fusion .f-code-head span { font-family: var(--font-mono); }
#w-fusion .f-code pre.code { font-size: 0.76rem; padding: 0.7rem 0.8rem; }
#w-fusion .f-viz { min-width: 0; }
#w-fusion .f-viz svg { width: 100%; height: auto; overflow: visible; }
#w-fusion .f-stats { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 0.6rem; margin-top: 1.1rem; }
@media (max-width: 640px) { #w-fusion .f-stats { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
#w-fusion .f-stat { display: block; padding: 0.6rem 0.75rem; border: 1px solid var(--line); border-radius: var(--radius-sm); background: var(--surface); min-width: 0; }
#w-fusion .f-stat .num { display: block; font-family: var(--font-body); font-size: 1.25rem; font-weight: 600; line-height: 1.2; letter-spacing: -0.01em; color: var(--ink); font-variant-numeric: proportional-nums; white-space: nowrap; }
#w-fusion .f-stat .num small { font-size: 0.78rem; font-weight: 500; color: var(--ink-2); margin-left: 0.2rem; }
#w-fusion[data-mode="eager"] .f-stat .num { color: var(--torch-deep); }
#w-fusion[data-mode="compiled"] .f-stat .num { color: var(--jax-deep); }
#w-fusion .f-stat .lbl { display: block; font-family: var(--font-body); font-size: 0.75rem; color: var(--ink-2); margin-top: 0.1rem; }
#w-fusion .f-stat .cmp { display: block; font-size: 0.7rem; color: var(--ink-3); margin-top: 0.35rem; line-height: 1.35; }
#w-fusion .f-note { margin: 0.75rem 0 0; font-size: 0.8125rem; line-height: 1.5; color: var(--ink-2); }
#w-fusion .f-note b { font-weight: 600; color: var(--ink); }
#w-fusion .stage-foot .f-sentence { flex: 1 1 22rem; }
#w-fusion .legend .sw.eager { background: var(--torch-soft); border: 1px solid var(--torch); }
#w-fusion .legend .sw.fused { background: var(--jax-soft); border: 1px solid var(--jax); }
#w-fusion .legend .sw.hbm { background: var(--surface); border: 1px solid var(--line-2); }

#w-fusion svg .band { fill: var(--surface-2); }
#w-fusion svg .lane { font-size: 11px; font-weight: 600; letter-spacing: 0.06em; fill: var(--ink-3); }
#w-fusion svg .lane-sub { font-size: 11px; font-weight: 400; letter-spacing: 0; fill: var(--ink-3); }
#w-fusion svg .k { transition: opacity 200ms; }
#w-fusion svg .k .kb { fill: var(--torch-soft); stroke: var(--torch-mid); stroke-width: 1.2; transition: fill 220ms, stroke 220ms; }
#w-fusion svg .k .kl { font-size: 12px; font-weight: 600; fill: var(--torch-deep); text-anchor: middle; }
#w-fusion svg .k.active .kb { fill: var(--torch-mid); stroke: var(--torch); }
#w-fusion svg .k.done .kb { stroke: var(--torch); }
#w-fusion svg .k.pending { opacity: 0.45; }
#w-fusion svg.compiled .k .kb { fill: var(--jax-soft); stroke: var(--jax-mid); }
#w-fusion svg.compiled .k .kl { fill: var(--jax-deep); }
#w-fusion svg.compiled .k.active .kb { fill: var(--jax-mid); stroke: var(--jax); }
#w-fusion svg.compiled .k.done .kb { stroke: var(--jax); }
#w-fusion svg .reg rect { fill: var(--surface); fill-opacity: 0.55; stroke: var(--jax-mid); stroke-width: 1; stroke-dasharray: 2.5 2; }
#w-fusion svg .reg text { font-family: var(--font-mono); font-size: 11px; fill: var(--jax-deep); opacity: 0.8; text-anchor: middle; }
#w-fusion svg .reg text.regl { font-family: var(--font-body); fill: var(--ink-3); text-anchor: start; opacity: 1; }
#w-fusion svg .t { transition: opacity 200ms; }
#w-fusion svg .t rect { fill: var(--surface); stroke: var(--line-2); stroke-width: 1.2; transition: stroke 160ms, stroke-width 160ms; }
#w-fusion svg .t text { font-family: var(--font-mono); font-size: 11px; fill: var(--ink); text-anchor: middle; }
#w-fusion svg .t.pending { opacity: 0.35; }
#w-fusion svg .t.pending rect { stroke-dasharray: 3 2.5; }
#w-fusion svg .t.reading rect { stroke: var(--ink-2); }
#w-fusion svg .t.writing rect { stroke: var(--live); stroke-width: 2; }
#w-fusion svg .t.writing text { fill: var(--live-deep); }
#w-fusion svg .arr { fill: none; stroke: var(--ink-3); stroke-width: 1.2; opacity: 0.7; marker-end: url(#f-arr); transition: stroke 140ms, stroke-width 140ms, opacity 140ms; }
#w-fusion svg .arr.dim { opacity: 0.22; }
#w-fusion svg .arr.pulse { stroke: var(--live); stroke-width: 2.2; opacity: 1; marker-end: url(#f-arr-live); }
#w-fusion svg .hint { font-size: 11px; fill: var(--ink-3); }
#w-fusion svg .axis { stroke: var(--line-2); stroke-width: 1; marker-end: url(#f-arr-soft); }
`;

  /* ---------------- SVG timeline ---------------- */
  const tw = (s, f, mono) => s.length * f * (mono ? 0.62 : 0.56);

  function fitLabel(ops, innerW, f) {
    const full = ops.map((o) => o.label).join(' + ');
    if (tw(full, f) <= innerW) return [full];
    if (ops.length > 1) {
      const h = Math.ceil(ops.length / 2);
      const l1 = ops.slice(0, h).map((o) => o.label).join(' + ');
      const l2 = '+ ' + ops.slice(h).map((o) => o.label).join(' + ');
      if (tw(l1, f) <= innerW && tw(l2, f) <= innerW) return [l1, l2];
    }
    const short = ops.map((o) => o.short).join('+');
    if (tw(short, f) <= innerW || ops.length === 1) return [short];
    const h = Math.ceil(ops.length / 2);
    return [ops.slice(0, h).map((o) => o.short).join('+'), '+' + ops.slice(h).map((o) => o.short).join('+')];
  }

  function marker(id, fill) {
    return JT.svg('marker', { id, viewBox: '0 0 10 10', refX: '9', refY: '5', markerWidth: '8', markerHeight: '8', markerUnits: 'userSpaceOnUse', orient: 'auto' },
      JT.svg('path', { d: 'M0 0 L10 5 L0 10 z', fill }));
  }

  function drawTimeline(p, wide) {
    const W = wide ? 720 : 430, padX = wide ? 14 : 10, gap = wide ? 40 : 30, box = wide ? 34 : 24, boxH = 24, F = wide ? 12 : 11;
    const inGap = wide ? 6 : 4; // spacing between the x and W boxes
    const H = 236;
    const hbm = { y: 22, h: 62 }, comp = { y: 138, h: 88 };
    const boxY = 42, boxBot = boxY + boxH;
    const kY = 146, kH = 60;
    const n = p.kernels.length;
    const label = `Timeline in ${p.mode} mode: ${n} kernel launch${n === 1 ? '' : 'es'}, ${p.intermediates} intermediate tensor${p.intermediates === 1 ? '' : 's'} in HBM, ${p.mb} MB moved.`;
    const svg = JT.svg('svg', { viewBox: `0 0 ${W} ${H}`, width: '100%', class: 'f-svg ' + p.mode, role: 'img', 'aria-label': label });
    svg.appendChild(JT.svg('defs', null, [marker('f-arr', JT.colors.ink3), marker('f-arr-live', JT.colors.live), marker('f-arr-soft', JT.colors.line2)]));

    // lanes
    const hbmTip = 'Illustrative traffic model: every kernel reads each 64 MB HBM input once and writes its output once. b and scale are tiny vectors, counted as 0. Softmax counts one read and one write; its two internal passes are ignored.';
    svg.appendChild(JT.svg('rect', { class: 'band', x: padX, y: hbm.y, width: W - 2 * padX, height: hbm.h, rx: 8 }));
    svg.appendChild(JT.svg('rect', { class: 'band', x: padX, y: comp.y, width: W - 2 * padX, height: comp.h, rx: 8 }));
    const laneHbm = JT.svg('text', { class: 'lane', x: padX + 8, y: hbm.y + 14, 'data-tip': hbmTip }, [
      JT.svg('tspan', { text: 'HBM' }), JT.svg('tspan', { class: 'lane-sub', dx: 6, text: wide ? 'device memory, 64 MB per tensor' : '64 MB / tensor' }),
    ]);
    svg.appendChild(laneHbm);
    svg.appendChild(JT.svg('text', { class: 'lane', x: padX + 8, y: comp.y + comp.h - 7 }, [
      JT.svg('tspan', { text: 'COMPUTE' }), JT.svg('tspan', { class: 'lane-sub', dx: 6, text: wide ? 'kernels, in launch order' : 'kernels' }),
    ]));
    // execution-order axis, bottom right of the compute band
    const axW = wide ? 90 : 60;
    svg.appendChild(JT.svg('text', { class: 'hint', x: W - padX - 8 - axW - 6, y: comp.y + comp.h - 7, 'text-anchor': 'end', text: 'execution order' }));
    svg.appendChild(JT.svg('line', { class: 'axis', x1: W - padX - 8 - axW, y1: comp.y + comp.h - 11, x2: W - padX - 8, y2: comp.y + comp.h - 11 }));

    // kernel geometry
    const avail = W - 2 * padX - (n - 1) * gap;
    const wsum = p.kernels.reduce((s, k) => s + k.ops.length + 0.5, 0);
    let x = padX;
    const geo = p.kernels.map((k) => { const w = (avail * (k.ops.length + 0.5)) / wsum; const g = { x0: x, x1: x + w, w }; x += w + gap; return g; });

    // tensors in HBM
    const tensors = {};
    const isInput = (name) => name === 'x' || name === 'W';
    function tensorBox(name, cx, tip) {
      const g = JT.svg('g', { class: 't', dataset: { name }, 'data-tip': tip }, [
        JT.svg('rect', { x: cx - box / 2, y: boxY, width: box, height: boxH, rx: 5 }),
        JT.svg('text', { x: cx, y: boxY + boxH / 2 + 4, text: name }),
      ]);
      svg.appendChild(g);
      tensors[name] = { el: g, cx };
    }
    p.kernels[0].inputs.forEach((name, i) => tensorBox(name, geo[0].x0 + 2 + box / 2 + i * (box + inGap),
      name === 'W' ? '<b>W</b>: weight matrix, 64 MB, read by kernel 1.' : '<b>x</b>: input activation, 64 MB, read by kernel 1.'));
    p.kernels.forEach((k, i) => {
      if (k.isIntermediate) tensorBox(k.output, geo[i].x1 + gap / 2, `<b>${k.output}</b>: intermediate. Written to HBM by kernel ${i + 1} (64 MB) and read straight back by kernel ${i + 2} (64 MB).`);
      else tensorBox(k.output, geo[i].x1 - 2 - box / 2, `<b>${k.output}</b>: final output, 64 MB, written by kernel ${i + 1}.`);
    });

    // kernels and arrows
    const kernelEls = [], arrowsRead = [], arrowsWrite = [];
    p.kernels.forEach((k, i) => {
      const g = geo[i];
      const innerW = g.w - 8;
      const tip = `<b>Kernel ${i + 1} of ${n}</b>: ${k.kind}.<br>Reads ${k.inputs.join(', ')} (${k.reads * TENSOR_MB} MB), writes ${k.output} (${TENSOR_MB} MB).`
        + (k.internal.length ? `<br>${k.internal.join(', ')} never leave registers.` : '')
        + '<br>Typical fusion; real compilers differ by backend and version.';
      const kg = JT.svg('g', { class: 'k', dataset: { kernel: String(i) }, 'data-tip': tip });
      kg.appendChild(JT.svg('rect', { class: 'kb', x: g.x0, y: kY, width: g.w, height: kH, rx: 7 }));
      const lines = fitLabel(k.ops, innerW, F);
      const hasReg = k.internal.length > 0;
      const cx = g.x0 + g.w / 2;
      let baseY;
      if (hasReg) baseY = lines.length === 2 ? kY + 16 : kY + 22;
      else baseY = lines.length === 2 ? kY + kH / 2 - 3 : kY + kH / 2 + 4;
      lines.forEach((ln, j) => kg.appendChild(JT.svg('text', { class: 'kl', x: cx, y: baseY + j * 14, 'font-size': F, text: ln })));
      if (hasReg) {
        const pillW = wide ? 24 : 22, pillH = 16, pg = 4, regY = kY + kH - 8 - pillH;
        const lbl = 'in registers';
        const total = k.internal.length * (pillW + pg) + 6 + tw(lbl, 11);
        const rg = JT.svg('g', { class: 'reg' });
        if (total <= innerW) {
          let px = cx - total / 2;
          k.internal.forEach((name) => {
            rg.appendChild(JT.svg('rect', { x: px, y: regY, width: pillW, height: pillH, rx: 4 }));
            rg.appendChild(JT.svg('text', { x: px + pillW / 2, y: regY + 12, text: name }));
            px += pillW + pg;
          });
          rg.appendChild(JT.svg('text', { class: 'regl', x: px + 4, y: regY + 12, text: lbl }));
        } else {
          rg.appendChild(JT.svg('text', { class: 'regl', x: g.x0 + 6, y: regY + 12, text: `${k.internal.length} in registers` }));
        }
        kg.appendChild(rg);
      }
      svg.appendChild(kg);
      kernelEls.push(kg);

      arrowsRead[i] = k.inputs.map((name) => {
        const t = tensors[name];
        const d = isInput(name) ? `M${t.cx} ${boxBot + 2} L${t.cx} ${kY - 1}` : `M${t.cx + 4} ${boxBot + 2} L${g.x0 + 12} ${kY - 1}`;
        const a = JT.svg('path', { class: 'arr read', d });
        svg.appendChild(a);
        return a;
      });
      const t = tensors[k.output];
      const d = k.isIntermediate ? `M${g.x1 - 12} ${kY} L${t.cx - 4} ${boxBot + 2}` : `M${t.cx} ${kY} L${t.cx} ${boxBot + 2}`;
      const wa = JT.svg('path', { class: 'arr write', d });
      svg.appendChild(wa);
      arrowsWrite[i] = wa;
    });

    if (wide) {
      const k0 = p.kernels[0], g0 = geo[0];
      const tIn = tensors[k0.inputs[0]];
      svg.appendChild(JT.svg('text', { class: 'hint', x: tIn.cx + 6, y: (boxBot + kY) / 2 + 4, text: 'read' }));
      const tOut = tensors[k0.output];
      const wx = k0.isIntermediate ? (g0.x1 - 12 + tOut.cx - 4) / 2 - 5 : tOut.cx - 6;
      svg.appendChild(JT.svg('text', { class: 'hint', x: wx, y: (boxBot + kY) / 2 + 4, 'text-anchor': 'end', text: 'write' }));
    }

    return { svg, kernelEls, tensors, arrowsRead, arrowsWrite, isInput };
  }

  /* ---------------- widget ---------------- */
  function init(container) {
    JT.style('fusion', CSS);
    const stage = JT.stage(container, { title: 'Operator fusion', hint: 'Toggle ops, switch mode, press Run' });
    container.dataset.mode = 'eager';

    const enabled = {}; OPS.forEach((o) => { enabled[o.id] = true; });
    let mode = 'eager';
    let wide = true;
    let runId = 0, running = false, tweenRaf = 0;
    let p = null, other = null, view = null, pre = null, lineOf = {};
    let shown = { launches: 0, intermediates: 0, mb: 0, ms: 0 };

    // controls row
    const chips = OPS.map((o) => {
      const b = JT.el('button', { class: 'chip', type: 'button', 'aria-pressed': 'true', 'data-op': o.id, 'data-tip': o.tip, text: o.chip });
      b.addEventListener('click', () => {
        const onCount = OPS.filter((q) => enabled[q.id]).length;
        if (enabled[o.id] && onCount === 1) return;
        enabled[o.id] = !enabled[o.id];
        b.setAttribute('aria-pressed', String(enabled[o.id]));
        rebuild();
      });
      return b;
    });
    const seg = JT.seg([
      { value: 'eager', label: 'Eager (PyTorch default)' },
      { value: 'compiled', label: 'Compiled (jax.jit / torch.compile)' },
    ], (v) => { mode = v; rebuild(); }, 'eager', 'torch');
    const segMeta = {
      eager: { short: 'Eager', tip: '<b>Eager model</b>: one accelerator kernel per displayed operation, with intermediate tensors materialized in HBM.' },
      compiled: { short: 'Compiled', tip: '<b>Compiled model</b>: XLA or Inductor fuses every compatible operation shown here. Real fusion decisions depend on the program, shapes, and backend.' },
    };
    JT.$$('button', seg).forEach((b) => {
      const m = segMeta[b.dataset.value];
      b.replaceChildren(JT.el('span', { class: 'f-long', text: b.textContent }), JT.el('span', { class: 'f-short', text: m.short }));
      b.setAttribute('data-tip', m.tip);
    });
    const top = JT.el('div', { class: 'f-top' }, [
      JT.el('div', { class: 'f-chips', role: 'group', 'aria-label': 'Ops in the chain' }, [JT.el('span', { class: 'f-lbl', text: 'chain' }), ...chips]),
      seg,
    ]);

    // program + timeline
    const codeWrap = JT.el('div', { class: 'f-code' });
    const codeHead = JT.el('div', { class: 'f-code-head' }, [JT.el('b', { text: 'program' }), JT.el('span', { text: 'h: [4096, 4096] f32' })]);
    codeWrap.appendChild(codeHead);
    const viz = JT.el('div', { class: 'f-viz' });
    const main = JT.el('div', { class: 'f-main' }, [codeWrap, viz]);

    // stats
    const STATS = [
      { key: 'launches', label: 'kernel launches', unit: '', fmt: (v) => String(Math.round(v)), cmp: (v) => `${v} launch${v === 1 ? '' : 'es'}`,
        tip: 'One GPU or TPU kernel per box in the compute lane. Each launch also costs a few microseconds of host and driver overhead, not counted here.' },
      { key: 'intermediates', label: 'intermediates in HBM', unit: '', fmt: (v) => String(Math.round(v)), cmp: (v) => `${v} tensor${v === 1 ? '' : 's'}`,
        tip: 'Tensors written to HBM for use by a later kernel. This model assumes fused intermediates stay in on-chip storage and do not cross HBM.' },
      { key: 'mb', label: 'HBM traffic', unit: 'MB', fmt: (v) => String(Math.round(v)), cmp: (v) => `${v} MB`,
        tip: 'Bytes crossing HBM. Illustrative model: each kernel reads every 64 MB input once and writes its output once; vectors count as 0; softmax\'s two internal passes are ignored.' },
      { key: 'ms', label: 'est. memory time', unit: 'ms', fmt: (v) => v.toFixed(2), cmp: (v) => `${v.toFixed(2)} ms`,
        tip: 'HBM traffic divided by 3.35 TB/s, an H100-class HBM3 peak. A lower bound: real kernels rarely reach peak bandwidth.' },
    ];
    const statEls = {};
    const stats = JT.el('div', { class: 'f-stats' }, STATS.map((s) => {
      const num = JT.el('span', { class: 'num' });
      const cmp = JT.el('span', { class: 'cmp' });
      statEls[s.key] = { num, cmp };
      return JT.el('div', { class: 'readout f-stat', 'data-tip': s.tip }, [num, JT.el('span', { class: 'lbl', text: s.label }), cmp]);
    }));
    const note = JT.el('p', { class: 'f-note' });

    stage.body.append(top, main, stats, note);

    // head control: Run
    const runBtn = JT.el('button', { class: 'btn primary', type: 'button', html: JT.icon('play') + '<span>Run</span>', 'aria-label': 'Run the program' });
    runBtn.addEventListener('click', () => { if (!running) run(); });
    stage.controls.appendChild(runBtn);
    function setRunBtn(state) {
      runBtn.disabled = state === 'running';
      runBtn.innerHTML = JT.icon('play') + `<span>${state === 'running' ? 'Running' : state === 'again' ? 'Run again' : 'Run'}</span>`;
    }

    // foot
    stage.foot.append(
      JT.el('div', { class: 'legend' }, [
        JT.el('span', null, [JT.el('span', { class: 'sw eager' }), 'eager kernel']),
        JT.el('span', null, [JT.el('span', { class: 'sw fused' }), 'fused kernel']),
        JT.el('span', null, [JT.el('span', { class: 'sw hbm' }), 'HBM tensor']),
      ]),
      JT.el('span', { class: 'f-sentence', text: 'Fusion can remove intermediate HBM traffic and launch overhead from compatible memory-bound operations.' }),
    );

    /* ---- rendering ---- */
    function renderStats(v) {
      shown = v;
      STATS.forEach((s) => {
        const el = statEls[s.key].num;
        el.replaceChildren(document.createTextNode(s.fmt(v[s.key])), s.unit ? JT.el('small', { text: s.unit }) : '');
      });
    }
    function renderCompare() {
      STATS.forEach((s) => { statEls[s.key].cmp.textContent = `${other.mode}: ${s.cmp(other[s.key])}`; });
    }
    function renderNote() {
      note.innerHTML = p.hasMatmul
        ? `<b>Simplified estimate:</b> the matmul is 2·4096³ ≈ 137 GFLOP, or <b>${p.matmulMs.toFixed(2)} ms</b> at an assumed 495 TFLOP/s. The model treats the pointwise operations as memory-bound and assumes they fuse into the matmul epilogue.`
        : '<b>Simplified estimate:</b> the model treats every selected pointwise operation as memory-bound and assumes compilation fuses them into one kernel.';
    }
    function tweenStats(from, to, ms) {
      cancelAnimationFrame(tweenRaf);
      if (JT.reducedMotion || ms <= 0) { renderStats(to); return; }
      const t0 = performance.now();
      const step = (now) => {
        const t = Math.min(1, (now - t0) / ms), e = 1 - Math.pow(1 - t, 3);
        const v = {};
        for (const k of Object.keys(to)) v[k] = JT.lerp(from[k], to[k], e);
        renderStats(v);
        if (t < 1) tweenRaf = requestAnimationFrame(step);
      };
      tweenRaf = requestAnimationFrame(step);
    }

    function rebuild() {
      runId++; running = false; cancelAnimationFrame(tweenRaf);
      container.dataset.mode = mode;
      seg.classList.toggle('torch', mode === 'eager');
      seg.classList.toggle('jax', mode === 'compiled');
      const ids = OPS.filter((o) => enabled[o.id]).map((o) => o.id);
      const onCount = ids.length;
      chips.forEach((c) => { c.disabled = onCount === 1 && enabled[c.dataset.op]; });
      p = plan(ids, mode);
      other = plan(ids, mode === 'eager' ? 'compiled' : 'eager');
      // code
      lineOf = {}; p.ops.forEach((o, i) => { lineOf[o.id] = i + 2; });
      const fresh = JT.code(codeSrc(p.ops));
      if (pre) pre.replaceWith(fresh); else codeWrap.appendChild(fresh);
      pre = fresh;
      // timeline
      view = drawTimeline(p, wide);
      viz.replaceChildren(view.svg);
      // numbers
      renderStats({ launches: p.launches, intermediates: p.intermediates, mb: p.mb, ms: p.ms });
      renderCompare();
      renderNote();
      setRunBtn('idle');
    }

    async function run() {
      const my = ++runId;
      running = true; setRunBtn('running');
      const v = view, plan0 = p;
      const allArrows = [...v.arrowsWrite, ...v.arrowsRead.flat()];
      v.kernelEls.forEach((el) => { el.classList.remove('active', 'done'); el.classList.add('pending'); });
      Object.entries(v.tensors).forEach(([name, t]) => { t.el.classList.remove('writing', 'reading'); t.el.classList.toggle('pending', !v.isInput(name)); });
      allArrows.forEach((a) => { a.classList.remove('pulse'); a.classList.add('dim'); });
      JT.clearLines(pre, 'now');
      let acc = { launches: 0, intermediates: 0, mb: 0, ms: 0 };
      renderStats(acc);
      try {
        for (let i = 0; i < plan0.kernels.length; i++) {
          const k = plan0.kernels[i], kel = v.kernelEls[i];
          kel.classList.remove('pending'); kel.classList.add('active');
          JT.clearLines(pre, 'now'); JT.markLines(pre, k.ops.map((o) => lineOf[o.id]), 'now');
          v.arrowsRead[i].forEach((a) => { a.classList.remove('dim'); a.classList.add('pulse'); });
          k.inputs.forEach((nm) => v.tensors[nm].el.classList.add('reading'));
          await JT.wait(230); if (my !== runId) return;
          v.arrowsRead[i].forEach((a) => a.classList.remove('pulse'));
          k.inputs.forEach((nm) => v.tensors[nm].el.classList.remove('reading'));
          const wa = v.arrowsWrite[i]; wa.classList.remove('dim'); wa.classList.add('pulse');
          const out = v.tensors[k.output].el; out.classList.remove('pending'); out.classList.add('writing');
          acc = { launches: acc.launches + 1, intermediates: acc.intermediates + (k.isIntermediate ? 1 : 0), mb: acc.mb + k.mb, ms: acc.ms + (k.bytes / HBM_BW) * 1e3 };
          tweenStats(shown, acc, 260);
          await JT.wait(270); if (my !== runId) return;
          wa.classList.remove('pulse'); out.classList.remove('writing');
          kel.classList.remove('active'); kel.classList.add('done');
        }
        JT.clearLines(pre, 'now');
        cancelAnimationFrame(tweenRaf);
        renderStats({ launches: plan0.launches, intermediates: plan0.intermediates, mb: plan0.mb, ms: plan0.ms });
      } finally {
        if (my === runId) { running = false; setRunBtn('again'); }
      }
    }

    // responsive: narrow layout under ~520px of timeline width
    const measure = () => { const w = viz.getBoundingClientRect().width; return w === 0 ? true : w >= 520; };
    wide = measure();
    rebuild();
    if ('ResizeObserver' in window) {
      new ResizeObserver(() => { const nw = measure(); if (nw !== wide) { wide = nw; rebuild(); } }).observe(viz);
    } else {
      window.addEventListener('resize', () => { const nw = measure(); if (nw !== wide) { wide = nw; rebuild(); } });
    }
    JT.bindTips(container);
  }

  init.model = { OPS, plan, TENSOR_MB, TENSOR_BYTES, HBM_BW };
  JT.widget('fusion', init);
})();
