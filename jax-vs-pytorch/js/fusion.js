/* ------------------------------------------------------------------
   Figure: operator fusion as two small multiples on one scale.
   A chain of ops on a [4096, 4096] float32 tensor runs eagerly (one
   kernel per op, every intermediate written to HBM) and compiled (fusible
   ops grouped into one kernel). Each op keeps the same horizontal slot in
   both drawings, so the reader can see which HBM writes disappear.
   ------------------------------------------------------------------ */
(function () {
  'use strict';
  const JT = window.JT;

  /* ---------------- model (pure, no DOM; tested in node) ---------------- */
  const TENSOR_MB = 64;                        // 4096 * 4096 * 4 bytes
  const TENSOR_BYTES = 4096 * 4096 * 4;
  const HBM_BW = 3.35e12;                      // H100 SXM HBM3 peak, bytes/s (vendor, approximate)
  const MATMUL_FLOPS = 2 * 4096 * 4096 * 4096; // 2 * N^3
  const TF32_RATE = 495e12;                    // H100 SXM dense TF32 tensor-core peak (vendor, approximate)

  const OPS = [
    { id: 'matmul', kind: 'matmul', chip: 'x @ W', label: 'matmul' },
    { id: 'bias', kind: 'pointwise', chip: '+ b', label: 'bias' },
    { id: 'relu', kind: 'pointwise', chip: 'relu', label: 'relu' },
    { id: 'scale', kind: 'pointwise', chip: '* scale', label: 'scale' },
    { id: 'softmax', kind: 'reduce', chip: 'softmax', label: 'softmax' },
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
        index: k, ops: g.ops, first: g.first, last: g.last, inputs, output, internal,
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

  /* ---------------- drawing, at real CSS pixels ---------------- */
  const MIN_W = 560;
  const G = 62, TB = 26, TH = 20, TG = 36, HB_Y = 8, K_Y = 62, K_H = 36, H = 104;

  function layout(n, W) {
    const s = (W - G - 70 - 12 - TB - 4 - (n - 1) * TG) / n;
    const x0 = G + 70;
    return { s, slotX: (i) => x0 + i * (s + TG), gapCx: (i) => x0 + i * (s + TG) + s + TG / 2 };
  }

  function arrow(svg, x1, y1, x2, y2) {
    const dx = x2 - x1, dy = y2 - y1, len = Math.hypot(dx, dy) || 1, ux = dx / len, uy = dy / len;
    const hl = 5, hw = 2.6, bx = x2 - ux * hl, by = y2 - uy * hl;
    svg.appendChild(JT.svg('line', { class: 'arr', x1, y1, x2: bx, y2: by }));
    svg.appendChild(JT.svg('polygon', { class: 'arrhead', points: `${x2},${y2} ${bx - uy * hw},${by + ux * hw} ${bx + uy * hw},${by - ux * hw}` }));
  }

  function draw(p, W) {
    const n = p.ops.length;
    const L = layout(n, W);
    const svg = JT.svg('svg', { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: 'img',
      'aria-label': `${p.mode === 'eager' ? 'Eager' : 'Compiled'}: ${p.launches} kernel launch${p.launches === 1 ? '' : 'es'}, ${p.intermediates} intermediate tensor${p.intermediates === 1 ? '' : 's'} written to HBM, ${p.mb} MB of HBM traffic.` });
    svg.appendChild(JT.svg('text', { class: 'label muted', x: 0, y: HB_Y + TH / 2 + 4, text: 'HBM' }));
    svg.appendChild(JT.svg('text', { class: 'label muted', x: 0, y: K_Y + K_H / 2 + 4, text: 'kernels' }));

    const tensor = (name, x, ghost) => {
      svg.appendChild(JT.svg('rect', { class: 'tbox', x, y: HB_Y, width: TB, height: TH, rx: 2, style: ghost ? { stroke: 'var(--rule-2)', strokeDasharray: '3 3' } : null }));
      if (!ghost) svg.appendChild(JT.svg('text', { class: 'tlab', x: x + TB / 2, y: HB_Y + TH / 2 + 4, 'text-anchor': 'middle', text: name }));
      return { cx: x + TB / 2 };
    };

    // inputs
    const k0 = p.kernels[0];
    const inputs = k0.inputs.map((name, i) => ({ name, t: tensor(name, G + i * (TB + 6), false) }));

    p.kernels.forEach((k, ki) => {
      const x0 = L.slotX(k.first), x1 = L.slotX(k.last) + L.s;
      svg.appendChild(JT.svg('rect', { class: 'kbox', x: x0, y: K_Y, width: x1 - x0, height: K_H, rx: 2 }));
      for (let i = k.first; i <= k.last; i++) {
        svg.appendChild(JT.svg('text', { class: 'label', x: L.slotX(i) + L.s / 2, y: K_Y + K_H / 2 + 4, 'text-anchor': 'middle', text: p.ops[i].label }));
        if (i < k.last) {
          const gx = L.gapCx(i);
          svg.appendChild(JT.svg('line', { class: 'kdiv', x1: gx, x2: gx, y1: K_Y + 6, y2: K_Y + K_H - 6 }));
          tensor('', gx - TB / 2, true); // kept on chip: a dashed outline where the HBM tensor would be
        }
      }
      // reads
      if (ki === 0) inputs.forEach((inp, i) => arrow(svg, inp.t.cx, HB_Y + TH + 1, x0 + 8 + i * 10, K_Y - 1));
      else { const cx = L.gapCx(k.first - 1); arrow(svg, cx + 4, HB_Y + TH + 1, x0 + 10, K_Y - 1); }
      // write
      if (k.isIntermediate) {
        const cx = L.gapCx(k.last);
        tensor(k.output, cx - TB / 2, false);
        arrow(svg, x1 - 10, K_Y - 1, cx - 4, HB_Y + TH + 1);
      } else {
        const tx = x1 + 12;
        const t = tensor(k.output, tx, false);
        arrow(svg, x1 - 6, K_Y - 1, t.cx, HB_Y + TH + 1);
      }
    });
    return svg;
  }

  /* ---------------- widget ---------------- */
  function init(container) {
    container.classList.add('fusion');
    const enabled = {}; OPS.forEach((o) => { enabled[o.id] = true; });
    let width = 0;

    const toggles = OPS.map((o) => {
      const b = JT.button(o.chip, () => {
        const onCount = OPS.filter((q) => enabled[q.id]).length;
        if (enabled[o.id] && onCount === 1) return;
        enabled[o.id] = !enabled[o.id];
        render();
      }, { class: 'btn mono toggle op', 'aria-pressed': 'true', dataset: { op: o.id } });
      return b;
    });
    container.appendChild(JT.el('div', { class: 'controls tight' }, [JT.el('span', { class: 'lbl', text: 'Operations in the chain' }), ...toggles]));

    const mk = (mode, title) => {
      const readline = JT.el('p', { class: 'readline', 'aria-live': 'polite' });
      const host = JT.el('div', { style: { overflowX: 'auto' } });
      const el = JT.el('div', { class: 'multiple ' + mode, dataset: { mode } }, [JT.el('h4', { html: title }), readline, host]);
      container.appendChild(el);
      return { mode, readline, host };
    };
    const views = [
      mk('eager', '<span class="torch-c">Eager</span> <span class="sub">one kernel per operation, as in PyTorch by default</span>'),
      mk('compiled', '<span class="jax-c">Compiled</span> <span class="sub">fused by XLA under jax.jit, or by Inductor under torch.compile</span>'),
    ];

    function render() {
      const ids = OPS.filter((o) => enabled[o.id]).map((o) => o.id);
      toggles.forEach((b) => {
        b.setAttribute('aria-pressed', String(enabled[b.dataset.op]));
        b.disabled = ids.length === 1 && enabled[b.dataset.op];
      });
      const W = Math.max(width || MIN_W, MIN_W);
      views.forEach((v) => {
        const p = plan(ids, v.mode);
        v.readline.innerHTML = `<b>${p.launches}</b> kernel launch${p.launches === 1 ? '' : 'es'} · <b>${p.intermediates}</b> intermediate${p.intermediates === 1 ? '' : 's'} written to HBM · <b>${p.mb} MB</b> of HBM traffic · at least <b>${p.ms.toFixed(2)} ms</b>`;
        v.host.replaceChildren(draw(p, W));
      });
    }
    JT.onWidth(container, (w) => { width = w; render(); });
  }

  init.model = { OPS, plan, TENSOR_MB, TENSOR_BYTES, HBM_BW };
  JT.widget('fusion', init);
})();
