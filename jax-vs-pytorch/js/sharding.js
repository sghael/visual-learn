/* ------------------------------------------------------------------
   Figure: annotate the two operands of y = x @ w with PartitionSpecs on a
   2 x 4 device mesh and see which collectives a deliberately simplified
   SPMD partitioner inserts. Each block of an array is labeled with the
   devices that hold it; color is reserved for the collectives (green,
   the page's "active" color). Registers JT.widget('sharding').
   ------------------------------------------------------------------ */
(function () {
  'use strict';
  const JT = window.JT;

  /* @model-start */
  /* Mesh: 2 x 4, axes ('data', 'model'); device id = 4 * data + model. */
  const AXES = { data: 2, model: 4 };
  const DEVICES = [0, 1, 2, 3, 4, 5, 6, 7];
  const coord = { data: (d) => d >> 2, model: (d) => d & 3 };

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
      notes.push("y cannot use '" + bx + "' on both of its dimensions, so w is gathered along F first");
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
    { id: 'ddp', label: 'Data parallel', x: ['data', null], w: [null, null], torch: 'DistributedDataParallel replicates w on every rank, splits the batch, and all-reduces gradients in the backward pass.' },
    { id: 'coltp', label: 'Column parallel', x: [null, null], w: [null, 'model'], torch: 'a column-parallel Linear layer (Megatron-style): each rank owns a slice of the output features.' },
    { id: 'rowtp', label: 'Row parallel', x: [null, 'model'], w: ['model', null], torch: 'a row-parallel Linear layer (Megatron-style): partial sums are added with an NCCL all-reduce.' },
    { id: 'fsdp', label: 'FSDP-like', x: ['data', null], w: ['data', null], torch: 'FullyShardedDataParallel: parameters are sharded and all-gathered right before use.' },
    { id: '2d', label: 'Data × model', x: ['data', null], w: [null, 'model'], torch: 'DTensor on a 2-D mesh: Shard(0) on the data dimension for x, Shard(1) on the model dimension for w.' },
  ];
  const stepStr = (s) => (s.type === 'all-gather' ? 'all-gather ' + s.of + " over '" + s.over + "'" : "all-reduce over '" + s.over + "'");

  /* ---------------------------------------------------------------- */
  /* Drawing, at real CSS pixels                                        */
  /* ---------------------------------------------------------------- */
  function devLabel(devs) {
    if (devs.length === 8) return 'all 8';
    if (devs.length > 2 && devs.every((d, i) => i === 0 || d === devs[i - 1] + 1)) return devs[0] + '–' + devs[devs.length - 1];
    return devs.join(' ');
  }

  /** The 2 x 4 mesh. groups: null, 'model' (rows communicate) or 'data' (columns communicate). */
  function drawMesh(o) {
    const cell = o.cell, gap = o.gap, ml = o.axes ? 58 : 4, mt = o.axes ? 34 : 4;
    const W = ml + 4 * cell + 3 * gap + 4, H = mt + 2 * cell + gap + 4;
    const svg = JT.svg('svg', { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': o.label });
    const at = (d) => ({ x: ml + coord.model(d) * (cell + gap), y: mt + coord.data(d) * (cell + gap) });
    if (o.groups) {
      const groups = o.groups === 'model' ? [[0, 1, 2, 3], [4, 5, 6, 7]] : [[0, 4], [1, 5], [2, 6], [3, 7]];
      for (const g of groups) {
        const a = at(g[0]), b = at(g[g.length - 1]);
        svg.appendChild(JT.svg('rect', { class: 'grp', x: a.x - 2.5, y: a.y - 2.5, width: b.x - a.x + cell + 5, height: b.y - a.y + cell + 5, rx: 2 }));
      }
    }
    for (const d of DEVICES) {
      const p = at(d);
      svg.appendChild(JT.svg('rect', { class: 'dev', x: p.x, y: p.y, width: cell, height: cell, rx: 1.5 }));
      if (o.ids) svg.appendChild(JT.svg('text', { class: 'did', x: p.x + cell / 2, y: p.y + cell / 2 + 4.5, 'text-anchor': 'middle', text: String(d) }));
    }
    if (o.axes) {
      svg.appendChild(JT.svg('text', { class: 'ax', x: ml, y: 12, text: "'model' axis (4) →" }));
      for (let m = 0; m < 4; m++) svg.appendChild(JT.svg('text', { class: 'ax', x: ml + m * (cell + gap) + cell / 2, y: mt - 6, 'text-anchor': 'middle', text: String(m) }));
      svg.appendChild(JT.svg('text', { class: 'ax', x: 0, y: mt + cell / 2 + 4, text: "'data'" }));
      svg.appendChild(JT.svg('text', { class: 'ax', x: 0, y: mt + cell / 2 + 19, text: 'axis (2)' }));
      for (let r = 0; r < 2; r++) svg.appendChild(JT.svg('text', { class: 'ax', x: ml - 8, y: mt + r * (cell + gap) + cell / 2 + 4, 'text-anchor': 'end', text: String(r) }));
    }
    return svg;
  }

  /** x @ w = y as three squares split into labeled blocks. */
  function drawArrays(specs, A) {
    const opW = 24, top = 34;
    const W = 3 * A + 2 * opW, H = top + A + 2;
    const svg = JT.svg('svg', { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: 'img',
      'aria-label': 'x ' + specStr(specs[0].spec) + ' times w ' + specStr(specs[1].spec) + ' gives y ' + specStr(specs[2].spec) });
    specs.forEach((a, idx) => {
      const x0 = idx * (A + opW);
      svg.appendChild(JT.svg('text', { class: 'aname', x: x0, y: 12, text: a.name + ' ' + a.dims }));
      svg.appendChild(JT.svg('text', { class: 'aspec', x: x0, y: 27, text: specStr(a.spec) }));
      svg.appendChild(JT.svg('rect', { class: 'arr-out', x: x0, y: top, width: A, height: A }));
      const nR = a.spec[0] ? AXES[a.spec[0]] : 1, nC = a.spec[1] ? AXES[a.spec[1]] : 1;
      const bh = A / nR, bw = A / nC;
      for (let i = 0; i < nR; i++) for (let j = 0; j < nC; j++) {
        const devs = DEVICES.filter((d) => (!a.spec[0] || coord[a.spec[0]](d) === i) && (!a.spec[1] || coord[a.spec[1]](d) === j));
        const bx = x0 + j * bw, by = top + i * bh;
        svg.appendChild(JT.svg('rect', { class: 'blk', x: bx, y: by, width: bw, height: bh }));
        svg.appendChild(JT.svg('text', { class: 'blab', x: bx + bw / 2, y: by + bh / 2 + 4, 'text-anchor': 'middle', text: devLabel(devs) }));
      }
      if (idx < 2) svg.appendChild(JT.svg('text', { class: 'op', x: x0 + A + opW / 2, y: top + A / 2 + 5, 'text-anchor': 'middle', text: idx === 0 ? '@' : '=' }));
    });
    return svg;
  }

  /* ---------------------------------------------------------------- */
  /* Widget                                                             */
  /* ---------------------------------------------------------------- */
  JT.widget('sharding', (container) => {
    container.classList.add('sharding');
    let xSpec = ['data', null], wSpec = [null, null], width = 0;

    const sel = (opts, label, onChange) => {
      const s = JT.el('select', { 'aria-label': label });
      opts.forEach((o) => s.appendChild(JT.el('option', { value: o.value, text: o.label })));
      s.addEventListener('change', () => onChange(s.value));
      return s;
    };
    const xSel = sel(X_OPTS, 'PartitionSpec for x', (v) => { xSpec = unkey(v); render(); });
    const wSel = sel(W_OPTS, 'PartitionSpec for w', (v) => { wSpec = unkey(v); render(); });
    const presetBtns = PRESETS.map((p) => JT.button(p.label, () => { xSpec = p.x.slice(); wSpec = p.w.slice(); render(); }, { class: 'btn toggle', 'aria-pressed': 'false', dataset: { preset: p.id } }));
    container.append(
      JT.el('div', { class: 'controls tight' }, [JT.el('span', { class: 'lbl', text: 'Layout' }), ...presetBtns]),
      JT.el('div', { class: 'controls' }, [JT.control('x', [xSel]), JT.control('w', [wSel])]),
    );

    const meshHost = JT.el('div');
    const arraysHost = JT.el('div');
    const readline = JT.el('p', { class: 'readline' });
    const coll = JT.el('ul', { class: 'coll', 'aria-live': 'polite' });
    const codeHost = JT.el('div');
    const torchEq = JT.el('p', { class: 'torch-eq' });
    container.append(
      JT.el('div', { class: 'top' }, [meshHost, arraysHost]),
      readline,
      JT.el('div', { class: 'cols' }, [
        JT.el('div', {}, [JT.el('h4', { text: 'Collectives the compiler inserts' }), coll]),
        JT.el('div', {}, [JT.el('h4', { html: '<span class="jax-c">JAX</span> <span class="sub">annotate the data, jit the math</span>' }), codeHost, torchEq]),
      ]),
    );
    meshHost.appendChild(drawMesh({ cell: 30, gap: 6, axes: true, ids: true, label: 'Device mesh: 2 rows along the data axis by 4 columns along the model axis; device id = 4 × data + model' }));

    function render() {
      const p = plan(xSpec, wSpec);
      xSel.value = key(xSpec); wSel.value = key(wSpec);

      const meshW = 208, avail = width >= 620 ? width - meshW - 32 : width;
      const A = Math.max(84, Math.min(132, Math.floor((avail - 48) / 3)));
      arraysHost.replaceChildren(drawArrays([
        { name: 'x', dims: '[B, D]', spec: xSpec },
        { name: 'w', dims: '[D, F]', spec: wSpec },
        { name: 'y', dims: '[B, F]', spec: p.y },
      ], A));
      readline.innerHTML = `Each device holds <b>1/${p.mem.x}</b> of x, <b>1/${p.mem.w}</b> of w and <b>1/${p.mem.y}</b> of y. y comes out ${p.y[0] || p.y[1] ? 'sharded' : 'replicated'} as <code>${JT.escape(specStr(p.y))}</code>.`;

      coll.replaceChildren();
      if (!p.steps.length) coll.appendChild(JT.el('li', { class: 'none', text: 'None. Every device already holds the operands it needs.' }));
      p.steps.forEach((s) => {
        const n = AXES[s.over], groups = 8 / n;
        const mini = drawMesh({ cell: 14, gap: 5, groups: s.over, label: `${s.type} over '${s.over}': ${groups} groups of ${n} devices` });
        const what = s.type === 'all-gather'
          ? `Before the matmul. ${groups} groups of ${n} devices; each group concatenates its slices of ${s.of}, so every member holds ${s.of} whole along that axis.`
          : `After the matmul. ${groups} groups of ${n} devices; each device holds a partial y from its slice of D, and each group sums them. XLA may choose a reduce-scatter instead when y should stay sharded.`;
        coll.appendChild(JT.el('li', {}, [mini, JT.el('span', { html: `<b>${JT.escape(stepStr(s))}</b>. ${what}` })]));
      });
      if (p.notes.length) coll.appendChild(JT.el('li', { text: 'Note: ' + p.notes.join('; ') + '.' }));

      const lines = [
        'from jax.sharding import NamedSharding',
        'from jax.sharding import PartitionSpec as P',
        "mesh = jax.make_mesh((2, 4), ('data', 'model'))",
        'shard = lambda spec: NamedSharding(mesh, spec)',
        'x = jax.device_put(x, shard(' + specStr(xSpec) + '))',
        'w = jax.device_put(w, shard(' + specStr(wSpec) + '))',
        'y = jax.jit(lambda x, w: x @ w)(x, w)',
        'print(y.sharding.spec)   # ' + specStr(p.y),
      ];
      const first = lines.length + 1;
      if (!p.steps.length) lines.push('# XLA inserts no collectives');
      else p.steps.forEach((s, i) => lines.push((i === 0 ? '# XLA inserts: ' : '#   then ') + stepStr(s)));
      const pre = JT.code(lines.join('\n'));
      JT.markLines(pre, lines.map((_, i) => i + 1).filter((n) => n >= first), 'now');
      codeHost.replaceChildren(pre);

      const active = PRESETS.find((q) => key(q.x) === key(xSpec) && key(q.w) === key(wSpec));
      presetBtns.forEach((b) => b.setAttribute('aria-pressed', String(!!active && b.dataset.preset === active.id)));
      torchEq.innerHTML = '<b>PyTorch equivalent</b>: ' + (active ? active.torch : 'a custom DTensor placement, written out with Shard and Replicate by hand.');
    }
    JT.onWidth(container, (w) => { width = w; render(); });
  });
})();
