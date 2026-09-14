/* Autodiff widget: the same tiny loss differentiated by PyTorch's tape
   and by JAX's grad transformation. A shared slider sets w. */
(function () {
  'use strict';
  const JT = window.JT;

  JT.style('autodiff', `
    #w-autodiff .formula { display: flex; flex-wrap: wrap; align-items: center; gap: 0.5rem 1.5rem; padding: 0 0 1rem; font-size: 0.9rem; color: var(--ink-2); }
    #w-autodiff .formula .f { font-family: var(--font-mono); font-size: 0.85rem; color: var(--ink); }
    #w-autodiff .formula .vals { font-family: var(--font-mono); font-size: 0.8rem; font-variant-numeric: tabular-nums; }
    #w-autodiff .formula .vals b { font-weight: 600; color: var(--ink); }
    #w-autodiff .slider { display: inline-flex; align-items: center; gap: 0.6rem; font-size: 0.8125rem; }
    #w-autodiff .slider input { width: 140px; accent-color: var(--ink); }
    #w-autodiff .slider output { font-family: var(--font-mono); font-size: 0.8rem; min-width: 5.5ch; font-variant-numeric: tabular-nums; }
    #w-autodiff .pane-body { padding: 0; }
    #w-autodiff .graph { padding: 0.75rem 0.75rem 0.25rem; border-bottom: 1px solid var(--line); }
    #w-autodiff .node circle { fill: var(--surface); stroke: var(--line-2); stroke-width: 1.2; transition: fill 200ms, stroke 200ms; }
    #w-autodiff .node text.op { font-size: 11px; font-weight: 600; fill: var(--ink-2); text-anchor: middle; dominant-baseline: central; }
    #w-autodiff .node text.val { font-family: var(--font-mono); font-size: 10px; fill: var(--ink-3); text-anchor: middle; }
    #w-autodiff .node.const circle { fill: var(--surface-2); stroke: var(--line); }
    #w-autodiff .node.fwd circle { fill: var(--live-soft); stroke: var(--live); }
    #w-autodiff .node.fwd text.op { fill: var(--live-deep); }
    #w-autodiff .pane.torch .node.done circle { fill: var(--torch-soft); stroke: var(--torch-mid); }
    #w-autodiff .pane.jax .node.done circle { fill: var(--jax-soft); stroke: var(--jax-mid); }
    #w-autodiff .node.bwd circle { fill: var(--ink); stroke: var(--ink); }
    #w-autodiff .node.bwd text.op { fill: #fff; }
    #w-autodiff .edge { stroke: var(--line-2); stroke-width: 1.2; fill: none; }
    #w-autodiff .edge.on { stroke: var(--ink); }
    #w-autodiff .elab { font-family: var(--font-mono); font-size: 9.5px; fill: var(--ink-3); text-anchor: middle; opacity: 0; transition: opacity 200ms; }
    #w-autodiff .elab.show { opacity: 1; }
    #w-autodiff .pane.torch .elab.show { fill: var(--torch-deep); }
    #w-autodiff .pane.jax .elab.show { fill: var(--jax-deep); }
    #w-autodiff .ctok { fill: var(--ink); opacity: 0; transition: opacity 200ms; }
    #w-autodiff .ctok.show { opacity: 1; }
    #w-autodiff .ctok text { fill: #fff; font-size: 9px; font-family: var(--font-mono); text-anchor: middle; dominant-baseline: central; }
    #w-autodiff pre.code { font-size: 0.76rem; }
    #w-autodiff .tape { border-top: 1px solid var(--line); padding: 0.6rem 0.9rem; min-height: 8.6rem; }
    #w-autodiff .tape h6 { margin: 0 0 0.4rem; font-size: 0.72rem; font-weight: 600; letter-spacing: 0.03em; text-transform: uppercase; color: var(--ink-3); }
    #w-autodiff .tape ol { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column-reverse; gap: 4px; }
    #w-autodiff .tape li { font-family: var(--font-mono); font-size: 0.74rem; padding: 0.3rem 0.55rem; border: 1px solid var(--line-2); border-radius: 6px; background: var(--surface); display: flex; justify-content: space-between; gap: 0.5rem; transition: background 200ms, border-color 200ms, opacity 200ms; animation: ad-in 300ms var(--ease-out); }
    @keyframes ad-in { from { opacity: 0; transform: translateY(-4px); } }
    #w-autodiff .tape li .g { color: var(--ink-3); }
    #w-autodiff .tape li.active { background: var(--torch-soft); border-color: var(--torch); }
    #w-autodiff .tape li.active .g { color: var(--torch-deep); font-weight: 600; }
    #w-autodiff .tape li.popped { opacity: 0.35; text-decoration: line-through; }
    #w-autodiff .tape .empty { font-size: 0.78rem; color: var(--ink-3); font-family: var(--font-body); }
    #w-autodiff .tape .gradbox { margin-top: 0.5rem; font-family: var(--font-mono); font-size: 0.76rem; color: var(--ink-2); }
    #w-autodiff .tape .gradbox b { color: var(--torch-deep); }
    #w-autodiff .jaxpr { border-top: 1px solid var(--line); min-height: 8.6rem; }
    #w-autodiff .jaxpr .ln.hl { background: var(--jax-soft); }
    #w-autodiff .jaxpr .ln.now { background: var(--live-soft); box-shadow: inset 2px 0 0 var(--live); }
    #w-autodiff .note { padding: 0.6rem 0.9rem; font-size: 0.83rem; color: var(--ink-2); border-top: 1px solid var(--line); min-height: 4.4rem; line-height: 1.5; }
    #w-autodiff .stepbar { padding: 0.6rem 0.9rem; border-top: 1px solid var(--line); background: var(--surface-2); }
    #w-autodiff .stepbar .controls { margin-left: 0; }
    #w-autodiff .two-col { align-items: start; }
  `);

  const X = 1.5, Y = 0.5;
  const f3 = (v) => (Math.round(v * 1000) / 1000).toFixed(3);
  function values(w) {
    const a = w * X, b = Math.sin(a), c = b - Y, L = c * c;
    const dc = 2 * c, db = dc, da = db * Math.cos(a), dw = da * X;
    return { w, a, b, c, L, dc, db, da, dw, cosa: Math.cos(a) };
  }

  /* graph: w → (mul) → (sin) → (sub) → (sq) → L, with constants x, y above */
  function graph() {
    const svg = JT.svg('svg', { viewBox: '0 0 360 150', role: 'img', 'aria-label': 'Computation graph of the loss' });
    const NX = { w: 28, mul: 96, sin: 164, sub: 232, sq: 300, L: 346 };
    const NY = 78, CY = 28;
    const nodes = {}, edges = {}, labels = {}, toks = {};
    function edge(id, x1, y1, x2, y2) { const p = JT.svg('path', { class: 'edge', d: `M${x1},${y1} L${x2},${y2}` }); svg.appendChild(p); edges[id] = p; }
    edge('w-mul', NX.w + 14, NY, NX.mul - 18, NY);
    edge('mul-sin', NX.mul + 18, NY, NX.sin - 18, NY);
    edge('sin-sub', NX.sin + 18, NY, NX.sub - 18, NY);
    edge('sub-sq', NX.sub + 18, NY, NX.sq - 18, NY);
    edge('sq-L', NX.sq + 18, NY, NX.L - 12, NY);
    edge('x-mul', NX.mul, CY + 12, NX.mul, NY - 18);
    edge('y-sub', NX.sub, CY + 12, NX.sub, NY - 18);
    function node(id, x, y, r, op, cls) {
      const g = JT.svg('g', { class: 'node ' + (cls || ''), 'data-node': id });
      g.appendChild(JT.svg('circle', { cx: x, cy: y, r }));
      g.appendChild(JT.svg('text', { class: 'op', x, y, text: op }));
      const val = JT.svg('text', { class: 'val', x, y: y + r + 12, text: '' });
      g.appendChild(val);
      svg.appendChild(g); nodes[id] = { g, val, x, y };
    }
    node('w', NX.w, NY, 14, 'w', 'input');
    node('x', NX.mul, CY, 12, 'x', 'const'); nodes.x.val.setAttribute('y', CY - 15); nodes.x.val.textContent = X;
    node('y', NX.sub, CY, 12, 'y', 'const'); nodes.y.val.setAttribute('y', CY - 15); nodes.y.val.textContent = Y;
    node('mul', NX.mul, NY, 18, '×');
    node('sin', NX.sin, NY, 18, 'sin');
    node('sub', NX.sub, NY, 18, '−');
    node('sq', NX.sq, NY, 18, '²');
    node('L', NX.L, NY, 12, 'L', 'input');
    // edge labels (local derivatives) above the horizontal edges, cotangent tokens below
    [['w-mul', (NX.w + NX.mul) / 2], ['mul-sin', (NX.mul + NX.sin) / 2], ['sin-sub', (NX.sin + NX.sub) / 2], ['sub-sq', (NX.sub + NX.sq) / 2], ['sq-L', (NX.sq + NX.L) / 2]].forEach(([id, x]) => {
      const t = JT.svg('text', { class: 'elab', x, y: NY - 26, text: '' }); svg.appendChild(t); labels[id] = t;
      const g = JT.svg('g', { class: 'ctok' }, [JT.svg('rect', { x: x - 20, y: NY + 38, width: 40, height: 14, rx: 7 }), JT.svg('text', { x, y: NY + 45, text: '' })]);
      svg.appendChild(g); toks[id] = g;
    });
    return {
      svg,
      setValues(v) {
        nodes.w.val.textContent = f3(v.w); nodes.mul.val.textContent = f3(v.a); nodes.sin.val.textContent = f3(v.b);
        nodes.sub.val.textContent = f3(v.c); nodes.sq.val.textContent = f3(v.L); nodes.L.val.textContent = '';
      },
      mark(ids, cls) { Object.values(nodes).forEach((n) => n.g.classList.remove('fwd', 'bwd')); ids.forEach((id) => nodes[id].g.classList.add(cls)); },
      done(ids) { ids.forEach((id) => nodes[id].g.classList.add('done')); },
      label(id, text) { labels[id].textContent = text; labels[id].classList.toggle('show', !!text); },
      tok(id, text) { const g = toks[id]; g.querySelector('text').textContent = text; g.classList.toggle('show', !!text); },
      edge(id, on) { if (edges[id]) edges[id].classList.toggle('on', on); },
      reset() {
        Object.values(nodes).forEach((n) => n.g.classList.remove('fwd', 'bwd', 'done'));
        Object.values(labels).forEach((l) => { l.textContent = ''; l.classList.remove('show'); });
        Object.values(toks).forEach((g) => g.classList.remove('show'));
        Object.values(edges).forEach((e) => e.classList.remove('on'));
      },
    };
  }

  JT.widget('autodiff', (container) => {
    const S = JT.stage(container, { title: 'One loss, two gradient machines', hint: 'L(w) = (sin(w·x) − y)²  with x = 1.5, y = 0.5' });
    let w = 0.8, v = values(w);

    // slider in the stage head
    const range = JT.el('input', { type: 'range', min: -3, max: 3, step: 0.05, value: w, 'aria-label': 'weight w' });
    const out = JT.el('output', { text: f3(w) });
    S.controls.appendChild(JT.el('label', { class: 'slider' }, ['w =', range, out]));

    const vals = JT.el('span', { class: 'vals' });
    S.body.appendChild(JT.el('div', { class: 'formula' }, [
      JT.el('span', { class: 'f', text: 'L(w) = (sin(w·x) − y)²' }),
      JT.el('span', { class: 'f', text: 'dL/dw = 2(sin(w·x) − y) · cos(w·x) · x' }),
      vals,
    ]));

    // ---- PyTorch pane ------------------------------------------------
    const gT = graph();
    const tapeList = JT.el('ol');
    const gradBox = JT.el('div', { class: 'gradbox' });
    const tape = JT.el('div', { class: 'tape' }, [JT.el('h6', { text: 'autograd tape (grad_fn chain)' }), tapeList, gradBox]);
    const codeT = JT.code(`w = torch.tensor(${f3(w)}, requires_grad=True)
L = (torch.sin(w * x) - y) ** 2   # forward builds the tape
L.backward()                       # walk it in reverse
w.grad`);
    const noteT = JT.el('div', { class: 'note' });
    const stepBarT = JT.el('div', { class: 'stepbar' });
    const paneT = JT.el('div', { class: 'pane torch' }, [
      JT.el('div', { class: 'pane-head', html: 'PyTorch <span class="sub">gradient as a side effect</span>' }),
      JT.el('div', { class: 'pane-body' }, [JT.el('div', { class: 'graph' }, gT.svg), codeT, tape, noteT, stepBarT]),
    ]);
    const TAPE = [
      { fn: 'MulBackward0', saves: 'saves x', node: 'mul' },
      { fn: 'SinBackward0', saves: 'saves a', node: 'sin' },
      { fn: 'SubBackward0', saves: '', node: 'sub' },
      { fn: 'PowBackward0', saves: 'saves c', node: 'sq' },
    ];
    const T_STEPS = 9;
    function torchStep(i) {
      const vv = values(w);
      gT.reset(); JT.clearLines(codeT, 'now'); JT.clearLines(codeT, 'hl');
      tapeList.replaceChildren(); gradBox.innerHTML = '';
      const fwd = Math.min(i + 1, 4);
      for (let k = 0; k < fwd; k++) {
        const t = TAPE[k];
        tapeList.appendChild(JT.el('li', { dataset: { k } }, [JT.el('span', { text: t.fn }), JT.el('span', { class: 'g', text: t.saves })]));
        gT.done([t.node]);
      }
      if (i < 4) {
        JT.markLines(codeT, [2], 'now');
        gT.mark([TAPE[i].node], 'fwd');
        const msgs = [
          'Forward, op 1: <code>w * x</code> runs on the device immediately. Because w requires grad, autograd appends a <code>MulBackward0</code> node that remembers x.',
          'Forward, op 2: <code>sin</code> runs; <code>SinBackward0</code> is appended and saves its input a, since d sin(a)/da = cos(a) needs it.',
          'Forward, op 3: subtraction. <code>SubBackward0</code> needs nothing saved: its local derivative is 1.',
          'Forward, op 4: the square. <code>PowBackward0</code> saves c. L is a tensor whose <code>grad_fn</code> points at the top of this chain.',
        ];
        noteT.innerHTML = msgs[i];
      } else if (i < 8) {
        JT.markLines(codeT, [3], 'now');
        const k = 7 - i; // backward walks 3,2,1,0
        const t = TAPE[k];
        JT.$$('li', tapeList).forEach((li) => { const kk = +li.dataset.k; li.classList.toggle('active', kk === k); li.classList.toggle('popped', kk > k); });
        gT.mark([t.node], 'bwd');
        const cot = [['sq-L', '1'], ['sub-sq', f3(vv.dc)], ['sin-sub', f3(vv.db)], ['mul-sin', f3(vv.da)], ['w-mul', f3(vv.dw)]];
        for (let e = 0; e <= 4 - k; e++) gT.tok(cot[e][0], cot[e][1]);
        const msgs = [
          `<code>L.backward()</code> seeds dL/dL = 1 and pops <code>PowBackward0</code>: grad_c = 2·c·1 = ${f3(vv.dc)}.`,
          `<code>SubBackward0</code>: the derivative of (b − y) with respect to b is 1, so grad_b = ${f3(vv.db)}.`,
          `<code>SinBackward0</code> multiplies by the saved cos(a) = ${f3(vv.cosa)}: grad_a = ${f3(vv.da)}.`,
          `<code>MulBackward0</code> multiplies by x = ${X}: grad_w = ${f3(vv.dw)}. An <code>AccumulateGrad</code> node adds it into <code>w.grad</code>.`,
        ];
        noteT.innerHTML = msgs[3 - k];
      } else {
        JT.markLines(codeT, [4], 'now');
        JT.$$('li', tapeList).forEach((li) => li.classList.add('popped'));
        [['sq-L', '1'], ['sub-sq', f3(vv.dc)], ['sin-sub', f3(vv.db)], ['mul-sin', f3(vv.da)], ['w-mul', f3(vv.dw)]].forEach(([id, t]) => gT.tok(id, t));
        gradBox.innerHTML = `w.grad = <b>tensor(${f3(vv.dw)})</b>`;
        noteT.innerHTML = 'Done. The gradient lives in <code>w.grad</code>, a mutable attribute. The graph has been freed; the next forward pass builds a fresh one. Gradients from later forward/backward passes accumulate unless <code>w.grad</code> is cleared.';
      }
    }
    function torchReset() { gT.reset(); gT.setValues(values(w)); tapeList.replaceChildren(JT.el('li', { class: 'empty', text: 'empty until forward runs' })); gradBox.innerHTML = 'w.grad = None'; JT.clearLines(codeT, 'now'); noteT.innerHTML = 'Step forward: four eager ops build the tape, then <code>backward()</code> unwinds it.'; }
    const stT = JT.stepper({ total: T_STEPS, interval: 1400, onStep: torchStep, onReset: torchReset });
    stepBarT.appendChild(stT.el);

    // ---- JAX pane ----------------------------------------------------
    const gJ = graph();
    const jaxprHost = JT.el('div', { class: 'jaxpr' });
    const codeJ = JT.code(`def L(w):
    return (jnp.sin(w * x) - y) ** 2

dL = jax.grad(L)     # a new function, nothing ran yet
dL(${f3(w)})`);
    const noteJ = JT.el('div', { class: 'note' });
    const stepBarJ = JT.el('div', { class: 'stepbar' });
    const paneJ = JT.el('div', { class: 'pane jax' }, [
      JT.el('div', { class: 'pane-head', html: 'JAX <span class="sub">gradient as a new function</span>' }),
      JT.el('div', { class: 'pane-body' }, [JT.el('div', { class: 'graph' }, gJ.svg), codeJ, jaxprHost, noteJ, stepBarJ]),
    ]);
    const FWD = `{ lambda ; a:f32[]. let
    b:f32[] = mul a 1.5
    c:f32[] = sin b
    d:f32[] = sub c 0.5
    e:f32[] = integer_pow[y=2] d
  in (e,) }`;
    const JVP = `# tangent rules, one per primitive (jvp); a' is the tangent of a
    b' = mul a' 1.5
    c' = mul b' (cos b)
    d' = c'
    e' = mul d' (2·d)`;
    const VJP = `# transposed: run the linear part backwards; e* is the cotangent of e
    d* = mul e* (2·d)
    c* = d*
    b* = mul c* (cos b)
    a* = mul b* 1.5`;
    const GRAD = `{ lambda ; a:f32[]. let
    b:f32[] = mul a 1.5
    c:f32[] = cos b
    d:f32[] = sin b
    e:f32[] = sub d 0.5
    f:f32[] = mul 2.0 e
    g:f32[] = mul f c
    h:f32[] = mul g 1.5
  in (h,) }`;
    const J_STEPS = 5;
    function jaxStep(i) {
      const vv = values(w);
      gJ.reset(); JT.clearLines(codeJ, 'now');
      if (i === 0) {
        JT.markLines(codeJ, [1, 2], 'now');
        jaxprHost.replaceChildren(JT.code(FWD, { lang: 'jaxpr' }));
        gJ.mark(['mul', 'sin', 'sub', 'sq'], 'fwd');
        noteJ.innerHTML = '<code>grad</code> first traces L to a jaxpr: four primitives, no values. This is the same tracing <code>jit</code> does.';
      } else if (i === 1) {
        JT.markLines(codeJ, [4], 'now');
        jaxprHost.replaceChildren(JT.code(JVP, { lang: 'jaxpr' }));
        gJ.done(['mul', 'sin', 'sub', 'sq']);
        gJ.label('w-mul', '×1.5'); gJ.label('mul-sin', '×cos b'); gJ.label('sin-sub', '×1'); gJ.label('sub-sq', '×2d');
        noteJ.innerHTML = 'Every primitive has a <em>JVP rule</em>: how a small change in its input changes its output. Applying them gives a linear function of the input tangent a′. This is forward mode.';
      } else if (i === 2) {
        JT.markLines(codeJ, [4], 'now');
        jaxprHost.replaceChildren(JT.code(VJP, { lang: 'jaxpr' }));
        gJ.done(['mul', 'sin', 'sub', 'sq']);
        gJ.mark(['sq', 'sub', 'sin', 'mul'], 'bwd');
        gJ.label('sq-L', 'e* = 1'); gJ.label('sub-sq', '×2d'); gJ.label('sin-sub', '×1'); gJ.label('mul-sin', '×cos b'); gJ.label('w-mul', '×1.5');
        ['sq-L', 'sub-sq', 'sin-sub', 'mul-sin', 'w-mul'].forEach((e) => gJ.edge(e, true));
        noteJ.innerHTML = 'The linear function is <em>transposed</em>: each multiplication is read backwards, from the output cotangent e* = 1 to the input cotangent a*. Forward mode plus transposition equals reverse mode.';
      } else if (i === 3) {
        JT.markLines(codeJ, [4], 'now');
        const pre = JT.code(GRAD, { lang: 'jaxpr' }); jaxprHost.replaceChildren(pre);
        JT.markLines(pre, [6, 7, 8], 'hl');
        gJ.done(['mul', 'sin', 'sub', 'sq']);
        noteJ.innerHTML = 'Stitched together, <code>dL</code> is an ordinary jaxpr: the forward values it needs (cos b, 2·e) and the transposed chain. No tape, no saved tensors on objects: just a function from w to dL/dw.';
      } else {
        JT.markLines(codeJ, [5], 'now');
        const pre = JT.code(GRAD + `\n\n# dL(${f3(w)}) → Array(${f3(vv.dw)}, dtype=float32)`, { lang: 'jaxpr' }); jaxprHost.replaceChildren(pre);
        JT.markLines(pre, [11], 'now');
        gJ.done(['mul', 'sin', 'sub', 'sq']);
        [['sq-L', '1'], ['sub-sq', f3(vv.dc)], ['sin-sub', f3(vv.db)], ['mul-sin', f3(vv.da)], ['w-mul', f3(vv.dw)]].forEach(([id, t]) => gJ.tok(id, t));
        noteJ.innerHTML = `Calling <code>dL(w)</code> returns <b>${f3(vv.dw)}</b>, the same number PyTorch put in <code>w.grad</code>. Because dL is a plain function, <code>jit(dL)</code>, <code>vmap(dL)</code> and <code>grad(dL)</code> all work without new machinery.`;
      }
    }
    function jaxReset() { gJ.reset(); gJ.setValues(values(w)); jaxprHost.replaceChildren(JT.el('pre', { class: 'code', html: '<span class="ln" style="color:var(--ink-3)">— grad(L) has not been traced yet —</span>' })); JT.clearLines(codeJ, 'now'); noteJ.innerHTML = 'Step forward: trace, linearize, transpose, and you have a gradient function.'; }
    const stJ = JT.stepper({ total: J_STEPS, interval: 2200, onStep: jaxStep, onReset: jaxReset });
    stepBarJ.appendChild(stJ.el);

    S.body.appendChild(JT.el('div', { class: 'two-col' }, [paneT, paneJ]));
    S.foot.innerHTML = '<span>Both differentiation APIs apply the same chain rule. PyTorch records the executed operations; JAX transforms the function and returns the gradient as a value.</span>';

    function setLine(pre, index, src) {
      const ln = JT.$$('.ln', pre)[index];
      ln.innerHTML = JT.highlight(src).replace(/^<span class="ln" data-line="1">/, '').replace(/<\/span>$/, '');
    }
    function refresh() {
      v = values(w); out.value = f3(w);
      vals.innerHTML = `L = <b>${f3(v.L)}</b> &nbsp; dL/dw = <b>${f3(v.dw)}</b>`;
      gT.setValues(v); gJ.setValues(v);
      setLine(codeT, 0, `w = torch.tensor(${f3(w)}, requires_grad=True)`);
      setLine(codeJ, 4, `dL(${f3(w)})`);
      if (stT.i >= 0) torchStep(stT.i);
      if (stJ.i >= 0) jaxStep(stJ.i);
    }
    range.addEventListener('input', () => { w = +range.value; refresh(); });
    torchReset(); jaxReset(); refresh();
  });
})();
