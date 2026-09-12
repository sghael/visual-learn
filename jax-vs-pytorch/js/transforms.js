/* Transformations widget: (a) vmap as batching-by-rewrite rather than a
   loop, (b) a composer for stacking jit / grad / vmap. */
(function () {
  'use strict';
  const JT = window.JT;

  JT.style('transforms', `
    #w-transforms .vgrid { display: grid; grid-template-columns: minmax(0, 1.3fr) minmax(0, 1fr); gap: 1.25rem; align-items: start; }
    #w-transforms .vctl { display: flex; flex-wrap: wrap; gap: 0.5rem 1rem; align-items: center; margin-bottom: 0.75rem; font-size: 0.8125rem; color: var(--ink-2); }
    #w-transforms .vctl .lbl { display: inline-flex; align-items: center; gap: 0.4rem; }
    #w-transforms .vctl .controls { margin-left: auto; }
    #w-transforms .cell { fill: var(--surface); stroke: var(--line-2); stroke-width: 1; transition: fill 200ms, stroke 200ms; }
    #w-transforms .cell.w { fill: var(--surface-2); }
    #w-transforms .cell.hot { fill: var(--live-soft); stroke: var(--live); }
    #w-transforms .cell.out { fill: var(--jax-soft); stroke: var(--jax-mid); }
    #w-transforms .cell.out.hot { fill: var(--jax); stroke: var(--jax); }
    #w-transforms .lab { font-size: 11px; fill: var(--ink-3); }
    #w-transforms .lab.mono { font-family: var(--font-mono); font-size: 10.5px; }
    #w-transforms .fnbox rect { fill: var(--surface); stroke: var(--ink-3); stroke-width: 1.2; transition: fill 200ms, stroke 200ms; }
    #w-transforms .fnbox text { font-family: var(--font-mono); font-size: 11px; fill: var(--ink); text-anchor: middle; dominant-baseline: central; }
    #w-transforms .fnbox.hot rect { fill: var(--live-soft); stroke: var(--live); }
    #w-transforms .flow { stroke: var(--line-2); stroke-width: 1.2; fill: none; transition: stroke 200ms; }
    #w-transforms .flow.hot { stroke: var(--live); stroke-width: 1.6; }
    #w-transforms .vread { display: flex; flex-wrap: wrap; gap: 0.5rem 1.5rem; margin-top: 0.5rem; }
    #w-transforms pre.code { border: 1px solid var(--line); border-radius: var(--radius-sm); font-size: 0.76rem; }
    #w-transforms .jaxpr-title { font-size: 0.72rem; font-weight: 600; letter-spacing: 0.03em; text-transform: uppercase; color: var(--ink-3); margin: 0.75rem 0 0.35rem; }
    #w-transforms .divider { border-top: 1px solid var(--line); margin: 1.5rem 0 1.25rem; }
    #w-transforms .composer { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1.2fr); gap: 1.25rem; align-items: start; }
    #w-transforms .chips { display: flex; flex-wrap: wrap; gap: 0.4rem; align-items: center; }
    #w-transforms .chips .lead { font-size: 0.8125rem; color: var(--ink-2); margin-right: 0.25rem; }
    #w-transforms .stackview { margin: 0.9rem 0; min-height: 3.2rem; display: flex; flex-wrap: wrap; align-items: center; gap: 0.25rem; font-family: var(--font-mono); font-size: 1rem; }
    #w-transforms .stackview .t { padding: 0.15rem 0.45rem; border-radius: 6px; background: var(--jax-soft); color: var(--jax-deep); font-weight: 500; animation: tf-in 300ms var(--ease-out); }
    @keyframes tf-in { from { opacity: 0; transform: scale(0.9); } }
    #w-transforms .stackview .f { color: var(--ink); }
    #w-transforms .stackview .p { color: var(--ink-3); }
    #w-transforms .sig { font-family: var(--font-mono); font-size: 0.8rem; color: var(--ink-2); margin-bottom: 0.5rem; }
    #w-transforms .sig b { color: var(--ink); font-weight: 600; }
    #w-transforms .expl { font-size: 0.875rem; color: var(--ink-2); line-height: 1.55; min-height: 4.5rem; }
    #w-transforms .expl.bad { color: var(--err); }
    #w-transforms .eq { margin-top: 0.75rem; }
    #w-transforms .eq .pane-body { padding: 0.5rem 0.75rem; font-size: 0.8rem; }
    #w-transforms .presets { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.75rem; }
    #w-transforms .scroller { overflow-x: auto; }
    @media (max-width: 860px) { #w-transforms .vgrid, #w-transforms .composer { grid-template-columns: 1fr; } }
    @media (max-width: 640px) { #w-transforms .scroller svg { min-width: 560px; } }
  `);

  // ---- vmap ------------------------------------------------------------
  function vmapPanel(S) {
    const B = 5, D = 3;
    let xAxis = 0, wAxis = null, mode = 'vmap';
    const wrap = JT.el('div');

    const segX = JT.seg([{ value: '0', label: 'in_axes x = 0' }, { value: '1', label: 'x = 1' }], (v) => { xAxis = +v; draw(); }, '0');
    const segW = JT.seg([{ value: 'None', label: 'w = None' }, { value: '0', label: 'w = 0' }], (v) => { wAxis = v === 'None' ? null : 0; draw(); }, 'None');
    const segMode = JT.seg([{ value: 'loop', label: 'Python loop' }, { value: 'vmap', label: 'vmap' }], (v) => { mode = v; draw(); }, 'vmap', 'jax');
    const btnRun = JT.el('button', { class: 'btn primary', type: 'button', html: JT.icon('play') + '<span>Run</span>' });
    const ctl = JT.el('div', { class: 'vctl' }, [
      JT.el('span', { class: 'lbl', text: 'batch axis:' }), segX, segW,
      JT.el('div', { class: 'controls' }, [segMode, btnRun]),
    ]);

    const svg = JT.svg('svg', { viewBox: '0 0 640 210', role: 'img', 'aria-label': 'vmap threads a batch axis through a per-example function' });
    const codeHost = JT.el('div');
    const jaxprHost = JT.el('div');
    const read = JT.el('div', { class: 'vread' });
    wrap.append(ctl, JT.el('div', { class: 'vgrid' }, [JT.el('div', {}, [JT.el('div', { class: 'scroller' }, svg), read]), JT.el('div', {}, [codeHost, JT.el('div', { class: 'jaxpr-title', text: 'jaxpr after vmap' }), jaxprHost])]));

    let cells = { x: [], w: [], y: [] }, fnbox = null, flows = [];
    function draw() {
      svg.replaceChildren();
      cells = { x: [], w: [], y: [] }; flows = [];
      const c = 20, g = 3;
      // w block
      const wRows = wAxis === 0 ? B : 1;
      svg.appendChild(JT.svg('text', { class: 'lab mono', x: 24, y: 16, text: wAxis === 0 ? `w  [${B}, ${D}]` : `w  [${D}]` }));
      for (let i = 0; i < wRows; i++) {
        cells.w[i] = [];
        for (let j = 0; j < D; j++) { const r = JT.svg('rect', { class: 'cell w', x: 24 + j * (c + g), y: 22 + i * (c + g), width: c, height: c, rx: 4 }); svg.appendChild(r); cells.w[i][j] = r; }
      }
      // x block
      const xTop = 22 + wRows * (c + g) + 26;
      const xRows = xAxis === 0 ? B : D, xCols = xAxis === 0 ? D : B;
      svg.appendChild(JT.svg('text', { class: 'lab mono', x: 24, y: xTop - 6, text: xAxis === 0 ? `X  [${B}, ${D}]  examples are rows` : `X  [${D}, ${B}]  examples are columns` }));
      for (let i = 0; i < xRows; i++) {
        cells.x[i] = [];
        for (let j = 0; j < xCols; j++) { const r = JT.svg('rect', { class: 'cell', x: 24 + j * (c + g), y: xTop + i * (c + g), width: c, height: c, rx: 4 }); svg.appendChild(r); cells.x[i][j] = r; }
      }
      // function box
      const fx = 300, fy = 105;
      fnbox = JT.svg('g', { class: 'fnbox' }, [JT.svg('rect', { x: fx - 62, y: fy - 22, width: 124, height: 44, rx: 9 }), JT.svg('text', { x: fx, y: fy, text: mode === 'vmap' ? 'vmap(predict)' : 'predict' })]);
      // flows: from each example to the box, and from box to each output
      const exampleCount = B;
      for (let e = 0; e < exampleCount; e++) {
        const sx = 24 + xCols * (c + g) + 4;
        const sy = xAxis === 0 ? xTop + e * (c + g) + c / 2 : xTop + (xRows * (c + g)) / 2;
        const p = JT.svg('path', { class: 'flow', d: `M${sx},${sy} C ${sx + 60},${sy} ${fx - 100},${fy} ${fx - 64},${fy}` });
        svg.insertBefore(p, svg.firstChild); flows.push(p);
      }
      // outputs
      svg.appendChild(JT.svg('text', { class: 'lab mono', x: 470, y: 16, text: `y  [${B}]` }));
      for (let e = 0; e < B; e++) {
        const r = JT.svg('rect', { class: 'cell out', x: 470, y: 22 + e * (c + g), width: c, height: c, rx: 4 }); svg.appendChild(r); cells.y[e] = r;
        const p = JT.svg('path', { class: 'flow', d: `M${fx + 62},${fy} C ${fx + 100},${fy} ${440},${22 + e * (c + g) + c / 2} ${468},${22 + e * (c + g) + c / 2}` });
        svg.insertBefore(p, svg.firstChild); flows.push(p);
      }
      svg.appendChild(fnbox);
      svg.appendChild(JT.svg('text', { class: 'lab', x: fx, y: fy + 38, 'text-anchor': 'middle', text: mode === 'vmap' ? 'one call, one batched dot_general' : 'five calls, five separate dots' }));

      const inAxes = `(${wAxis === null ? 'None' : '0'}, ${xAxis})`;
      codeHost.replaceChildren(JT.code(mode === 'vmap'
        ? `def predict(w, x):          # one example
    return jnp.dot(w, x)     # x: [3] → scalar

batched = jax.vmap(predict, in_axes=${inAxes})
y = batched(w, X)            # y: [${B}]`
        : `def predict(w, x):          # one example
    return jnp.dot(w, x)     # x: [3] → scalar

y = jnp.stack([predict(w${wAxis === 0 ? '[i]' : ''}, X[${xAxis === 0 ? 'i' : ':, i'}])
               for i in range(${B})])`));
      const wSig = wAxis === 0 ? `a:f32[${B},${D}]` : `a:f32[${D}]`;
      const xSig = xAxis === 0 ? `b:f32[${B},${D}]` : `b:f32[${D},${B}]`;
      let dn;
      if (wAxis === 0) dn = `contracting=([1],[${xAxis === 0 ? 1 : 0}]) batch=([0],[${xAxis === 0 ? 0 : 1}])`;
      else dn = `contracting=([0],[${xAxis === 0 ? 1 : 0}]) batch=([],[])`;
      jaxprHost.replaceChildren(JT.code(mode === 'vmap'
        ? `{ lambda ; ${wSig} ${xSig}. let
    c:f32[${B}] = dot_general[
      ${dn}
    ] a b
  in (c,) }`
        : `# no single jaxpr: five traces of
{ lambda ; a:f32[${D}] b:f32[${D}]. let
    c:f32[] = dot_general[contracting=([0],[0])] a b
  in (c,) }
# then a stack`, { lang: 'jaxpr' }));
      read.innerHTML = `<span class="readout">calls: <b>${mode === 'vmap' ? 1 : B}</b></span><span class="readout">kernel launches: <b>${mode === 'vmap' ? 1 : B}</b></span><span class="readout">output: <b>f32[${B}]</b></span>`;
    }

    function hotExample(e, on) {
      const xs = xAxis === 0 ? cells.x[e] : cells.x.map((row) => row[e]);
      xs.forEach((r) => r.classList.toggle('hot', on));
      if (wAxis === 0) cells.w[e].forEach((r) => r.classList.toggle('hot', on)); else cells.w[0].forEach((r) => r.classList.toggle('hot', on));
      flows[e].classList.toggle('hot', on); flows[B + e].classList.toggle('hot', on);
    }
    let running = false;
    async function run() {
      if (running) return; running = true; btnRun.disabled = true;
      JT.$$('.hot', svg).forEach((el) => el.classList.remove('hot'));
      cells.y.forEach((r) => r.classList.remove('hot'));
      if (mode === 'loop') {
        for (let e = 0; e < B; e++) {
          hotExample(e, true); fnbox.classList.add('hot');
          await JT.wait(420);
          cells.y[e].classList.add('hot');
          await JT.wait(200);
          hotExample(e, false); fnbox.classList.remove('hot');
        }
      } else {
        for (let e = 0; e < B; e++) hotExample(e, true);
        fnbox.classList.add('hot');
        await JT.wait(700);
        cells.y.forEach((r) => r.classList.add('hot'));
        await JT.wait(500);
        for (let e = 0; e < B; e++) hotExample(e, false);
        fnbox.classList.remove('hot');
      }
      running = false; btnRun.disabled = false;
    }
    btnRun.addEventListener('click', run);
    draw();
    return { el: wrap, run };
  }

  // ---- composer ----------------------------------------------------------
  function composerPanel() {
    const wrap = JT.el('div', { class: 'composer' });
    let stack = []; // outermost last
    const stackView = JT.el('div', { class: 'stackview' });
    const sig = JT.el('div', { class: 'sig' });
    const expl = JT.el('div', { class: 'expl' });
    const eq = JT.el('div', { class: 'pane torch eq' }, [JT.el('div', { class: 'pane-head', text: 'PyTorch equivalent' }), JT.el('div', { class: 'pane-body' })]);

    const chips = JT.el('div', { class: 'chips' }, [JT.el('span', { class: 'lead', text: 'Wrap with:' })]);
    ['jit', 'grad', 'vmap', 'jacfwd'].forEach((t) => {
      chips.appendChild(JT.el('button', { class: 'chip', type: 'button', text: t + '( · )', onClick: () => { if (stack.length < 4) { stack.push(t); render(); } } }));
    });
    chips.appendChild(JT.el('button', { class: 'btn', type: 'button', html: JT.icon('reset') + '<span>clear</span>', onClick: () => { stack = []; render(); } }));
    const presets = JT.el('div', { class: 'presets' }, [
      JT.el('button', { class: 'chip', type: 'button', text: 'per-example gradients', onClick: () => { stack = ['grad', 'vmap']; render(); } }),
      JT.el('button', { class: 'chip', type: 'button', text: 'compiled training gradient', onClick: () => { stack = ['grad', 'jit']; render(); } }),
      JT.el('button', { class: 'chip', type: 'button', text: 'Hessian (second derivative)', onClick: () => { stack = ['grad', 'jacfwd']; render(); } }),
      JT.el('button', { class: 'chip', type: 'button', text: 'the wrong order', onClick: () => { stack = ['vmap', 'grad']; render(); } }),
    ]);

    function evaluate() {
      // state of the function being wrapped: loss(params, x, y) -> scalar
      let scalar = true, order = 0, batches = 0, compiled = false, error = null;
      const notes = [];
      for (const t of stack) {
        if (t === 'grad') {
          if (!scalar) {
            error = order
              ? 'grad needs a scalar-output function, but the function it wraps already returns a pytree shaped like params. grad(grad(loss)) is an error in JAX. For second derivatives of a pytree-valued gradient use jacfwd(grad(loss)), which is exactly how jax.hessian is defined.'
              : `grad needs a scalar output, but the function it wraps returns f32[${Array(batches).fill('B').join(',')}]. Sum the batch first, or move vmap outside: vmap(grad(loss)).`;
            break;
          }
          order += 1; scalar = false;
          notes.push('grad: the result is a function returning ∂loss/∂params, a pytree with the same structure as params. It requires a scalar output, which is why it must go innermost, before vmap.');
        } else if (t === 'jacfwd') {
          order += 1; scalar = false;
          notes.push(order === 1
            ? 'jacfwd: the forward-mode Jacobian. On a scalar loss that is the same gradient, computed one input element per pass, so reverse mode (grad) is the cheaper choice at this position.'
            : 'jacfwd over grad: the Jacobian of the gradient is the Hessian, computed forward-over-reverse. jax.hessian(f) is defined as jacfwd(jacrev(f)); the second derivative is just another composition.');
        } else if (t === 'vmap') {
          batches += 1; scalar = false; // a batch of losses is a vector, not a scalar
          notes.push(`vmap: every output gains a batch axis. With in_axes=(None, 0, 0) params are shared and x, y are split per example; the result is ${order ? 'one derivative per example' : 'one loss per example'}.`);
        } else if (t === 'jit') {
          compiled = true;
          notes.push('jit: trace the whole composed function once and compile it with XLA. Output shapes are unchanged; call cost drops to one dispatch.');
        }
      }
      const bd = batches ? Array(batches).fill('B').join(',') + ',' : '';
      let outShape;
      if (order === 0) outShape = batches ? `f32[${bd.slice(0, -1)}]` : 'f32[]';
      else if (order === 1) outShape = `{w: f32[${bd}…], b: f32[${bd}…]}`;
      else outShape = `{w: {w: f32[${bd}…,…], b: f32[${bd}…]}, b: {w: …, b: …}}`;
      return { error, notes, outShape, compiled, batches, order };
    }

    function torchEquivalent() {
      if (!stack.length) return 'loss(params, x, y)';
      let expr = 'loss';
      for (const t of stack) {
        if (t === 'grad') expr = `torch.func.grad(${expr})`;
        else if (t === 'vmap') expr = `torch.func.vmap(${expr}, in_dims=(None, 0, 0))`;
        else if (t === 'jacfwd') expr = `torch.func.jacfwd(${expr})`;
        else expr = `torch.compile(${expr})`;
      }
      return expr;
    }

    function render() {
      stackView.replaceChildren();
      if (!stack.length) stackView.appendChild(JT.el('span', { class: 'f', text: 'loss(params, x, y)' }));
      else {
        [...stack].reverse().forEach((t) => stackView.appendChild(JT.el('span', { class: 't', text: t + '(' })));
        stackView.appendChild(JT.el('span', { class: 'f', text: 'loss' }));
        stackView.appendChild(JT.el('span', { class: 'p', text: ')'.repeat(stack.length) + '(params, x, y)' }));
      }
      const r = evaluate();
      if (r.error) { sig.innerHTML = 'returns: <b>error at trace time</b>'; expl.textContent = r.error; expl.classList.add('bad'); }
      else {
        sig.innerHTML = `returns: <b>${JT.escape(r.outShape)}</b>${r.compiled ? ' · compiled' : ''}`;
        expl.classList.remove('bad');
        expl.innerHTML = r.notes.length ? r.notes.map((n) => `<div>${n}</div>`).join('') : 'Start with a plain loss function of the parameters and one batch. Add transformations from the inside out; each one returns a new function, so the next can wrap it.';
      }
      JT.$('.pane-body', eq).replaceChildren(JT.code(torchEquivalent() + (stack.length ? '' : '   # plain eager call')));
    }

    wrap.append(
      JT.el('div', {}, [chips, stackView, presets]),
      JT.el('div', {}, [sig, expl, eq]),
    );
    render();
    return { el: wrap };
  }

  JT.widget('transforms', (container) => {
    const S = JT.stage(container, { title: 'vmap: batch by rewriting, not by looping', hint: 'then stack the transformations' });
    const v = vmapPanel(S);
    const comp = composerPanel();
    S.body.append(v.el, JT.el('div', { class: 'divider' }), JT.el('h4', { style: { margin: '0 0 0.75rem', fontSize: '0.95rem' }, text: 'Compose jit, grad and vmap' }), comp.el);
    S.foot.innerHTML = '<span>In the composer, the inner transformation is applied first: <code class="inline-code">vmap(grad(loss))</code> differentiates one example, then batches the result.</span>';
    JT.onVisible(container, () => v.run(), { threshold: 0.5 });
  });
})();
