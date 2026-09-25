/* Two figures about function transformations.
   'vmap':     a Python loop and jax.vmap side by side (small multiples), with
               the batch axis of X as the one control. Jaxprs from JAX 0.11.2,
               preferred_element_type and literal annotations removed.
   'composer': stack jit, grad, vmap and jacfwd and see what the result returns. */
(function () {
  'use strict';
  const JT = window.JT;
  const B = 5, D = 3;

  /* ---------------------------------------------------------------- */
  /* vmap versus a loop                                                 */
  /* ---------------------------------------------------------------- */
  function sketch(mode, axis) {
    const W = 290, H = 142, c = 16, g = 2, s = c + g;
    const svg = JT.svg('svg', { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: 'img',
      'aria-label': mode === 'loop' ? 'Five examples, each sent through its own dot product' : 'Five examples sent through one batched dot_general' });
    const gy = 40, rowY = (e) => gy + e * s + c / 2;
    // X
    const rows = axis === 0 ? B : D, cols = axis === 0 ? D : B;
    svg.appendChild(JT.svg('text', { class: 'lab mono', x: 0, y: gy - 8, text: axis === 0 ? 'X [5, 3]' : 'X [3, 5]' }));
    for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) svg.appendChild(JT.svg('rect', { class: 'cell', x: j * s, y: gy + i * s, width: c, height: c }));
    const anchor = (e) => (axis === 0 ? { x: cols * s + 2, y: rowY(e) } : { x: e * s + c / 2, y: gy + rows * s + 2 });
    svg.appendChild(JT.svg('text', { class: 'lab', x: 0, y: H - 2, text: axis === 0 ? 'one example per row' : 'one example per column' }));
    // w, shared
    const wx = 150;
    svg.appendChild(JT.svg('text', { class: 'lab mono', x: wx - 44, y: 17, text: 'w [3]' }));
    for (let j = 0; j < D; j++) svg.appendChild(JT.svg('rect', { class: 'cell w', x: wx + j * s, y: 6, width: c, height: c }));
    // op(s)
    const ox = 158, ow = mode === 'loop' ? 34 : 92;
    const flows = JT.svg('g');
    svg.appendChild(flows);
    if (mode === 'loop') {
      for (let e = 0; e < B; e++) {
        svg.appendChild(JT.svg('rect', { class: 'op', x: ox, y: rowY(e) - 7, width: ow, height: 14, rx: 2 }));
        svg.appendChild(JT.svg('text', { class: 'oplab', x: ox + ow / 2, y: rowY(e) + 4, 'text-anchor': 'middle', text: 'dot' }));
      }
    } else {
      svg.appendChild(JT.svg('rect', { class: 'op', x: ox - 30, y: gy - 2, width: ow, height: B * s + 2, rx: 2 }));
      svg.appendChild(JT.svg('text', { class: 'oplab', x: ox - 30 + ow / 2, y: rowY(2) + 4, 'text-anchor': 'middle', text: 'dot_general' }));
    }
    const inX = mode === 'loop' ? ox : ox - 30;
    const outX = mode === 'loop' ? ox + ow : ox - 30 + ow;
    const yx = 272;
    for (let e = 0; e < B; e++) {
      const a = anchor(e), ty = rowY(e);
      const d = axis === 0
        ? `M${a.x},${a.y} C${a.x + 30},${a.y} ${inX - 30},${ty} ${inX},${ty}`
        : `M${a.x},${a.y} C${a.x},${a.y + 14} ${inX - 40},${ty} ${inX},${ty}`;
      flows.appendChild(JT.svg('path', { class: 'flow', d }));
      flows.appendChild(JT.svg('path', { class: 'flow', d: `M${outX},${ty} L${yx},${ty}` }));
      svg.appendChild(JT.svg('rect', { class: 'cell y', x: yx, y: ty - c / 2, width: c, height: c }));
    }
    svg.appendChild(JT.svg('text', { class: 'lab mono', x: yx - 6, y: gy - 8, text: 'y [5]' }));
    return svg;
  }

  function loopCode(axis) {
    return `def predict(w, x):      # x: one example, f32[3]
    return jnp.dot(w, x)

y = jnp.stack([predict(w, X[${axis === 0 ? 'i' : ':, i'}])
               for i in range(5)])`;
  }
  const loopJaxpr = `{ lambda ; a:f32[3] b:f32[3]. let
    c:f32[] = dot_general[
      dimension_numbers=(([0], [0]), ([], []))
    ] a b
  in (c,) }`;
  function vmapCode(axis) {
    return `def predict(w, x):      # x: one example, f32[3]
    return jnp.dot(w, x)

batched = jax.vmap(predict, in_axes=(None, ${axis}))
y = batched(w, X)       # X: ${axis === 0 ? 'f32[5, 3]' : 'f32[3, 5]'}`;
  }
  function vmapJaxpr(axis) {
    return `{ lambda ; a:f32[3] b:${axis === 0 ? 'f32[5,3]' : 'f32[3,5]'}. let
    c:f32[5] = dot_general[
      dimension_numbers=(([0], [${axis === 0 ? 1 : 0}]), ([], []))
    ] a b
  in (c,) }`;
  }

  JT.widget('vmap', (container) => {
    container.classList.add('vmap');
    let axis = 0;
    const seg = JT.seg([{ value: '0', label: 'axis 0 (rows)' }, { value: '1', label: 'axis 1 (columns)' }], (v) => { axis = +v; render(); }, '0', 'Batch axis of X');
    container.appendChild(JT.el('div', { class: 'controls' }, [JT.control('Examples of X lie along', [seg])]));
    const mk = (mode, title) => {
      const el = JT.el('div', { dataset: { mode } });
      const head = JT.el('h4', { html: title });
      const read = JT.el('p', { class: 'readline' });
      const sk = JT.el('div');
      const code = JT.el('div');
      const jx = JT.el('div');
      el.append(head, read, sk, code, JT.el('h4', { html: mode === 'loop' ? 'Jaxpr of one call <span class="sub">(run five times)</span>' : 'Jaxpr of the batched call' }), jx);
      return { el, read, sk, code, jx, mode };
    };
    const loop = mk('loop', 'Python loop');
    const vm = mk('vmap', '<span class="jax-c">jax.vmap</span>');
    container.appendChild(JT.el('div', { class: 'cols' }, [loop.el, vm.el]));

    function render() {
      loop.read.innerHTML = '<b>5</b> calls to predict · <b>5</b> dot products, one at a time · then a stack';
      vm.read.innerHTML = '<b>1</b> call · <b>1</b> <code>dot_general</code> over the whole batch';
      loop.sk.replaceChildren(sketch('loop', axis));
      vm.sk.replaceChildren(sketch('vmap', axis));
      loop.code.replaceChildren(JT.code(loopCode(axis)));
      vm.code.replaceChildren(JT.code(vmapCode(axis)));
      loop.jx.replaceChildren(JT.code(loopJaxpr, { lang: 'jaxpr' }));
      vm.jx.replaceChildren(JT.code(vmapJaxpr(axis), { lang: 'jaxpr' }));
    }
    render();
  });

  /* ---------------------------------------------------------------- */
  /* Composer                                                           */
  /* ---------------------------------------------------------------- */
  /** Pure model: what does the composed function return? stack is innermost first. */
  function evaluate(stack) {
    let scalar = true, order = 0, batches = 0, compiled = false, error = null;
    const notes = [];
    for (const t of stack) {
      if (t === 'grad') {
        if (!scalar) {
          error = order
            ? 'grad needs a function with a scalar output, but the function it wraps already returns a pytree shaped like params, so grad(grad(loss)) fails. For second derivatives of a pytree-valued gradient use jacfwd(grad(loss)), which is how jax.hessian is defined.'
            : `grad needs a scalar output, but the function it wraps returns f32[${Array(batches).fill('B').join(',')}]. Sum over the batch first, or move vmap outside: vmap(grad(loss)).`;
          break;
        }
        order += 1; scalar = false;
        notes.push('grad returns a function that computes ∂loss/∂params, a pytree with the same structure as params. It requires a scalar output, so it goes inside vmap.');
      } else if (t === 'jacfwd') {
        order += 1; scalar = false;
        notes.push(order === 1
          ? 'jacfwd computes the Jacobian in forward mode. On a scalar loss that is the same gradient, computed one input element per pass, so grad (reverse mode) is cheaper here.'
          : 'jacfwd of the gradient is the Hessian, computed forward-over-reverse. jax.hessian(f) is defined as jacfwd(jacrev(f)).');
      } else if (t === 'vmap') {
        batches += 1; scalar = false;
        notes.push(`vmap adds a batch axis to every output. With in_axes=(None, 0, 0), params are shared and x and y are split per example; the result is ${order ? 'one derivative per example' : 'one loss per example'}.`);
      } else if (t === 'jit') {
        compiled = true;
        notes.push('jit traces the whole composed function once and compiles it with XLA. Output shapes do not change.');
      }
    }
    const bd = batches ? Array(batches).fill('B').join(',') + ',' : '';
    let outShape;
    if (order === 0) outShape = batches ? `f32[${bd.slice(0, -1)}]` : 'f32[]';
    else if (order === 1) outShape = `{w: f32[${bd}…], b: f32[${bd}…]}`;
    else outShape = `{w: {w: f32[${bd}…,…], b: f32[${bd}…]}, b: {w: …, b: …}}`;
    return { error, notes, outShape, compiled, batches, order };
  }
  function torchEquivalent(stack) {
    if (!stack.length) return 'loss(params, x, y)   # a plain eager call';
    let expr = 'loss';
    for (const t of stack) {
      if (t === 'grad') expr = `torch.func.grad(${expr})`;
      else if (t === 'vmap') expr = `torch.func.vmap(${expr}, in_dims=(None, 0, 0))`;
      else if (t === 'jacfwd') expr = `torch.func.jacfwd(${expr})`;
      else expr = `torch.compile(${expr})`;
    }
    return expr;
  }

  const PRESETS = [
    { label: 'per-example gradients', stack: ['grad', 'vmap'] },
    { label: 'compiled gradient', stack: ['grad', 'jit'] },
    { label: 'Hessian', stack: ['grad', 'jacfwd'] },
    { label: 'the wrong order', stack: ['vmap', 'grad'] },
  ];

  JT.widget('composer', (container) => {
    container.classList.add('composer');
    let stack = ['grad', 'vmap']; // innermost first; rests on per-example gradients

    const wrap = ['jit', 'grad', 'vmap', 'jacfwd'].map((t) => JT.button(t + '( )', () => { if (stack.length < 4) { stack.push(t); render(); } }, { class: 'btn mono', dataset: { wrap: t } }));
    const clear = JT.button('Clear', () => { stack = []; render(); }, { dataset: { role: 'clear' } });
    const presets = PRESETS.map((p) => JT.button(p.label, () => { stack = p.stack.slice(); render(); }, { class: 'btn toggle', 'aria-pressed': 'false', dataset: { preset: p.label } }));
    container.append(
      JT.el('div', { class: 'controls tight' }, [JT.el('span', { class: 'lbl', text: 'Wrap with' }), ...wrap, clear]),
      JT.el('div', { class: 'controls tight presets' }, [JT.el('span', { class: 'lbl', text: 'Presets' }), ...presets]),
    );
    const expr = JT.el('p', { class: 'expr', 'aria-live': 'polite' });
    const sig = JT.el('p', { class: 'sig' });
    const expl = JT.el('div', { class: 'expl' });
    const eqHost = JT.el('div');
    container.append(expr, sig, expl, JT.el('div', { class: 'eq' }, [JT.el('h4', { text: 'PyTorch equivalent, with torch.func' }), eqHost]));

    function render() {
      expr.replaceChildren();
      if (!stack.length) expr.appendChild(JT.el('span', { text: 'loss(params, x, y)' }));
      else {
        [...stack].reverse().forEach((t) => expr.appendChild(JT.el('span', { class: 't', text: t + '(' })));
        expr.appendChild(JT.el('span', { text: 'loss' }));
        expr.appendChild(JT.el('span', { class: 'p', text: ')'.repeat(stack.length) + '(params, x, y)' }));
      }
      const r = evaluate(stack);
      if (r.error) { sig.innerHTML = 'Returns: an error at trace time'; expl.textContent = r.error; expl.classList.add('bad'); }
      else {
        sig.innerHTML = `Returns: <code>${JT.escape(r.outShape)}</code>${r.compiled ? ', compiled' : ''}`;
        expl.classList.remove('bad');
        expl.innerHTML = r.notes.length ? r.notes.map((n) => `<p>${n}</p>`).join('') : '<p>A plain loss of the parameters and one batch. Wrappers apply from the inside out; each returns a new function for the next one to wrap.</p>';
      }
      eqHost.replaceChildren(JT.code(torchEquivalent(stack)));
      const key = stack.join(',');
      presets.forEach((b, i) => b.setAttribute('aria-pressed', String(PRESETS[i].stack.join(',') === key)));
    }
    render();
  });
})();
