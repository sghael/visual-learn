/* Figure: one gradient, two machines, one step control.
   L(w) = (sin(1.5 w) - 0.5)^2. Each step moves PyTorch's autograd tape and
   JAX's gradient jaxpr through the same operation of the same graph.
   The JAX jaxpr is jax.make_jaxpr(jax.grad(L))(w) from JAX 0.11.2 with
   literal type annotations removed. It rests on the finished computation. */
(function () {
  'use strict';
  const JT = window.JT;
  const X = 1.5, Y = 0.5;
  const f3 = JT.f3;

  function values(w) {
    const a = w * X, b = Math.sin(a), c = b - Y, L = c * c;
    const dc = 2 * c, db = dc, da = db * Math.cos(a), dw = da * X;
    return { w, a, b, c, L, dL: 1, dc, db, da, dw, cosa: Math.cos(a) };
  }

  // graph nodes, left to right: input w and the four operations
  const NODES = [
    { id: 'w', op: 'w', val: 'w', grad: 'dw' },
    { id: 'mul', op: '×', val: 'a', grad: 'da', konst: 'x = 1.5' },
    { id: 'sin', op: 'sin', val: 'b', grad: 'db' },
    { id: 'sub', op: '−', val: 'c', grad: 'dc', konst: 'y = 0.5' },
    { id: 'sq', op: '²', val: 'L', grad: 'dL' },
  ];

  const TAPE = [
    { fn: 'MulBackward0', saves: 'saves x' },
    { fn: 'SinBackward0', saves: 'saves a' },
    { fn: 'SubBackward0', saves: 'saves nothing' },
    { fn: 'PowBackward0', saves: 'saves c' },
  ];

  // jaxpr lines contributed by each step (forward 0-3, backward 4-7)
  const JAXPR_HEAD = '{ lambda ; a:f32[]. let';
  const JAXPR = [
    ['    b:f32[] = mul a 1.5'],
    ['    c:f32[] = sin b', '    d:f32[] = cos b'],
    ['    e:f32[] = sub c 0.5'],
    ['    _:f32[] = integer_pow[y=2] e'],
    ['    f:f32[] = mul 2.0 e', '    g:f32[] = mul 1.0 f'],
    [],
    ['    h:f32[] = mul g d'],
    ['    i:f32[] = mul h 1.5', '  in (i,) }'],
  ];

  const NOTES = [
    (v) => `Forward, <code>w * x</code> = ${f3(v.a)}. PyTorch computes it now and appends <code>MulBackward0</code>, which keeps x. JAX records <code>mul</code> with no value attached.`,
    (v) => `Forward, <code>sin</code> = ${f3(v.b)}. <code>SinBackward0</code> keeps its input a, because the derivative cos(a) needs it. JAX records <code>sin b</code> and also <code>cos b</code>, the value its backward pass will need.`,
    (v) => `Forward, subtract y = ${f3(v.c)}. The derivative of a subtraction is 1, so <code>SubBackward0</code> keeps nothing.`,
    (v) => `Forward, square: L = ${f3(v.L)}. In JAX the loss itself is not needed for the gradient, so its variable is <code>_</code>.`,
    (v) => `Backward starts from ∂L/∂L = 1. The square's derivative is 2c: ∂L/∂c = ${f3(v.dc)}. PyTorch runs <code>PowBackward0</code> now; JAX appends <code>mul 2.0 e</code> and the seed <code>mul 1.0</code>.`,
    (v) => `Subtraction passes the gradient through unchanged: ∂L/∂b = ${f3(v.db)}. JAX needs no equation for it.`,
    (v) => `Multiply by the saved cos(a) = ${f3(v.cosa)}: ∂L/∂a = ${f3(v.da)}.`,
    (v) => `Multiply by x = 1.5: ∂L/∂w = ${f3(v.dw)}. PyTorch adds it into <code>w.grad</code>; the JAX jaxpr now returns it.`,
    (v) => `Done. PyTorch stored ${f3(v.dw)} in <code>w.grad</code> and freed its graph. JAX built a function whose equations hold for every w; calling it at w = ${f3(v.w)} returns ${f3(v.dw)}.`,
  ];
  const TOTAL = NOTES.length; // 9 steps: 4 forward, 4 backward, done

  JT.widget('autodiff', (container) => {
    container.classList.add('autodiff');
    let w = 0.8, width = 0;

    const range = JT.el('input', { type: 'range', min: -3, max: 3, step: 0.05, value: w, 'aria-label': 'Weight w', id: 'ad-w' });
    const out = JT.el('output', { for: 'ad-w', text: f3(w) });
    range.addEventListener('input', () => { w = +range.value; out.value = f3(w); render(); });
    const stepper = JT.stepper({ total: TOTAL, interval: 1400, onStep: render, onReset: render, noun: 'step' });
    container.appendChild(JT.el('div', { class: 'controls' }, [JT.control('w', [range, out]), stepper.el]));

    const graphHost = JT.el('div');
    const note = JT.el('p', { class: 'note', 'aria-live': 'polite' });
    const torchCode = JT.code(`L = (torch.sin(w * x) - y) ** 2
L.backward()
w.grad`, { class: 'torch-code' });
    const tape = JT.el('ol', { class: 'tape' });
    const torchResult = JT.el('p', { class: 'result' });
    const jaxCode = JT.code(`dL = jax.grad(L)
jax.make_jaxpr(dL)(w)
dL(w)`);
    const jaxprHost = JT.el('div');
    const jaxResult = JT.el('p', { class: 'result' });

    container.append(graphHost, note, JT.el('div', { class: 'cols' }, [
      JT.el('div', {}, [JT.el('h4', { html: '<span class="torch-c">PyTorch</span> <span class="sub">records a tape as it runs</span>' }), torchCode, tape, torchResult]),
      JT.el('div', {}, [JT.el('h4', { html: '<span class="jax-c">JAX</span> <span class="sub">builds a gradient function</span>' }), jaxCode, jaxprHost, jaxResult]),
    ]));

    function drawGraph(v, i) {
      const W = Math.max(300, Math.min(width || 560, 620));
      const G = 66, top = 22, ny = 64, r = 17;
      const span = (W - G - 24) / (NODES.length - 1);
      const nx = (k) => G + 4 + k * span;
      const fwdDone = i < 0 ? -1 : Math.min(i, 3);      // last op whose forward value is known
      const bwdFrom = i < 4 ? 99 : 4 - Math.min(i - 3, 4); // leftmost node whose gradient is known
      const activeNode = i < 0 ? -1 : i < 4 ? i + 1 : i < 8 ? 4 - (i - 4) : -1;
      const svg = JT.svg('svg', { width: W, height: 146, viewBox: `0 0 ${W} 146`, role: 'img', 'aria-label': 'Computation graph of L(w) with forward values and gradients' });
      svg.appendChild(JT.svg('text', { class: 'label muted', x: 0, y: ny + 44, text: 'value' }));
      svg.appendChild(JT.svg('text', { class: 'label muted', x: 0, y: ny + 66, text: 'gradient' }));
      for (let k = 0; k < NODES.length - 1; k++) svg.appendChild(JT.svg('line', { class: 'edge', x1: nx(k) + r, y1: ny, x2: nx(k + 1) - r, y2: ny }));
      NODES.forEach((n, k) => {
        if (n.konst) {
          svg.appendChild(JT.svg('line', { class: 'edge', x1: nx(k), y1: top + 6, x2: nx(k), y2: ny - r }));
          svg.appendChild(JT.svg('text', { class: 'const', x: nx(k), y: top, 'text-anchor': 'middle', text: n.konst }));
        }
        const g = JT.svg('g', { class: 'node' + (k === activeNode ? ' on' : '') }, [
          JT.svg('circle', { cx: nx(k), cy: ny, r }),
          JT.svg('text', { x: nx(k), y: ny + 4.5, 'text-anchor': 'middle', text: n.op }),
        ]);
        svg.appendChild(g);
        const known = k === 0 || k - 1 <= fwdDone;
        svg.appendChild(JT.svg('text', { class: 'val' + (known ? '' : ' pending'), x: nx(k), y: ny + 44, 'text-anchor': 'middle', text: known ? f3(v[n.val]) : '·' }));
        const gk = k >= bwdFrom;
        svg.appendChild(JT.svg('text', { class: gk ? 'grad' : 'val pending', x: nx(k), y: ny + 66, 'text-anchor': 'middle', text: gk ? f3(v[n.grad]) : '·' }));
      });
      graphHost.replaceChildren(svg);
    }

    function render() {
      const i = stepper.i;
      const v = values(w);
      drawGraph(v, i);
      note.innerHTML = i < 0 ? 'Nothing has run. Press Play or Step to run the forward pass, then the backward pass.' : NOTES[i](v);

      // PyTorch
      JT.clearLines(torchCode, 'now');
      if (i >= 0) JT.markLines(torchCode, [i < 4 ? 1 : i < 8 ? 2 : 3], 'now');
      const nTape = i < 0 ? 0 : Math.min(i + 1, 4);
      tape.replaceChildren();
      if (!nTape) tape.appendChild(JT.el('li', { class: 'empty', text: 'Tape: empty until the forward pass runs.' }));
      for (let k = nTape - 1; k >= 0; k--) {
        const t = TAPE[k];
        const onK = i >= 4 && i < 8 ? 3 - (i - 4) : -1;
        const cls = k === onK ? 'on' : (i >= 4 && (i >= 8 || k > onK) ? 'popped' : '');
        tape.appendChild(JT.el('li', { class: cls }, [JT.el('span', { text: t.fn }), JT.el('span', { class: 's', text: t.saves })]));
      }
      torchResult.innerHTML = i >= 7 ? `w.grad = <b>tensor(${f3(v.dw)})</b>` : 'w.grad = None';

      // JAX
      JT.clearLines(jaxCode, 'now');
      if (i >= 0) JT.markLines(jaxCode, [i < 8 ? 2 : 3], 'now');
      if (i < 0) jaxprHost.replaceChildren(JT.placeholder('grad(L) has not been traced yet'));
      else {
        const lines = [JAXPR_HEAD];
        for (let k = 0; k <= Math.min(i, 7); k++) lines.push(...JAXPR[k]);
        const pre = JT.code(lines.join('\n'), { lang: 'jaxpr' });
        const fresh = i < 8 ? JAXPR[i].length : 0;
        if (fresh) JT.$$('.ln', pre).slice(-fresh).forEach((l) => l.classList.add('new'));
        jaxprHost.replaceChildren(pre);
      }
      jaxResult.innerHTML = i >= 8 ? `dL(${f3(w)}) = <b>${f3(v.dw)}</b>` : 'dL(w) not called yet';
    }

    JT.onWidth(container, (wd) => { width = wd; render(); });
    stepper.goto(TOTAL - 1); // rest on the finished computation
  });
})();
