/* Hero widget: follow one function call from Python to the accelerator,
   on the JAX route (trace → compile → cache) and the PyTorch route
   (dispatch each op as it is called). */
(function () {
  'use strict';
  const JT = window.JT;

  JT.style('pipeline', `
    #w-pipeline .pipe-grid { display: grid; grid-template-columns: 270px 1fr; gap: 1.25rem; align-items: start; }
    #w-pipeline .pipe-code { border: 1px solid var(--line); border-radius: var(--radius-sm); overflow: hidden; }
    #w-pipeline .pipe-code .pane-head { background: var(--surface-2); color: var(--ink-2); }
    #w-pipeline .pipe-code pre.code { font-size: 0.78rem; }
    #w-pipeline .lane-label { font-weight: 600; font-size: 12px; }
    #w-pipeline .lane-label.jax { fill: var(--jax-deep); }
    #w-pipeline .lane-label.torch { fill: var(--torch-deep); }
    #w-pipeline .box rect { fill: var(--surface); stroke: var(--line-2); stroke-width: 1; transition: fill 220ms, stroke 220ms; }
    #w-pipeline .box text { fill: var(--ink-2); font-size: 11.5px; transition: fill 220ms; }
    #w-pipeline .box.hw rect { fill: var(--surface-2); }
    #w-pipeline .box.on rect { fill: var(--live-soft); stroke: var(--live); }
    #w-pipeline .box.on text { fill: var(--live-deep); font-weight: 600; }
    #w-pipeline .box.jax.done rect { fill: var(--jax-soft); stroke: var(--jax-mid); }
    #w-pipeline .box.jax.done text { fill: var(--jax-deep); }
    #w-pipeline .box.torch.done rect { fill: var(--torch-soft); stroke: var(--torch-mid); }
    #w-pipeline .box.torch.done text { fill: var(--torch-deep); }
    #w-pipeline .box.skip rect { stroke-dasharray: 3 3; opacity: 0.55; }
    #w-pipeline .box.skip text { opacity: 0.55; }
    #w-pipeline .arrow { stroke: var(--line-2); stroke-width: 1.2; fill: none; }
    #w-pipeline .arrow.cache { stroke: var(--jax-mid); stroke-dasharray: 4 3; transition: stroke 220ms; }
    #w-pipeline .arrow.cache.on { stroke: var(--jax); }
    #w-pipeline .cache-label { fill: var(--jax); font-size: 10.5px; font-weight: 500; }
    #w-pipeline .token circle { stroke: #fff; stroke-width: 1.5; }
    #w-pipeline .token text { font-size: 9px; font-weight: 600; fill: #fff; text-anchor: middle; dominant-baseline: central; font-family: var(--font-mono); }
    #w-pipeline .status { font-size: 11px; fill: var(--ink-3); }
    #w-pipeline .status.jax { fill: var(--jax-deep); }
    #w-pipeline .status.torch { fill: var(--torch-deep); }
    #w-pipeline .opq rect { fill: var(--torch-soft); stroke: var(--torch-mid); }
    #w-pipeline .opq text { fill: var(--torch-deep); font-size: 10px; font-family: var(--font-mono); }
    #w-pipeline .opq.gone { opacity: 0.25; }
    #w-pipeline .counts { display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem 1.5rem; }
    #w-pipeline .counts .c { font-size: 0.8125rem; color: var(--ink-2); }
    #w-pipeline .counts .c b { font-weight: 600; }
    #w-pipeline .counts .c.jax b { color: var(--jax-deep); }
    #w-pipeline .counts .c.torch b { color: var(--torch-deep); }
    #w-pipeline .scroller { overflow-x: auto; }
    @media (max-width: 820px) { #w-pipeline .pipe-grid { grid-template-columns: 1fr; } #w-pipeline .counts { grid-template-columns: 1fr; } #w-pipeline .scroller svg { min-width: 760px; } }
  `);

  JT.widget('pipeline', (container) => {
    const S = JT.stage(container, { title: 'Follow one function call to the silicon', hint: 'Same three operations, two routes' });

    const src = `@jax.jit
def step(w, x):
    h = jnp.tanh(x @ w)  # 2 ops
    return h.sum()       # 1 op

step(w, x)   # first call
step(w, x)   # second call`;
    const code = JT.el('div', { class: 'pipe-code' }, [
      JT.el('div', { class: 'pane-head', text: 'The function under the microscope' }),
      JT.code(src),
    ]);

    // ---- SVG scene -------------------------------------------------
    const W = 960, H = 386;
    const svg = JT.svg('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'Two execution pipelines: JAX traces, compiles and caches; PyTorch dispatches each op.' });

    const boxes = {};
    function box(id, x, y, w, label, cls, tip) {
      const g = JT.svg('g', { class: 'box ' + cls, 'data-tip': tip, tabindex: 0 });
      g.appendChild(JT.svg('rect', { x, y, width: w, height: 44, rx: 9 }));
      const lines = label.split('\n');
      lines.forEach((t, i) => g.appendChild(JT.svg('text', { x: x + w / 2, y: y + 22 + (i - (lines.length - 1) / 2) * 13, 'text-anchor': 'middle', 'dominant-baseline': 'central', text: t })));
      svg.appendChild(g);
      boxes[id] = { g, x, y, w, cx: x + w / 2, cy: y + 22 };
      return g;
    }
    function arrow(x1, y1, x2, y2, cls) {
      const p = JT.svg('path', { class: 'arrow ' + (cls || ''), d: `M${x1},${y1} L${x2},${y2}`, 'marker-end': 'url(#pipe-arrow)' });
      svg.appendChild(p); return p;
    }
    const defs = JT.svg('defs', {}, [
      JT.svg('marker', { id: 'pipe-arrow', viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' }, [
        JT.svg('path', { d: 'M0,1 L9,5 L0,9 z', fill: JT.colors.line2 }),
      ]),
    ]);
    svg.appendChild(defs);

    // lane labels
    const JY = 96, TY = 262;
    svg.appendChild(JT.svg('text', { class: 'lane-label jax', x: 16, y: 24, text: 'JAX  ·  trace once, compile once, then reuse' }));
    svg.appendChild(JT.svg('text', { class: 'lane-label torch', x: 16, y: TY - 64, text: 'PyTorch (eager)  ·  dispatch every op, every call' }));

    // JAX lane boxes
    const jx = [16, 168, 322, 490, 656, 820];
    box('j-py', jx[0], JY, 128, 'Python call', 'jax', '<b>Python call</b><br>step(w, x) is invoked. jit checks its cache using the shapes, dtypes and pytree structure of the arguments.');
    box('j-trace', jx[1], JY, 130, 'trace → jaxpr', 'jax', '<b>Tracing</b><br>The function runs once with tracers (shape + dtype, no values). Every primitive it touches becomes an equation in a jaxpr.');
    box('j-lower', jx[2], JY, 146, 'lower → StableHLO', 'jax', '<b>Lowering</b><br>The jaxpr is translated into StableHLO, the MLIR dialect XLA consumes.');
    box('j-xla', jx[3], JY, 144, 'XLA compile', 'jax', '<b>XLA</b><br>Whole-program optimization: fusion, layout assignment, buffer planning, scheduling. Emits a backend-specific executable.');
    box('j-exe', jx[4], JY, 140, 'cached\nexecutable', 'jax', '<b>Executable</b><br>Kept in a cache keyed by the call signature. Later calls with the same shapes skip straight here.');
    box('j-hw', jx[5], JY, 124, 'TPU · GPU · CPU', 'jax hw', '<b>Run</b><br>PJRT hands the executable to the device. The whole function is a handful of fused kernels.');
    for (let i = 0; i < jx.length - 1; i++) arrow(jx[i] + boxes[Object.keys(boxes)[i]].w, JY + 22, jx[i + 1] - 2, JY + 22);
    // cache bypass arc from Python call to executable
    const cacheArc = JT.svg('path', { class: 'arrow cache', d: `M${boxes['j-py'].cx},${JY - 2} C ${boxes['j-py'].cx},${JY - 40} ${boxes['j-exe'].cx},${JY - 40} ${boxes['j-exe'].cx},${JY - 2}`, 'marker-end': 'url(#pipe-arrow)' });
    svg.appendChild(cacheArc);
    svg.appendChild(JT.svg('text', { class: 'cache-label', x: (boxes['j-py'].cx + boxes['j-exe'].cx) / 2, y: JY - 50, 'text-anchor': 'middle', text: 'cache hit: same shapes → skip trace and compile' }));
    const jStatus = JT.svg('text', { class: 'status jax', x: 16, y: JY + 72, text: '' });
    svg.appendChild(jStatus);

    // PyTorch lane boxes
    const tx = [16, 300, 520, 820];
    box('t-py', tx[0], TY, 128, 'Python call', 'torch', '<b>Python call</b><br>step(w, x) runs as ordinary Python. Each tensor operation is a separate call into the C++ core.');
    box('t-disp', tx[1], TY, 150, 'dispatcher', 'torch', '<b>Dispatcher</b><br>Picks the kernel for this op from its dispatch keys: device (CUDA), dtype, autograd, and so on.');
    box('t-kern', tx[2], TY, 190, 'ATen op → CUDA kernel', 'torch', '<b>Kernel launch</b><br>A pre-compiled kernel (cuBLAS for matmul, a pointwise kernel for tanh, a reduction for sum) is queued on a CUDA stream.');
    box('t-hw', tx[3], TY, 124, 'GPU', 'torch hw', '<b>Run</b><br>The GPU executes the kernel asynchronously while Python is already dispatching the next op.');
    arrow(tx[0] + 128, TY + 22, tx[1] - 2, TY + 22);
    arrow(tx[1] + 150, TY + 22, tx[2] - 2, TY + 22);
    arrow(tx[2] + 190, TY + 22, tx[3] - 2, TY + 22);
    // return arrow from GPU lane back to python (loop)
    svg.appendChild(JT.svg('path', { class: 'arrow', d: `M${tx[3] + 62},${TY + 46} C ${tx[3] + 62},${TY + 84} ${tx[0] + 64},${TY + 84} ${tx[0] + 64},${TY + 48}`, 'marker-end': 'url(#pipe-arrow)' }));
    svg.appendChild(JT.svg('text', { class: 'status', x: W / 2, y: TY + 94, 'text-anchor': 'middle', text: 'next op' }));
    // op queue chips
    const OPS = ['matmul', 'tanh', 'sum'];
    // draw queue as three small tags above the Python box on torch lane
    const opTags = OPS.map((op, i) => {
      const x = 16 + i * 44, y = TY - 52 + 12;
      const g = JT.svg('g', { class: 'opq', 'data-tip': 'Operations queued for this call, dispatched one at a time in program order.' });
      g.appendChild(JT.svg('rect', { x, y, width: 40, height: 18, rx: 5 }));
      g.appendChild(JT.svg('text', { x: x + 20, y: y + 9, 'text-anchor': 'middle', 'dominant-baseline': 'central', text: op }));
      svg.appendChild(g);
      return g;
    });
    const tStatus = JT.svg('text', { class: 'status torch', x: 16, y: TY + 112, text: '' });
    svg.appendChild(tStatus);

    // tokens
    function token(color, label) {
      const g = JT.svg('g', { class: 'token', opacity: 0 });
      g.appendChild(JT.svg('circle', { r: 11, fill: color }));
      g.appendChild(JT.svg('text', { text: label }));
      svg.appendChild(g);
      return g;
    }
    const jTok = token(JT.colors.jax, 'f');
    const tToks = OPS.map((o, i) => token(JT.colors.torch, ['⋅', 'th', 'Σ'][i]));

    function place(tok, x, y) { tok.setAttribute('transform', `translate(${x},${y})`); }
    function moveTo(tok, x, y, ms, rid) {
      return new Promise((resolve) => {
        const m = /translate\(([-\d.]+),([-\d.]+)\)/.exec(tok.getAttribute('transform') || '');
        const x0 = m ? +m[1] : x, y0 = m ? +m[2] : y;
        if (JT.reducedMotion || ms <= 0) { place(tok, x, y); resolve(); return; }
        const t0 = performance.now();
        const ease = (t) => 1 - Math.pow(1 - t, 3);
        (function frame(now) {
          if (rid !== undefined && stale(rid)) { resolve(); return; }
          const t = Math.min(1, (now - t0) / ms), e = ease(t);
          place(tok, JT.lerp(x0, x, e), JT.lerp(y0, y, e));
          if (t < 1) requestAnimationFrame(frame); else resolve();
        })(t0);
      });
    }
    function moveAlong(tok, pathEl, ms, rid) {
      return new Promise((resolve) => {
        const L = pathEl.getTotalLength();
        if (JT.reducedMotion) { const p = pathEl.getPointAtLength(L); place(tok, p.x, p.y); resolve(); return; }
        const t0 = performance.now();
        (function frame(now) {
          if (rid !== undefined && stale(rid)) { resolve(); return; }
          const t = Math.min(1, (now - t0) / ms);
          const p = pathEl.getPointAtLength(L * (1 - Math.pow(1 - t, 2)));
          place(tok, p.x, p.y);
          if (t < 1) requestAnimationFrame(frame); else resolve();
        })(t0);
      });
    }

    // ---- state & runs ----------------------------------------------
    const counts = { traces: 0, compiles: 0, jaxRuns: 0, launches: 0, torchCalls: 0 };
    let running = false, runId = 0, calls = 0;
    const stale = (rid) => rid !== runId;
    /** Sleep, then report whether this run still owns the widget. Every DOM write after a wait is gated on it. */
    const pause = async (ms, rid) => { await JT.wait(ms); return !stale(rid); };
    const foot = S.foot;
    const cJ = JT.el('div', { class: 'c jax' });
    const cT = JT.el('div', { class: 'c torch' });
    foot.appendChild(JT.el('div', { class: 'counts' }, [cJ, cT]));
    function renderCounts() {
      cJ.innerHTML = `<b>JAX</b> · traced <b>${counts.traces}</b>× · compiled <b>${counts.compiles}</b>× · executed <b>${counts.jaxRuns}</b>×`;
      cT.innerHTML = `<b>PyTorch</b> · calls <b>${counts.torchCalls}</b> · kernel launches <b>${counts.launches}</b> (3 per call)`;
    }
    renderCounts();

    const btnFirst = JT.el('button', { class: 'btn primary', type: 'button', html: JT.icon('play') + '<span>Run first call</span>' });
    const btnAgain = JT.el('button', { class: 'btn', type: 'button', html: JT.icon('next') + '<span>Call again</span>', disabled: true });
    const btnReset = JT.el('button', { class: 'btn', type: 'button', html: JT.icon('reset') + '<span>Reset</span>' });
    S.controls.append(btnFirst, btnAgain, btnReset);

    const setBox = (id, cls) => { const g = boxes[id].g; g.classList.remove('on', 'done', 'skip'); if (cls) cls.split(' ').forEach((c) => g.classList.add(c)); };
    const clearBoxes = () => Object.keys(boxes).forEach((id) => setBox(id, ''));

    async function runJax(first, rid) {
      const step = JT.reducedMotion ? 0 : 420;
      jTok.setAttribute('opacity', 1);
      place(jTok, boxes['j-py'].cx, boxes['j-py'].cy);
      setBox('j-py', 'on'); jStatus.textContent = first ? 'jit: no executable for (f32[512,512], f32[512,512]) → trace' : 'jit: signature seen before → cache hit';
      if (!(await pause(step, rid))) return;
      if (first) {
        setBox('j-py', 'done');
        for (const [id, msg, key] of [['j-trace', 'tracing with abstract inputs; Python side effects happen now', 'traces'], ['j-lower', 'emitting StableHLO', null], ['j-xla', 'XLA fusing tanh into the matmul epilogue, planning buffers', 'compiles'], ['j-exe', 'executable stored under the call signature', null]]) {
          await moveTo(jTok, boxes[id].cx, boxes[id].cy, step * 1.6, rid);
          if (stale(rid)) return;
          setBox(id, 'on'); jStatus.textContent = msg;
          if (key) { counts[key]++; renderCounts(); }
          if (!(await pause(step * (id === 'j-xla' ? 2.2 : 1), rid))) return;
          setBox(id, 'done');
        }
      } else {
        ['j-trace', 'j-lower', 'j-xla'].forEach((id) => setBox(id, 'skip'));
        cacheArc.classList.add('on');
        await moveAlong(jTok, cacheArc, step * 2.4, rid);
        if (stale(rid)) return;
        setBox('j-py', 'done'); setBox('j-exe', 'on'); jStatus.textContent = 'reusing the compiled executable; nothing in Python runs';
        if (!(await pause(step, rid))) return;
        setBox('j-exe', 'done');
      }
      await moveTo(jTok, boxes['j-hw'].cx, boxes['j-hw'].cy, step * 1.6, rid);
      if (stale(rid)) return;
      setBox('j-hw', 'on'); counts.jaxRuns++; renderCounts(); jStatus.textContent = 'device runs the whole function as fused kernels; result returned asynchronously';
      if (!(await pause(step * 1.4, rid))) return;
      setBox('j-hw', 'done'); cacheArc.classList.remove('on');
      jTok.setAttribute('opacity', 0);
    }

    async function runTorch(rid) {
      const step = JT.reducedMotion ? 0 : 300;
      opTags.forEach((g) => g.classList.remove('gone'));
      for (let i = 0; i < OPS.length; i++) {
        if (stale(rid)) return;
        const tok = tToks[i];
        tok.setAttribute('opacity', 1);
        place(tok, boxes['t-py'].cx, boxes['t-py'].cy);
        setBox('t-py', 'on'); tStatus.textContent = `Python evaluates \`${OPS[i]}\` and calls into torch`;
        opTags[i].classList.add('gone');
        if (!(await pause(step, rid))) return;
        setBox('t-py', 'done');
        for (const [id, msg] of [['t-disp', `dispatcher resolves ${OPS[i]} for (CUDA, float32, autograd)`], ['t-kern', `kernel for ${OPS[i]} queued on the CUDA stream`], ['t-hw', `GPU runs the ${OPS[i]} kernel; output tensor written to HBM`]]) {
          await moveTo(tok, boxes[id].cx, boxes[id].cy, step * 1.6, rid);
          if (stale(rid)) return;
          setBox(id, 'on'); tStatus.textContent = msg;
          if (id === 't-kern') { counts.launches++; renderCounts(); }
          if (!(await pause(step, rid))) return;
          setBox(id, 'done');
        }
        tok.setAttribute('opacity', 0);
      }
      if (stale(rid)) return;
      tStatus.textContent = 'three launches, three intermediate tensors in memory; next call does it all again';
    }

    async function run(first) {
      if (running) return;
      const rid = ++runId; running = true;
      btnFirst.disabled = true; btnAgain.disabled = true;
      calls++;
      counts.torchCalls++; renderCounts();
      clearBoxes();
      await Promise.all([runJax(first, rid), runTorch(rid)]);
      if (stale(rid)) return; // a reset happened mid-run; the newer run owns the UI
      running = false; btnAgain.disabled = false; btnFirst.disabled = true;
    }
    function reset() {
      runId++; running = false; calls = 0;
      Object.assign(counts, { traces: 0, compiles: 0, jaxRuns: 0, launches: 0, torchCalls: 0 }); renderCounts();
      clearBoxes(); cacheArc.classList.remove('on');
      jTok.setAttribute('opacity', 0); tToks.forEach((t) => t.setAttribute('opacity', 0));
      opTags.forEach((g) => g.classList.remove('gone'));
      jStatus.textContent = ''; tStatus.textContent = '';
      btnFirst.disabled = false; btnAgain.disabled = true;
    }
    btnFirst.addEventListener('click', () => run(true));
    btnAgain.addEventListener('click', () => run(false));
    btnReset.addEventListener('click', reset);

    S.body.appendChild(JT.el('div', { class: 'pipe-grid' }, [code, JT.el('div', { class: 'scroller' }, svg)]));
    JT.bindTips(container);
    JT.onVisible(container, () => { if (!running && calls === 0) run(true); }, { threshold: 0.5 });
  });
})();
