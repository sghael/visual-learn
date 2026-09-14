/* Tracing playground: step through what jax.jit records, then call the
   function with different shapes to see the cache, side effects and the
   control-flow rule in action. */
(function () {
  'use strict';
  const JT = window.JT;

  JT.style('tracing', `
    #w-tracing .tgrid { display: grid; grid-template-columns: minmax(0, 1.05fr) minmax(0, 1.25fr) minmax(0, 1fr); gap: 1rem; align-items: start; }
    #w-tracing .pane-body { padding: 0; }
    #w-tracing pre.code { font-size: 0.78rem; min-height: 9.5rem; }
    #w-tracing .fixrow { display: flex; gap: 0.4rem; flex-wrap: wrap; padding: 0.6rem 0.9rem; border-top: 1px solid var(--line); background: var(--surface-2); }
    #w-tracing .stepbar { padding: 0.6rem 0.9rem; border-top: 1px solid var(--line); }
    #w-tracing .stepbar .controls { margin-left: 0; }
    #w-tracing .tracers { padding: 0.55rem 0.9rem; font-family: var(--font-mono); font-size: 0.74rem; color: var(--ink-2); border-bottom: 1px solid var(--line); background: var(--surface-2); min-height: 2.2rem; }
    #w-tracing .tracers b { color: var(--jax-deep); font-weight: 500; }
    #w-tracing .jaxpr { min-height: 9.5rem; }
    #w-tracing .jaxpr .ln.new { background: var(--jax-soft); animation: tr-pop 500ms var(--ease-out); }
    @keyframes tr-pop { from { background: var(--live-soft); } }
    #w-tracing .note { padding: 0.65rem 0.9rem; font-size: 0.85rem; color: var(--ink-2); border-top: 1px solid var(--line); min-height: 4.2rem; line-height: 1.5; }
    #w-tracing .note.err { color: var(--err); }
    #w-tracing .stdout { margin: 0; padding: 0.55rem 0.9rem; font-family: var(--font-mono); font-size: 0.74rem; background: var(--ink); color: #d8d6cf; min-height: 2.6rem; white-space: pre-wrap; word-break: break-word; }
    #w-tracing .stdout .k { color: var(--ink-3); }
    #w-tracing .stdout .t { color: var(--jax-mid); }
    #w-tracing .stdout .v { color: var(--torch-mid); }
    #w-tracing .callbtns { display: flex; flex-wrap: wrap; gap: 0.4rem; padding: 0.7rem 0.9rem; border-bottom: 1px solid var(--line); }
    #w-tracing .callbtns .btn { font-family: var(--font-mono); font-size: 0.75rem; }
    #w-tracing .log { list-style: none; margin: 0; padding: 0.4rem 0; min-height: 6rem; }
    #w-tracing .log li { display: grid; grid-template-columns: 2.2rem 1fr; gap: 0.5rem; padding: 0.35rem 0.9rem; font-size: 0.8rem; line-height: 1.4; border-bottom: 1px solid var(--line); animation: tr-in 400ms var(--ease-out); }
    @keyframes tr-in { from { opacity: 0; transform: translateY(4px); } }
    #w-tracing .log li:last-child { border-bottom: 0; }
    #w-tracing .log .n { font-family: var(--font-mono); color: var(--ink-3); font-size: 0.72rem; padding-top: 0.15rem; }
    #w-tracing .log .sig { font-family: var(--font-mono); font-size: 0.74rem; color: var(--ink-2); }
    #w-tracing .log .miss { color: var(--jax-deep); font-weight: 600; }
    #w-tracing .log .hit { color: var(--live-deep); font-weight: 600; }
    #w-tracing .log .bad { color: var(--err); font-weight: 600; }
    #w-tracing .log .out { font-family: var(--font-mono); font-size: 0.72rem; color: var(--ink-3); display: block; }
    #w-tracing .log li.empty { display: block; color: var(--ink-3); font-size: 0.8rem; padding: 0.6rem 0.9rem; }
    #w-tracing .cache { padding: 0.6rem 0.9rem; border-top: 1px solid var(--line); font-size: 0.78rem; color: var(--ink-2); }
    #w-tracing .cache .sigs { display: flex; flex-wrap: wrap; gap: 0.35rem; margin-top: 0.35rem; }
    #w-tracing .cache .sigs span { font-family: var(--font-mono); font-size: 0.72rem; background: var(--jax-soft); color: var(--jax-deep); border-radius: 6px; padding: 0.15rem 0.45rem; }
    #w-tracing .cache .sigs .none { background: transparent; color: var(--ink-3); padding-left: 0; }
    #w-tracing .torchnote { margin-top: 1rem; }
    #w-tracing .torchnote .pane-body { padding: 0.7rem 0.9rem; font-size: 0.85rem; color: var(--ink-2); line-height: 1.5; }
    #w-tracing .torchnote ul { margin: 0.4rem 0 0; padding-left: 1.1rem; }
    #w-tracing .torchnote li { margin: 0.2rem 0; }
    #w-tracing .torchnote code { font-family: var(--font-mono); font-size: 0.78rem; }
    @media (max-width: 980px) { #w-tracing .tgrid { grid-template-columns: 1fr 1fr; } #w-tracing .col3 { grid-column: 1 / -1; } }
    @media (max-width: 640px) { #w-tracing .tgrid { grid-template-columns: 1fr; } }
  `);

  // ---- examples ------------------------------------------------------
  const H = (n) => `{ lambda ; a:f32[${n}] b:f32[${n}]. let`;
  const EXAMPLES = [
    {
      id: 'pure', label: 'Pure math',
      src: `@jax.jit
def f(x, y):
    a = x * 2.0
    b = jnp.sin(a)
    return b + y`,
      tracers: 'x → <b>Traced&lt;ShapedArray(float32[3])&gt;</b>, y → <b>Traced&lt;ShapedArray(float32[3])&gt;</b>',
      steps: [
        { line: 2, note: 'jit calls f with two tracers. They carry a shape and a dtype and nothing else: no values exist yet.', jaxpr: [H(3)] },
        { line: 3, note: '<code>x * 2.0</code> dispatches the <code>mul</code> primitive on a tracer. JAX records one equation and hands back a new tracer named c.', jaxpr: ['    c:f32[3] = mul a 2.0'] },
        { line: 4, note: '<code>jnp.sin</code> is a thin wrapper over the <code>sin</code> primitive. Another equation, another tracer.', jaxpr: ['    d:f32[3] = sin c'] },
        { line: 5, note: 'The return value is a tracer too, so JAX knows which variable is the output. The jaxpr is closed.', jaxpr: ['    e:f32[3] = add d b', '  in (e,) }'] },
        { line: null, note: 'Trace complete. The jaxpr is lowered to StableHLO and compiled by XLA into one executable, cached under the signature <code>(f32[3], f32[3])</code>. Python is finished with this function until a new signature shows up.', jaxpr: [], done: true },
      ],
      call: (shape, cached) => ({ effects: [], error: null }),
      torch: 'Eager PyTorch dispatches the three tensor operations on every call. In this example they queue three pieces of accelerator work. Eager execution has no compiled graph cache for this function.',
    },
    {
      id: 'effect', label: 'Side effect',
      src: `@jax.jit
def f(x):
    print("tracing:", x)
    return x + 1`,
      tracers: 'x → <b>Traced&lt;ShapedArray(float32[3])&gt;</b>',
      steps: [
        { line: 2, note: 'jit calls f with a tracer for x.', jaxpr: ['{ lambda ; a:f32[3]. let'] },
        { line: 3, note: '<code>print</code> is ordinary Python, not a JAX primitive. It runs right now, during tracing, and what it prints is the tracer itself. Nothing is recorded in the jaxpr.', jaxpr: [], stdout: '<span class="k">tracing:</span> <span class="t">Traced&lt;ShapedArray(float32[3])&gt;with&lt;DynamicJaxprTrace&gt;</span>' },
        { line: 4, note: '<code>x + 1</code> is a primitive, so it is recorded.', jaxpr: ['    b:f32[3] = add a 1.0', '  in (b,) }'] },
        { line: null, note: 'Trace complete and compiled. The executable contains an <code>add</code> and nothing else. The print will not happen again for this signature. For output that survives compilation, use <code>jax.debug.print</code>.', jaxpr: [], done: true },
      ],
      call: (shape, cached) => ({ effects: cached ? [] : [`tracing: Traced<ShapedArray(float32[${shape}])>`], error: null }),
      torch: 'Eager PyTorch executes the <code>print</code> on every call and prints concrete tensor values: <code>tracing: tensor([1., 1., 1.])</code>.',
    },
    {
      id: 'cond', label: 'Python if',
      src: `@jax.jit
def f(x):
    if x.sum() > 0:
        return x
    return -x`,
      fixedSrc: `@jax.jit
def f(x):
    return lax.cond(x.sum() > 0,
                    lambda x: x,
                    lambda x: -x, x)`,
      tracers: 'x → <b>Traced&lt;ShapedArray(float32[3])&gt;</b>',
      steps: [
        { line: 2, note: 'jit calls f with a tracer for x.', jaxpr: ['{ lambda ; a:f32[3]. let'] },
        { line: 3, note: '<code>x.sum()</code> and <code>&gt; 0</code> are primitives and get recorded. Then Python\'s <code>if</code> asks the boolean tracer for a concrete True or False. It has none.', jaxpr: ['    b:f32[] = reduce_sum[axes=(0,)] a', '    c:bool[] = gt b 0.0'] },
        { line: 3, note: '<b>TracerBoolConversionError</b>: attempted boolean conversion of traced array with shape bool[]. The trace is abandoned. JAX cannot record which branch you meant because it never saw the value.', jaxpr: ['    # trace aborted'], error: true },
      ],
      fixedSteps: [
        { line: 2, note: 'jit calls f with a tracer for x.', jaxpr: ['{ lambda ; a:f32[3]. let'] },
        { line: 3, note: 'The predicate is recorded as before.', jaxpr: ['    b:f32[] = reduce_sum[axes=(0,)] a', '    c:bool[] = gt b 0.0'] },
        { line: 4, note: '<code>lax.cond</code> traces <em>both</em> branch functions into sub-jaxprs and records one <code>cond</code> equation. The choice is made on the device, at run time, by the compiled program.', jaxpr: ['    d:f32[3] = cond[', '      branches=(', '        { lambda ; e:f32[3]. let f:f32[3] = neg e in (f,) }', '        { lambda ; g:f32[3]. let  in (g,) }', '      )', '    ] c a'] },
        { line: 5, note: 'The cond result is the output.', jaxpr: ['  in (d,) }'] },
        { line: null, note: 'Trace complete. For elementwise selection, <code>jnp.where(pred, x, -x)</code> is simpler and evaluates both sides; <code>lax.cond</code> runs only one branch. Shapes must match across branches either way.', jaxpr: [], done: true },
      ],
      call: (shape, cached, fixed) => (fixed ? { effects: [], error: null } : { effects: [], error: 'TracerBoolConversionError during tracing' }),
      torch: 'Eager PyTorch evaluates <code>x.sum() &gt; 0</code> to a Python boolean, synchronizing if <code>x</code> is on a GPU, and then runs the selected branch. With <code>torch.compile</code>, this data-dependent branch commonly causes a graph break unless it is expressed with supported structured control flow.',
    },
  ];

  JT.widget('tracing', (container) => {
    const S = JT.stage(container, { title: 'Watch jit trace a function', hint: 'Step through the trace, then call it' });
    let ex = EXAMPLES[0], fixed = false;

    const seg = JT.seg(EXAMPLES.map((e) => ({ value: e.id, label: e.label })), (v) => { ex = EXAMPLES.find((e) => e.id === v); fixed = false; load(); });
    S.controls.appendChild(seg);

    // column 1: code
    const codeHost = JT.el('div');
    const fixRow = JT.el('div', { class: 'fixrow' });
    const stepBar = JT.el('div', { class: 'stepbar' });
    const col1 = JT.el('div', { class: 'pane col1' }, [JT.el('div', { class: 'pane-head', text: 'Your function' }), JT.el('div', { class: 'pane-body' }, [codeHost, fixRow, stepBar])]);

    // column 2: what jit records
    const tracers = JT.el('div', { class: 'tracers' });
    const jaxprPre = JT.el('pre', { class: 'code jaxpr' });
    const note = JT.el('div', { class: 'note' });
    const stdout = JT.el('pre', { class: 'stdout' });
    const col2 = JT.el('div', { class: 'pane jax col2' }, [
      JT.el('div', { class: 'pane-head', html: 'What jit records <span class="sub">jaxpr</span>' }),
      JT.el('div', { class: 'pane-body' }, [tracers, jaxprPre, note, JT.el('div', { class: 'pane-head', style: { borderTop: '1px solid var(--line)', background: 'var(--surface-2)', color: 'var(--ink-2)' }, text: 'Python stdout' }), stdout]),
    ]);

    // column 3: call log
    const callBtns = JT.el('div', { class: 'callbtns' });
    const log = JT.el('ul', { class: 'log' });
    const cacheBox = JT.el('div', { class: 'cache' });
    const torchNote = JT.el('div', { class: 'pane torch torchnote' }, [JT.el('div', { class: 'pane-head', text: 'Same function, eager PyTorch' }), JT.el('div', { class: 'pane-body' })]);
    const col3 = JT.el('div', { class: 'col3' }, [
      JT.el('div', { class: 'pane' }, [JT.el('div', { class: 'pane-head', html: 'Call it <span class="sub">signature → cache</span>' }), JT.el('div', { class: 'pane-body' }, [callBtns, log, cacheBox])]),
      torchNote,
    ]);

    S.body.appendChild(JT.el('div', { class: 'tgrid' }, [col1, col2, col3]));
    S.foot.innerHTML = '<span class="legend"><span><span class="sw" style="background:var(--live-soft);border:1px solid var(--live)"></span>line being traced</span><span><span class="sw" style="background:var(--jax-soft)"></span>equation just recorded</span><span><span class="sw" style="background:var(--err-soft)"></span>trace aborted</span></span>';

    // ---- state -------------------------------------------------------
    let pre = null, stepper = null, cache = new Set(), calls = 0;

    function currentSteps() { return fixed && ex.fixedSteps ? ex.fixedSteps : ex.steps; }

    function renderJaxpr(upto) {
      const steps = currentSteps();
      const lines = [];
      for (let i = 0; i <= upto && i < steps.length; i++) lines.push(...steps[i].jaxpr);
      jaxprPre.innerHTML = lines.length ? JT.highlight(lines.join('\n'), 'jaxpr') : '<span class="ln" style="color:var(--ink-3)">— nothing recorded yet —</span>';
      if (upto >= 0 && steps[upto].jaxpr.length) {
        const all = JT.$$('.ln', jaxprPre);
        all.slice(all.length - steps[upto].jaxpr.length).forEach((l) => l.classList.add('new'));
      }
    }

    function onStep(i) {
      const steps = currentSteps();
      const s = steps[i];
      if (!s) return; // stale call from a stepper that was replaced
      JT.clearLines(pre, 'now'); JT.clearLines(pre, 'err'); JT.clearLines(pre, 'dim');
      if (s.line) JT.markLines(pre, [s.line], s.error ? 'err' : 'now');
      if (s.done) JT.markLines(pre, '*', 'dim');
      note.innerHTML = s.note; note.classList.toggle('err', !!s.error);
      renderJaxpr(i);
      if (s.stdout) stdout.innerHTML = s.stdout;
      tracers.innerHTML = i >= 0 ? ex.tracers : '';
    }
    function onReset() {
      JT.clearLines(pre, 'now'); JT.clearLines(pre, 'err'); JT.clearLines(pre, 'dim');
      note.innerHTML = 'Press <b>Play</b> or step forward to trace this function with abstract inputs of shape (3,).'; note.classList.remove('err');
      renderJaxpr(-1); stdout.innerHTML = ''; tracers.innerHTML = '';
    }

    function renderCache() {
      const sigs = Array.from(cache);
      cacheBox.innerHTML = `<b>Compiled executables in jit's cache</b><div class="sigs">${sigs.length ? sigs.map((s) => `<span>${JT.escape(s)}</span>`).join('') : '<span class="none">none yet</span>'}</div>`;
    }
    function sigFor(n) { return ex.id === 'pure' ? `(f32[${n}], f32[${n}])` : `(f32[${n}])`; }
    function addLog(html) {
      const empty = JT.$('.empty', log); if (empty) empty.remove();
      calls++;
      log.appendChild(JT.el('li', {}, [JT.el('span', { class: 'n', text: '#' + calls }), JT.el('span', { html })]));
      log.scrollTop = log.scrollHeight;
    }
    function callLabel(n) { return ex.id === 'pure' ? `f(ones(${n}), ones(${n}))` : `f(ones(${n}))`; }
    function call(n) {
      const sig = sigFor(n);
      const hit = cache.has(sig);
      const r = ex.call(n, hit, fixed);
      let html = `<span class="sig">${JT.escape(callLabel(n))}</span> · signature <span class="sig">${JT.escape(sig)}</span><br>`;
      if (hit) html += `<span class="hit">cache hit</span> → run the compiled executable. No Python ran.`;
      else if (r.error) html += `<span class="miss">cache miss</span> → trace → <span class="bad">${JT.escape(r.error)}</span>`;
      else { html += `<span class="miss">cache miss</span> → trace + XLA compile (tens of ms) → run`; cache.add(sig); }
      r.effects.forEach((e) => { html += `<span class="out">stdout: ${JT.escape(e)}</span>`; });
      addLog(html);
      renderCache();
    }
    function resetCalls() { cache = new Set(); calls = 0; log.replaceChildren(JT.el('li', { class: 'empty', text: 'No calls yet. Each button calls f with a fresh array of that shape.' })); renderCache(); }

    function load() {
      if (stepper) stepper.pause();
      codeHost.replaceChildren(pre = JT.code(fixed && ex.fixedSrc ? ex.fixedSrc : ex.src));
      fixRow.replaceChildren();
      if (ex.fixedSrc) {
        const asWritten = JT.el('button', { class: 'chip', type: 'button', text: 'as written', 'aria-pressed': String(!fixed) });
        const useCond = JT.el('button', { class: 'chip', type: 'button', text: 'fix with lax.cond', 'aria-pressed': String(fixed) });
        asWritten.addEventListener('click', () => { if (fixed) { fixed = false; load(); } });
        useCond.addEventListener('click', () => { if (!fixed) { fixed = true; load(); } });
        fixRow.append(asWritten, useCond);
        fixRow.hidden = false;
      } else fixRow.hidden = true;
      stepper = JT.stepper({ total: currentSteps().length, interval: 1500, onStep, onReset });
      stepBar.replaceChildren(stepper.el);
      onReset();
      callBtns.replaceChildren(
        JT.el('button', { class: 'btn', type: 'button', text: callLabel(3), onClick: () => call(3) }),
        JT.el('button', { class: 'btn', type: 'button', text: callLabel(4), onClick: () => call(4) }),
        JT.el('button', { class: 'btn', type: 'button', text: callLabel(3) + ' again', onClick: () => call(3) }),
        JT.el('button', { class: 'btn', type: 'button', html: JT.icon('reset') + '<span>clear</span>', onClick: resetCalls }),
      );
      resetCalls();
      JT.$('.pane-body', torchNote).innerHTML = ex.torch;
    }
    load();
    JT.onVisible(container, () => { if (stepper && stepper.i < 0) stepper.play(); }, { threshold: 0.6 });
  });
})();
