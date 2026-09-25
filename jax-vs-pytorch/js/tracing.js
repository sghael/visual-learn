/* Figure: what jax.jit records when it traces a function, and what later
   calls do with the cache. Jaxprs are the output of jax.make_jaxpr in
   JAX 0.11.2 with literal type annotations and some parameters removed.
   The resting state is the finished trace plus a few calls; nothing moves
   until the reader presses Play or Step. */
(function () {
  'use strict';
  const JT = window.JT;

  const EXAMPLES = [
    {
      id: 'pure', label: 'Pure math', arity: 2,
      src: `@jax.jit
def f(x, y):
    a = x * 2.0
    b = jnp.sin(a)
    return b + y`,
      tracers: 'x, y → JitTracer(float32[3])',
      steps: [
        { lines: [2], note: 'jit calls <code>f</code> with two tracers. Each carries a shape and a dtype, float32[3], and no values.', jaxpr: ['{ lambda ; a:f32[3] b:f32[3]. let'] },
        { lines: [3], note: '<code>x * 2.0</code> applies the <code>mul</code> primitive to a tracer. JAX records one equation and returns a new tracer, <code>c</code>.', jaxpr: ['    c:f32[3] = mul a 2.0'] },
        { lines: [4], note: '<code>jnp.sin</code> is a thin wrapper over the <code>sin</code> primitive: one more equation.', jaxpr: ['    d:f32[3] = sin c'] },
        { lines: [5], note: 'The return value is a tracer too, so JAX knows which variable is the output, and the jaxpr is closed.', jaxpr: ['    e:f32[3] = add d b', '  in (e,) }'] },
        { lines: [], done: true, note: 'Trace complete: three equations. JAX lowers the jaxpr to StableHLO, XLA compiles it, and the executable is cached under the signature <code>(f32[3], f32[3])</code>.', jaxpr: [] },
      ],
      call: () => ({ effects: [], error: null }),
      eager: 'runs the three tensor operations on every call and keeps no compiled cache for this function.',
    },
    {
      id: 'effect', label: 'Side effect', arity: 1,
      src: `@jax.jit
def f(x):
    print("tracing:", x)
    return x + 1`,
      tracers: 'x → JitTracer(float32[3])',
      steps: [
        { lines: [2], note: 'jit calls <code>f</code> with a tracer for <code>x</code>.', jaxpr: ['{ lambda ; a:f32[3]. let'] },
        { lines: [3], note: '<code>print</code> is ordinary Python, not a JAX primitive. It runs now, during tracing, and prints the tracer. Nothing is recorded.', jaxpr: [], stdout: true },
        { lines: [4], note: '<code>x + 1</code> applies the <code>add</code> primitive, so it is recorded.', jaxpr: ['    b:f32[3] = add a 1.0', '  in (b,) }'] },
        { lines: [], done: true, note: 'Trace complete. The executable contains one <code>add</code> and no print, so later calls with this signature print nothing. <code>jax.debug.print</code> prints from inside the compiled program.', jaxpr: [] },
      ],
      call: (n, hit) => ({ effects: hit ? [] : [`tracing: JitTracer(float32[${n}])`], error: null }),
      eager: 'runs the <code>print</code> on every call and prints concrete values: <code>tracing: tensor([1., 1., 1.])</code>.',
    },
    {
      id: 'cond', label: 'Python if', arity: 1,
      src: `@jax.jit
def f(x):
    if x.sum() > 0:
        return x
    return -x`,
      tracers: 'x → JitTracer(float32[3])',
      steps: [
        { lines: [2], note: 'jit calls <code>f</code> with a tracer for <code>x</code>.', jaxpr: ['{ lambda ; a:f32[3]. let'] },
        { lines: [3], note: '<code>x.sum()</code> and <code>&gt; 0</code> are primitives, so they are recorded. Then the Python <code>if</code> asks the boolean tracer for <code>True</code> or <code>False</code>.', jaxpr: ['    b:f32[] = reduce_sum[axes=(0,)] a', '    c:bool[] = gt b 0.0'] },
        { lines: [3], error: true, note: '<code>TracerBoolConversionError</code>: attempted boolean conversion of a traced array with shape <code>bool[]</code>. The tracer has no value, so the trace stops and nothing is compiled.', jaxpr: ['    # trace stops here'] },
      ],
      fixedSrc: `@jax.jit
def f(x):
    return lax.cond(x.sum() > 0,
                    lambda x: x,
                    lambda x: -x, x)`,
      fixedSteps: [
        { lines: [2], note: 'jit calls <code>f</code> with a tracer for <code>x</code>.', jaxpr: ['{ lambda ; a:f32[3]. let'] },
        { lines: [3], note: 'The predicate is recorded as before, then converted to an integer branch index, 0 or 1.', jaxpr: ['    b:f32[] = reduce_sum[axes=(0,)] a', '    c:bool[] = gt b 0.0', '    d:i32[] = convert_element_type[new_dtype=int32] c'] },
        { lines: [3, 4, 5], note: '<code>lax.cond</code> traces both branch functions into sub-jaxprs and records one <code>cond</code> equation. The compiled program picks the branch at run time.', jaxpr: ['    e:f32[3] = cond[', '      branches=(', '        { lambda ; f:f32[3]. let g:f32[3] = neg f in (g,) }', '        { lambda ; h:f32[3]. let  in (h,) }', '      )', '    ] d a', '  in (e,) }'] },
        { lines: [], done: true, note: 'Trace complete. <code>lax.cond</code> runs one branch; <code>jnp.where(x.sum() &gt; 0, x, -x)</code> would compute both and select. Both branches must return the same shape.', jaxpr: [] },
      ],
      call: (n, hit, fixed) => (fixed ? { effects: [], error: null } : { effects: [], error: 'TracerBoolConversionError' }),
      eager: 'evaluates <code>x.sum() &gt; 0</code> to a Python <code>bool</code> (waiting for the GPU if <code>x</code> lives there) and runs one branch. Under <code>torch.compile</code> this data-dependent branch usually causes a graph break.',
    },
  ];

  JT.widget('tracing', (container) => {
    container.classList.add('tracing');
    let ex = EXAMPLES[0], fixed = false, stepper = null, pre = null;
    let cache = new Set(), calls = 0;

    // controls: example, version (Python if only), stepper
    const exSeg = JT.seg(EXAMPLES.map((e) => ({ value: e.id, label: e.label })), (v) => { ex = EXAMPLES.find((e) => e.id === v); fixed = false; load(); }, ex.id, 'Example');
    const fixSeg = JT.seg([{ value: 'if', label: 'as written' }, { value: 'cond', label: 'with lax.cond' }], (v) => { fixed = v === 'cond'; load(); }, 'if', 'Version');
    const fixCtl = JT.control('Version', [fixSeg]);
    const stepHost = JT.el('span', { class: 'control' });
    const controls = JT.el('div', { class: 'controls' }, [JT.control('Example', [exSeg]), fixCtl, stepHost]);

    // trace row
    const codeHost = JT.el('div');
    const tracers = JT.el('p', { class: 'tracers' });
    const jaxprHost = JT.el('div');
    const note = JT.el('p', { class: 'note', 'aria-live': 'polite' });
    const stdout = JT.el('p', { class: 'stdout' });

    // call row
    const callBtns = JT.el('div', { class: 'controls' });
    const log = JT.el('ol', { class: 'log', 'aria-live': 'polite' });
    const cacheLine = JT.el('p', { class: 'cache' });
    const eager = JT.el('p', { class: 'eager' });

    container.append(
      controls,
      JT.el('div', { class: 'grid2' }, [
        JT.el('div', {}, [JT.el('h4', { text: 'Your function' }), codeHost]),
        JT.el('div', {}, [JT.el('h4', { html: 'What <code>jit</code> records <span class="sub">(the jaxpr)</span>' }), tracers, jaxprHost, note, JT.el('h4', { text: 'Python output during tracing' }), stdout]),
        JT.el('div', {}, [JT.el('h4', { text: 'Call the jitted function' }), callBtns]),
        JT.el('div', {}, [JT.el('h4', { text: 'Call log' }), log, cacheLine]),
      ]),
      eager,
    );

    const steps = () => (fixed && ex.fixedSteps ? ex.fixedSteps : ex.steps);

    function renderJaxpr(upto) {
      const s = steps(), lines = [];
      for (let i = 0; i <= upto; i++) lines.push(...s[i].jaxpr);
      if (!lines.length) { jaxprHost.replaceChildren(JT.placeholder('nothing recorded yet')); return; }
      const p = JT.code(lines.join('\n'), { lang: 'jaxpr' });
      const fresh = upto >= 0 ? s[upto].jaxpr.length : 0;
      if (fresh && !s[upto].done) JT.$$('.ln', p).slice(-fresh).forEach((l) => l.classList.add('new'));
      jaxprHost.replaceChildren(p);
    }
    function printedSoFar(upto) {
      for (let i = 0; i <= upto; i++) if (steps()[i].stdout) return true;
      return false;
    }
    function onStep(i) {
      const s = steps()[i];
      if (!s) return;
      JT.clearLines(pre, 'now'); JT.clearLines(pre, 'err');
      JT.markLines(pre, s.lines, s.error ? 'err' : 'now');
      note.innerHTML = s.note; note.classList.toggle('err', !!s.error);
      renderJaxpr(i);
      tracers.textContent = ex.tracers;
      stdout.innerHTML = printedSoFar(i) ? '<span class="k">tracing:</span> JitTracer(float32[3])' : '(nothing printed)';
    }
    function onReset() {
      JT.clearLines(pre, 'now'); JT.clearLines(pre, 'err');
      note.innerHTML = 'Press Play or Step to trace <code>f</code> with abstract inputs of shape (3,).'; note.classList.remove('err');
      renderJaxpr(-1); tracers.textContent = ''; stdout.textContent = '(nothing printed)';
    }

    const sigFor = (n) => '(' + Array(ex.arity).fill(`f32[${n}]`).join(', ') + ')';
    const callLabel = (n) => 'f(' + Array(ex.arity).fill(`ones(${n})`).join(', ') + ')';
    function renderCache() {
      const sigs = Array.from(cache);
      cacheLine.innerHTML = 'Cached executables: ' + (sigs.length ? sigs.map((s) => `<code>${JT.escape(s)}</code>`).join(', ') : 'none');
    }
    function call(n) {
      const sig = sigFor(n), hit = cache.has(sig), r = ex.call(n, hit, fixed);
      let html = `<code>${JT.escape(callLabel(n))}</code>, signature <code>${JT.escape(sig)}</code>: `;
      if (hit) html += '<span class="hit">cache hit</span>. The cached executable runs; no Python runs.';
      else if (r.error) html += `<span class="miss">cache miss</span>, trace fails with <span class="bad">${JT.escape(r.error)}</span>. Nothing is cached.`;
      else { html += '<span class="miss">cache miss</span>. JAX traces, compiles and runs.'; cache.add(sig); }
      r.effects.forEach((e) => { html += `<span class="out">printed: ${JT.escape(e)}</span>`; });
      const empty = JT.$('.empty', log); if (empty) empty.remove();
      calls++;
      log.appendChild(JT.el('li', {}, [JT.el('span', { class: 'n', text: String(calls) }), JT.el('span', { html })]));
      renderCache();
    }
    function clearCalls() {
      cache = new Set(); calls = 0;
      log.replaceChildren(JT.el('li', { class: 'empty', text: 'No calls yet.' }));
      renderCache();
    }

    function load() {
      if (stepper) stepper.destroy();
      pre = JT.code(fixed && ex.fixedSrc ? ex.fixedSrc : ex.src);
      codeHost.replaceChildren(pre);
      fixCtl.hidden = !ex.fixedSrc;
      fixSeg.setValue(fixed ? 'cond' : 'if');
      stepper = JT.stepper({ total: steps().length, interval: 1500, onStep, onReset, noun: 'step' });
      stepHost.replaceChildren(stepper.el);
      stepper.goto(steps().length - 1); // rest on the finished trace
      callBtns.replaceChildren(
        JT.button(callLabel(3), () => call(3), { class: 'btn mono' }),
        JT.button(callLabel(4), () => call(4), { class: 'btn mono' }),
        JT.button('Clear log', clearCalls),
      );
      clearCalls();
      const errs = ex.call(3, false, fixed).error;
      (errs ? [3] : [3, 3, 4]).forEach(call); // a resting log that already shows miss, hit, miss
      eager.innerHTML = '<b>Eager PyTorch</b> ' + ex.eager;
    }
    load();
  });
})();
