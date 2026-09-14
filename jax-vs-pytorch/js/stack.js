/* Layered stack diagram: framework → IR → compiler → kernels → runtime → hardware,
   JAX and PyTorch side by side. Hover or focus a cell to read about it. */
(function () {
  'use strict';
  const JT = window.JT;

  JT.style('stack', `
    #w-stack .grid { display: grid; grid-template-columns: 9rem 1fr 1fr; gap: 6px; align-items: stretch; }
    #w-stack .rowlab { display: flex; flex-direction: column; justify-content: center; font-size: 0.78rem; color: var(--ink-3); padding-right: 0.5rem; }
    #w-stack .rowlab b { color: var(--ink-2); font-weight: 600; font-size: 0.8125rem; }
    #w-stack .colhead { font-size: 0.8125rem; font-weight: 600; padding: 0.2rem 0.6rem 0.4rem; }
    #w-stack .colhead.jax { color: var(--jax-deep); }
    #w-stack .colhead.torch { color: var(--torch-deep); }
    #w-stack .cell {
      appearance: none; font: inherit; text-align: left; cursor: pointer; width: 100%;
      border: 1px solid var(--line); border-radius: 10px; background: var(--surface); padding: 0.6rem 0.75rem;
      display: flex; flex-direction: column; gap: 0.15rem; min-height: 3.6rem;
      transition: background 140ms, border-color 140ms, transform 200ms var(--ease-out);
    }
    #w-stack .cell .name { font-weight: 600; font-size: 0.875rem; color: var(--ink); }
    #w-stack .cell .sub { font-size: 0.75rem; color: var(--ink-3); line-height: 1.35; }
    #w-stack .cell.jax { border-color: var(--jax-soft); background: color-mix(in oklab, var(--jax-soft) 45%, var(--surface)); }
    #w-stack .cell.torch { border-color: var(--torch-soft); background: color-mix(in oklab, var(--torch-soft) 45%, var(--surface)); }
    #w-stack .cell.jax:hover, #w-stack .cell.jax.active { border-color: var(--jax); background: var(--jax-soft); }
    #w-stack .cell.torch:hover, #w-stack .cell.torch.active { border-color: var(--torch); background: var(--torch-soft); }
    #w-stack .cell.active { transform: translateX(2px); }
    #w-stack .cell .x { display: inline-flex; align-items: center; gap: 0.3rem; margin-top: 0.25rem; font-size: 0.7rem; color: var(--ink-2); border: 1px dashed var(--line-2); border-radius: 999px; padding: 0.1rem 0.5rem; width: fit-content; background: var(--surface); }
    #w-stack .cell .x svg { width: 12px; height: 12px; }
    #w-stack .detail { margin-top: 1rem; border-top: 1px solid var(--line); padding-top: 1rem; display: grid; grid-template-columns: 1fr minmax(0, 22rem); gap: 1.25rem; min-height: 8.5rem; }
    #w-stack .detail h5 { margin: 0 0 0.35rem; font-size: 0.95rem; font-weight: 600; }
    #w-stack .detail h5 .who { font-weight: 500; font-size: 0.75rem; padding: 0.1rem 0.5rem; border-radius: 999px; margin-left: 0.5rem; vertical-align: middle; }
    #w-stack .detail h5 .who.jax { background: var(--jax-soft); color: var(--jax-deep); }
    #w-stack .detail h5 .who.torch { background: var(--torch-soft); color: var(--torch-deep); }
    #w-stack .detail p { margin: 0; font-size: 0.9rem; color: var(--ink-2); line-height: 1.55; }
    #w-stack .detail pre.code { border: 1px solid var(--line); border-radius: var(--radius-sm); font-size: 0.75rem; }
    @media (max-width: 760px) {
      #w-stack .grid { grid-template-columns: 1fr 1fr; }
      #w-stack .rowlab { grid-column: 1 / -1; flex-direction: row; gap: 0.5rem; align-items: baseline; padding: 0.5rem 0 0; }
      #w-stack .detail { grid-template-columns: 1fr; }
    }
  `);

  const ROWS = [
    {
      label: 'You write', sub: 'the API',
      jax: { name: 'jax.numpy + transformations', sub: 'jit · grad · vmap · shard_map', body: 'JAX provides a NumPy-shaped array API and transformations over Python functions. Operations can run without an outer jit; transformations produce compiled, differentiated, vectorized, or explicitly sharded functions.', code: 'y = jax.jit(jax.grad(loss))(params, batch)' },
      torch: { name: 'torch tensors + nn.Module', sub: 'eager ops · autograd · optimizers', body: 'Tensor operations run eagerly by default. Modules store parameters, autograd records differentiable operations, and optimizers update parameters. torch.compile can compile compatible regions.', code: 'loss = model(batch).mean(); loss.backward(); opt.step()' },
    },
    {
      label: 'Intermediate form', sub: 'what gets optimized',
      jax: { name: 'jaxpr → StableHLO', sub: 'created on a trace', body: 'A JAX transformation can trace a function into jaxpr, a typed functional intermediate representation. For jit compilation, JAX lowers that computation through StableHLO. A compatible cache hit reuses the compiled executable without retracing.', code: 'print(jax.make_jaxpr(f)(x))\n# { lambda ; a:f32[3]. let b:f32[3] = sin a in (b,) }' },
      torch: { name: 'eager operations · FX graph (compile)', sub: 'TorchDynamo captures bytecode', body: 'Eager mode dispatches tensor operations directly. torch.compile uses TorchDynamo to capture FX graph regions and attaches guards to the assumptions used during capture. A failed guard can select another cached graph or trigger recompilation; unsupported code can create a graph break.', code: 'compiled = torch.compile(model)   # capture begins on first call' },
    },
    {
      label: 'Compiler', sub: 'fusion, layout, memory',
      jax: { name: 'XLA', sub: 'optimizes a jitted computation', body: 'XLA optimizes the computation captured by jit. It can fuse operations, assign layouts, plan buffers, and schedule work for the target. GPU output includes generated device code and calls to optimized libraries.', code: 'jax.jit(f).lower(x).compile()   # compile explicitly' },
      torch: { name: 'TorchInductor (default compile backend)', sub: 'generates GPU and CPU code', body: 'Under torch.compile, AOTAutograd captures backward graphs where needed and TorchInductor lowers captured regions. It can generate fused GPU kernels and optimized CPU code. Graph breaks divide a function into separate compiled regions.', code: 'torch.compile(f, mode="max-autotune")' },
    },
    {
      label: 'Kernels', sub: 'the code that runs',
      jax: { name: 'XLA fusions + libraries', sub: 'Pallas for custom kernels', body: 'XLA generates kernels and calls optimized libraries for operations such as matrix multiplication. Pallas exposes lower-level custom kernels; current JAX uses Mosaic on TPU and Mosaic GPU on supported NVIDIA GPUs, with a Triton GPU backend still available.', code: 'pl.pallas_call(kernel, out_shape=...)(x)' },
      torch: { name: 'ATen operators', sub: 'vendor libraries · native kernels · Triton', body: 'ATen operators select implementations for each backend. Those implementations include vendor libraries, generated code, and native CPU or accelerator kernels. Extensions can use C++, CUDA, or Triton.', code: 'torch.utils.cpp_extension.load(name="my_op", sources=[...])' },
    },
    {
      label: 'Runtime', sub: 'devices and streams',
      jax: { name: 'PJRT', sub: 'async dispatch, device plugins', body: 'PJRT connects the framework, compiler, and device runtime. JAX can dispatch work asynchronously and return a jax.Array before device work finishes. Host materialization or block_until_ready() waits for completion.', code: 'y = f(x); y.block_until_ready()' },
      torch: { name: 'c10 dispatcher + CUDA runtime', sub: 'streams, caching allocator, NCCL', body: 'The dispatcher routes each op to a kernel by device and dtype; CUDA streams run them asynchronously; a caching allocator avoids cudaMalloc costs; NCCL handles multi-GPU collectives.', code: 'torch.cuda.synchronize()' },
    },
    {
      label: 'Hardware', sub: 'where it finally runs',
      jax: { name: 'TPU', sub: 'also GPU and CPU through XLA', body: 'TPU TensorCores combine systolic matrix units with vector and scalar units and an explicit memory hierarchy. XLA schedules compiled operations and data movement for that architecture.', code: 'jax.devices()  # [TpuDevice(id=0, ...), ...]', cross: 'NVIDIA / AMD GPU via XLA:GPU' },
      torch: { name: 'NVIDIA GPU', sub: 'also CPU, AMD ROCm, Apple MPS', body: 'GPUs schedule thread blocks onto streaming multiprocessors at run time. Caches, shared memory, and many resident warps help hide memory latency. Both eager and compiled PyTorch programs use this hardware scheduling.', code: 'torch.device("cuda")', cross: 'TPU via PyTorch/XLA' },
    },
  ];

  JT.widget('stack', (container) => {
    const S = JT.stage(container, { title: 'Execution layers', hint: 'Hover or focus a layer to compare' });
    const grid = JT.el('div', { class: 'grid' });
    grid.append(JT.el('div'), JT.el('div', { class: 'colhead jax', text: 'JAX · XLA · TPU' }), JT.el('div', { class: 'colhead torch', text: 'PyTorch · CUDA · GPU' }));

    const detailTitle = JT.el('h5');
    const detailBody = JT.el('p');
    const detailCode = JT.el('div');
    const detail = JT.el('div', { class: 'detail', 'aria-live': 'polite' }, [JT.el('div', {}, [detailTitle, detailBody]), detailCode]);

    let active = null;
    function show(cell, row, side) {
      if (active) active.classList.remove('active');
      active = cell; cell.classList.add('active');
      const d = row[side];
      detailTitle.innerHTML = `${JT.escape(d.name)} <span class="who ${side}">${side === 'jax' ? 'JAX' : 'PyTorch'} · ${JT.escape(row.label.toLowerCase())}</span>`;
      detailBody.textContent = d.body;
      detailCode.replaceChildren(JT.code(d.code));
    }
    function intro() {
      if (active) active.classList.remove('active'); active = null;
      detailTitle.textContent = 'Compare the execution paths';
      detailBody.textContent = 'Both stacks dispatch work to an accelerator. JAX transformations stage computations for compilation; eager PyTorch dispatches tensor operations directly, and torch.compile adds graph capture and compilation. Read down each column, then compare the same layer across.';
      detailCode.replaceChildren(JT.code('jax:     python → trace → compile → run\ntorch:   python → dispatch → run (→ compile, if you ask)', { lang: 'plain' }));
    }

    const crossIcon = '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" aria-hidden="true"><path d="M2 12 L14 4"/><path d="M10 4h4v4"/></svg>';
    ROWS.forEach((row) => {
      grid.appendChild(JT.el('div', { class: 'rowlab' }, [JT.el('b', { text: row.label }), JT.el('span', { text: row.sub })]));
      ['jax', 'torch'].forEach((side) => {
        const d = row[side];
        const cell = JT.el('button', { class: 'cell ' + side, type: 'button' }, [
          JT.el('span', { class: 'name', text: d.name }),
          JT.el('span', { class: 'sub', text: d.sub }),
          d.cross ? JT.el('span', { class: 'x', html: crossIcon + '<span>' + JT.escape(d.cross) + '</span>' }) : null,
        ]);
        cell.addEventListener('pointerenter', () => show(cell, row, side));
        cell.addEventListener('focus', () => show(cell, row, side));
        cell.addEventListener('click', () => show(cell, row, side));
        grid.appendChild(cell);
      });
    });
    grid.addEventListener('pointerleave', () => { /* keep last selection visible */ });

    S.body.append(grid, detail);
    S.foot.innerHTML = '<span>Dashed badges show supported paths to hardware more closely associated with the other stack.</span>';
    intro();
  });
})();
