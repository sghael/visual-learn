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
      jax: { name: 'jax.numpy + transformations', sub: 'jit · grad · vmap · shard_map', body: 'A NumPy-shaped array library plus a small set of function transformations. You write plain Python functions over immutable arrays; the transformations turn them into compiled, differentiated or vectorized versions.', code: 'y = jax.jit(jax.grad(loss))(params, batch)' },
      torch: { name: 'torch tensors + nn.Module', sub: 'eager ops · autograd · optimizers', body: 'Tensors that execute as soon as you touch them, modules that own their parameters, and autograd tracking every operation behind the scenes. Compilation is an opt-in wrapper.', code: 'loss = model(batch).mean(); loss.backward(); opt.step()' },
    },
    {
      label: 'Intermediate form', sub: 'what gets optimized',
      jax: { name: 'jaxpr → StableHLO', sub: 'always, for every jitted call', body: 'Tracing produces a jaxpr: a typed, functional program with one equation per primitive. It is lowered to StableHLO, an MLIR dialect, which is the contract between JAX and the compiler.', code: 'print(jax.make_jaxpr(f)(x))\n# { lambda ; a:f32[3]. let b:f32[3] = sin a in (b,) }' },
      torch: { name: 'none (eager) · FX graph (compile)', sub: 'TorchDynamo captures bytecode', body: 'Eager mode has no program representation: the op stream is the program. torch.compile intercepts Python bytecode with TorchDynamo, builds an FX graph, and adds guards so it can fall back to eager if an assumption breaks.', code: 'compiled = torch.compile(model)   # graph captured on first call' },
    },
    {
      label: 'Compiler', sub: 'fusion, layout, memory',
      jax: { name: 'XLA', sub: 'whole-program, ahead of time', body: 'XLA sees the entire function at once: it fuses elementwise ops into their neighbours, chooses memory layouts, plans buffers, and schedules everything for the target. On TPU it is the only way to run; on GPU it emits PTX plus library calls.', code: 'jax.jit(f).lower(x).compile()   # explicit AOT if you want it' },
      torch: { name: 'TorchInductor (opt-in)', sub: 'generates Triton / C++ kernels', body: 'Under torch.compile, Inductor lowers the FX graph into fused Triton kernels for GPU or C++ for CPU. Eager mode skips this layer entirely and pays per-op launch costs instead.', code: 'torch.compile(f, mode="max-autotune")' },
    },
    {
      label: 'Kernels', sub: 'the code that runs',
      jax: { name: 'XLA fusions + libraries', sub: 'Pallas for custom kernels', body: 'Most kernels are generated by XLA. Matmuls go to the MXU on TPU or cuBLAS/cuDNN on GPU. For hand-tuned kernels, Pallas is a Python DSL that lowers to Mosaic on TPU and Triton on GPU.', code: 'pl.pallas_call(kernel, out_shape=...)(x)' },
      torch: { name: 'ATen kernels', sub: 'cuBLAS · cuDNN · hand-written CUDA · Triton', body: 'Thousands of pre-compiled kernels ship with PyTorch: vendor libraries for matmul and convolution, hand-written CUDA for everything else. Custom kernels are CUDA C++ extensions or Triton.', code: 'torch.utils.cpp_extension.load(name="my_op", sources=[...])' },
    },
    {
      label: 'Runtime', sub: 'devices and streams',
      jax: { name: 'PJRT', sub: 'async dispatch, device plugins', body: 'PJRT is the plugin interface between JAX and any backend. Calls return immediately with a future-like array; the device works asynchronously and you block only when you read a value.', code: 'y = f(x); y.block_until_ready()' },
      torch: { name: 'c10 dispatcher + CUDA runtime', sub: 'streams, caching allocator, NCCL', body: 'The dispatcher routes each op to a kernel by device and dtype; CUDA streams run them asynchronously; a caching allocator avoids cudaMalloc costs; NCCL handles multi-GPU collectives.', code: 'torch.cuda.synchronize()' },
    },
    {
      label: 'Hardware', sub: 'where it finally runs',
      jax: { name: 'TPU', sub: 'also GPU and CPU through XLA', body: 'The native pairing. TPUs are compiler-scheduled machines with systolic matrix units and no hardware caches, so a whole-program compiler that knows every shape in advance is exactly what they need.', code: 'jax.devices()  # [TpuDevice(id=0, ...), ...]', cross: 'NVIDIA / AMD GPU via XLA:GPU' },
      torch: { name: 'NVIDIA GPU', sub: 'also CPU, AMD ROCm, Apple MPS', body: 'The native pairing. GPUs schedule thousands of threads in hardware and hide latency with caches, which is what lets an eager op-by-op stream run well without a compiler in the loop.', code: 'torch.device("cuda")', cross: 'TPU via PyTorch/XLA' },
    },
  ];

  JT.widget('stack', (container) => {
    const S = JT.stage(container, { title: 'The two stacks, layer by layer', hint: 'Hover or focus any layer' });
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
      detailTitle.textContent = 'Same shape, different timing';
      detailBody.textContent = 'Both stacks turn Python into kernels on an accelerator. JAX commits to a compiler at every call; PyTorch commits to a dispatcher and lets you add the compiler later. Read down each column, then compare across.';
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
    S.foot.innerHTML = '<span>Dashed badges mark the crossings: each framework can reach the other\'s hardware, with a translation layer in between.</span>';
    intro();
  });
})();
