# JAX Explainer

An interactive, visual tutorial on how JAX, XLA and TPUs run your code, side by
side with PyTorch and CUDA. Built in the spirit of the
[Polo Club](https://poloclub.github.io/) explainers.

## Open it

It is a standalone page. Either double-click `index.html`, or serve the folder:

```bash
python3 -m http.server 8000 --bind 0.0.0.0
```

No build step, no package manager, no tracking. Fonts load from Google Fonts
when online and fall back to system faces offline.

## What is inside

| Section | Widget | What it shows |
|---|---|---|
| Hero | `js/pipeline.js` | One call travelling both routes: trace → compile → cache versus dispatch every op |
| Stacks | `js/stack.js` | Framework, IR, compiler, kernels, runtime and hardware layers, hover to read |
| Tracing | `js/tracing.js` | Step through what `jax.jit` records; call with different shapes; see the cache, side effects and the control-flow rule |
| Fusion | `js/fusion.js` | Kernel launches and HBM traffic for a chain of ops, eager versus compiled |
| Autodiff | `js/autodiff.js` | PyTorch's tape versus JAX's `grad` as a transformation, with a shared slider |
| Transformations | `js/transforms.js` | `vmap` as rewriting rather than looping, and a composer for `jit`, `grad`, `vmap` |
| State | `js/purity.js` | The same training step in both frameworks with concept highlighting; splittable keys versus a global RNG |
| Silicon | `js/silicon.js` | A 4 × 4 systolic array next to GPU thread blocks scheduled onto SMs |
| Sharding | `js/sharding.js` | A 2 × 4 device mesh, PartitionSpecs, and the collectives the partitioner inserts |
| Choosing | static | Comparison table and a short decision guide |

`js/common.js` is the shared toolkit (element builders, syntax highlighting,
stepper, segmented control, tooltips, widget registry). `styles.css` holds the
design tokens and shared chrome; each widget injects its own scoped CSS.

## Tests

The page has no runtime dependencies. The tests do: Playwright drives the
page in headless Chrome (the installed Google Chrome, or Playwright's own
Chromium as a fallback).

```bash
pnpm install
pnpm test
```

`test/models.test.mjs` checks the pure arithmetic behind the fusion, sharding
and systolic-array widgets in node. `test/page.test.mjs` renders the page at
desktop and phone widths and drives the interactions that have bitten before:
switching tracing examples during autoplay, resetting the hero mid-run, the
call-button arity per example, and every composer preset.

## Design notes

- Entity colors are fixed: JAX is violet, PyTorch is orange, and "currently
  active" is green. The palette was validated for colorblind safety.
- Wide diagrams scroll inside their own container on narrow screens rather than
  widening the page.
- Animations respect `prefers-reduced-motion`.

## Accuracy

Framework behavior reflects JAX 0.6 and PyTorch 2.7. Hardware figures are
rounded vendor peaks and marked approximate. The fusion and sharding rule
engines are deliberate simplifications of what XLA and the SPMD partitioner do;
the page says so where it applies.
