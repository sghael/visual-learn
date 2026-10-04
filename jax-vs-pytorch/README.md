# How JAX and PyTorch run your code

A single-page explainer on what happens between a Python function call and the
accelerator in JAX and in PyTorch. It follows one function down two common
routes (`jax.jit` traces and compiles once; eager PyTorch dispatches every
operation on every call), then shows what that difference means for fusion,
gradients, batching, state and randomness, TPU and GPU scheduling, and
sharding across devices. Ten sections, about 25 minutes.

Live at <https://sghael.github.io/visual-learn/jax-vs-pytorch/>.

## Open it

It is a standalone page with no build step. Open `index.html` directly, or
serve the folder:

```bash
python3 -m http.server 8000 --bind 0.0.0.0
```

Fonts (Source Serif 4, Source Sans 3, IBM Plex Mono) load from Google Fonts
when online and fall back to system faces offline. There is no tracking.

## Figures

| # | Figure | Kind | What it shows |
|---|---|---|---|
| 1 | What each call does | static table | JAX's first call, JAX's second call and an eager PyTorch call, step by step |
| 2 | The two stacks | static table | API, intermediate form, compiler, kernels, runtime and hardware in each framework |
| 3 | `js/tracing.js` | stepper + call log | What `jit` records line by line for three functions (pure math, a `print`, a Python `if` and its `lax.cond` fix), and which calls hit the cache |
| 4 | `js/fusion.js` | small multiples | The same chain of operations eager and compiled, on one horizontal scale; toggles remove operations |
| 5 | `js/autodiff.js` | one stepper, one slider | PyTorch's autograd tape and JAX's gradient jaxpr advancing through the same graph; `w` changes the numbers but not the jaxpr |
| 6 | `js/transforms.js` (`vmap`) | small multiples | A Python loop next to `jax.vmap`, with the batch axis of X as the only control |
| 7 | `js/transforms.js` (`composer`) | builder | Stacks of `jit`, `grad`, `vmap` and `jacfwd`, what they return, and the `torch.func` equivalent |
| 8 | `js/purity.js` | concept picker | The same training step in JAX + Optax and in PyTorch, with the lines for one concept highlighted |
| 9 | Random draws | static tables | An extra draw advances a shared PyTorch generator; JAX draws keep their values when their assigned keys stay fixed |
| 10 | `js/silicon.js` (`systolic`) | stepper | A 4 × 4 weight-stationary systolic array computing C = A·B cycle by cycle |
| 11 | `js/silicon.js` (`gpu`) | static drawing | The same output as 16 thread blocks on 8 SMs in two waves |
| 12 | `js/sharding.js` | presets + selects | `y = x @ w` on a 2 × 4 mesh: which devices hold each block and which collectives the compiler inserts |

Nothing animates on load or on scroll. Every stepper rests on a meaningful
frame (the finished trace, the finished gradient, cycle 4 of the systolic
array) and moves only when the reader presses Play or Step.

`js/common.js` is the page's small toolkit (element builders, syntax
highlighting, the stepper, segmented controls, a width observer, the widget
registry). `styles.css` starts with a copy of the Visual Learn house style and
adds the page's own rules after it. Entity colors are fixed: JAX violet,
PyTorch orange, "active right now" green.

## Tests

```bash
pnpm install --frozen-lockfile
pnpm test
```

`test/models.test.mjs` checks the arithmetic behind the fusion, sharding and
systolic-array figures in node. `test/page.test.mjs` drives the page in
headless Chrome at 1440 px and 390 px (no console errors, no horizontal
overflow, no chart text under 11 px) and covers the interactions that have
broken before or could break: nothing moving on load or scroll, switching
tracing examples during playback, a stepper reset mid-run under a fake clock,
call-button arity per example, every composer preset and repeated differentiation through order four, fusion readouts against
the model, the autodiff result against the analytic gradient, the `vmap` axis
control, concept highlighting, sharding presets, and the top-bar section
marker.

## Accuracy

- Framework behavior was checked against the JAX 0.11 and PyTorch 2.13
  documentation. The jaxprs and the tracer text (`JitTracer(float32[3])`) were
  printed with JAX 0.11.2; the page removes type annotations on literals and
  some primitive parameters, and says so in the captions.
- Hardware numbers (H100 SXM: 989 TFLOP/s dense bf16, 495 TFLOP/s dense TF32,
  3.35 TB/s HBM, 132 SMs) are vendor peaks, rounded and approximate.
- Compilation is opt-in in both frameworks: the main comparison uses `jax.jit`
  and eager PyTorch. Undecorated JAX dispatches operations individually, as the
  [JAX JIT guide](https://docs.jax.dev/en/latest/jit-compilation.html) explains.
- The fusion figure is a simplified traffic model: one read per HBM input and
  one write per kernel output, vectors counted as zero, softmax counted as one
  read and one write, memory time as a lower bound at peak bandwidth. Real
  fusion depends on backend, shapes and compiler version. A 4096² float32 tensor
  occupies 64 MiB (67,108,864 bytes); all traffic labels use binary MiB while
  peak bandwidth is decimal TB/s. Cache reuse and multi-kernel operators are
  omitted.
- The systolic array is a 4 × 4 weight-stationary model with a self-check that
  it reproduces A·B in 10 cycles. The GPU drawing keeps one block per SM, which
  real SMs do not.
- The sharding figure applies three fixed rules to one matmul. The real
  partitioner (GSPMD or Shardy) propagates layouts through a whole program and
  chooses collectives with cost models.
- The composer models a scalar loss with `params = {w: f32[D], b: f32[]}`.
  Derivatives append the input pytree to each output leaf; each `vmap` adds a
  leading batch axis and explicitly shares parameters with `in_axes=(None, 0, 0)`.
  Repeated batching requires correspondingly higher-rank inputs.
- The random-draw figure counts calls to a shared PyTorch generator and holds
  JAX key assignments fixed. It does not assume a particular number of internal
  random bits per call or prefix stability when changing `split` counts. Stable
  IDs passed to `fold_in` illustrate explicit key assignment; a separate
  `torch.Generator` can also isolate debugging draws.
