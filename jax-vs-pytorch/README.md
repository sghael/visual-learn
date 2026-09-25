# How JAX and PyTorch run your code

A single-page explainer on what happens between a Python function call and the
accelerator in JAX and in PyTorch. It follows one function down both default
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
| 9 | Random draws | static tables | Adding one draw shifts every later PyTorch draw; JAX draws keyed separately stay the same |
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
call-button arity per example, every composer preset, fusion readouts against
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
- The fusion figure is a simplified traffic model: one read per HBM input and
  one write per kernel output, vectors counted as zero, softmax counted as one
  read and one write, memory time as a lower bound at peak bandwidth. Real
  fusion depends on backend, shapes and compiler version.
- The systolic array is a 4 × 4 weight-stationary model with a self-check that
  it reproduces A·B in 10 cycles. The GPU drawing keeps one block per SM, which
  real SMs do not.
- The sharding figure applies three fixed rules to one matmul. The real
  partitioner (GSPMD or Shardy) propagates layouts through a whole program and
  chooses collectives with cost models.
- The random-draw figure uses stream positions and key names rather than
  numbers. That JAX 0.11's `split(key, 4)` keeps the first three keys of
  `split(key, 3)` was checked by running both with the default threefry
  implementation.
