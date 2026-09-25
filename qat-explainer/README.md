# Quantization-aware training

An explainer on how quantization-aware training (QAT) works: rounding weights
onto an integer grid, the rounding and clipping error that causes, training
through the rounding with a straight-through estimator, and how QAT compares
with post-training quantization (PTQ). Live at
https://sghael.github.io/visual-learn/qat-explainer/.

## Open it

Open `index.html` from disk, or serve the folder:

```bash
cd qat-explainer
pnpm serve      # python3 -m http.server on 0.0.0.0:8000
```

The page has no dependencies. `index.html` holds the prose, `styles.css` the
Visual Learn house style plus page rules, and `qat.js` every chart, drawn as
SVG at the pixel width of its container. Fonts come from Google Fonts with
system fallbacks.

## Figures

| # | Figure | What it shows | Interaction |
|---|---|---|---|
| 1 | Weight memory | Dot plot of bf16, int8 and int4 weight storage for 1B to 405B models on a log scale, against 16, 24 and 80 GB GPUs | None: all five sizes at once |
| 2 | The grid | (a) number line with the levels, the clipping range, a value *x* and its rounded value; (b) the error at every *x*; (c) mean squared error over normally distributed weights as α varies, split into rounding and clipping parts | Sliders for bits, α and *x*; drag the dot |
| 3 | One scale for many weights | Error maps for per-tensor scaling without and with an outlier, and per-row scaling with the outlier, on one color scale | None: three small multiples |
| 4 | One QAT step | Forward pass down the left, backward pass up the right, for one linear layer | None |
| 5 | Gradients through a staircase | The quantizer, its true derivative and the straight-through estimate, on one *x*-axis | None: three small multiples |
| 6 | Master weight trace | A simulated master weight drifting across rounding boundaries, with the value the forward pass sees | None |
| 7 | Training lab | Trains a 97-parameter network in fp32, then compares PTQ and QAT at 2, 3 and 4 bits: fitted functions and QAT loss curves, directly labeled | Seed buttons retrain in the browser (about 0.2 s) |
| 8 | All seeds | PTQ loss ÷ QAT loss for all eight seeds at each bit width | Highlights the seed chosen in Figure 7 |

Tables cover which tensors a transformer quantizes (with parameter shares
computed for Llama 3 8B) and five published QAT recipes.

## Simplifications and caveats

- The training lab is a 1 → 32 tanh → 1 regression network, not a language
  model. It quantizes weights only, per tensor, with the scale recomputed
  from the current master weights every step (so no weight is ever clipped),
  a straight-through gradient, full-batch Adam and a cosine learning-rate
  decay. There is no activation quantization and no distillation. The page
  says "simulated and simplified" in the figure caption.
- The lab is trained on page load for seed 2 and on each seed click. Nothing
  animates. Figure 8's ratios for all eight seeds are precomputed with the
  same code (running all eight takes over a second); a test recomputes them.
- With these settings QAT beats PTQ on every seed at 3 and 4 bits and on
  seven of eight seeds at 2 bits. Seed 3 at 2 bits ends worse than PTQ; the
  page shows it.
- Figure 2's weights are normal with σ = 0.35. The best clipping range it
  reports (1.2σ at 2 bits, 2.5σ at 4 bits, 3.9σ at 8 bits) is for that
  distribution and this symmetric quantizer, which uses 2^b − 1 levels.
- Figure 3's matrix is simulated (normal, σ = 0.3, one outlier of 2.6) at
  4 bits.
- Figure 6 is a simulated update sequence (mean −0.03 per step, standard
  deviation 0.06, fixed seed), with s = 1.
- Memory figures are weight-only and ignore group scales, activations and
  the KV cache. GPU sizes are nominal.
- The transformer table describes one illustrative W4A8 layout. Real
  recipes differ in which tensors they quantize and how finely. Parameter
  shares are computed from the Llama 3 paper (Table 3) and the released
  model configuration (128,256-token vocabulary, untied embeddings).
- Every row of the recipe table links its primary source; the figures quoted
  are the ones those sources state.

## Tests

```bash
pnpm install --frozen-lockfile
pnpm test       # Playwright against the installed Chrome, or bundled Chromium in CI
```

The tests open the page at 1440 px and 390 px and fail on any console or
page error, on horizontal overflow, on chart text smaller than 11 px, or on
chart text that runs outside its SVG. They also check the quantizer, the
number line (drag and keyboard, rounding versus clipping), the best-α label
against the computed minimum, the heatmap numbers, the step diagram's layout
on a phone, the straight-through gradient, the drift trace's annotations,
every direct label in the training lab against the computed losses, rapid
seed changes, the precomputed all-seed ratios, and the top bar's section
tracking.
