# Quantization-Aware Training

An interactive explainer on how quantization-aware training (QAT) simulates a
target quantizer during training and reduces the resulting task error.
Live at https://sghael.github.io/visual-learn/qat-explainer/.

## Open it

Open `index.html` from disk, or serve the folder:

```bash
cd qat-explainer
pnpm serve      # http://0.0.0.0:8000/
```

The page loads D3 7.9.0 from cdnjs and three Google Fonts with system-font
fallbacks. Everything else, including the training lab, is inline.

## What each widget does

| Section | Widget | Interaction |
|---|---|---|
| 1 Weight memory | Weight-memory bar chart | Pick a model size; bars for bf16, int8, int4 against GPU memory lines |
| 2 Quantizing one number | Number line | Drag x or use its slider; sliders for bit-width and clip range; readout of q, x̂, error |
| 3 Quantizing a matrix | Three heatmaps | Bit-width slider, per-tensor vs per-row scales, outlier toggle |
| 4 Rounding in the loop | QAT loop diagram | Six step buttons highlight each stage and explain it |
| 5 Surrogate gradients | Forward/backward plots, drift animation | Toggle true derivative vs STE; play a master weight crossing a rounding boundary |
| 6 Training lab | Live training in the browser | Choose 2, 3 or 4 bits and a seed; compare fp32 pretraining, PTQ, and QAT |
| 7 Transformer tensors | Transformer block map | Hover, tap, or tab to inspect the precision assigned to each tensor in the example |
| 8 Published recipes | Table | LLM-QAT, BitNet b1.58, EfficientQAT, Gemma 3 QAT, ParetoQ |
| 9 Review questions | Three reveal questions | Open each answer |

## Simplifications and caveats

- The training lab is a 97-parameter regression network (1 → 32 tanh → 1),
  not a language model. It quantizes weights only, per tensor, with a scale
  recomputed from the current master weights every step (so no weight is
  ever clipped), a straight-through gradient, Adam, and a cosine
  learning-rate decay. Under `prefers-reduced-motion` the lab computes the
  whole run at once and paints only the final state. There
  is no activation quantization and no distillation. The page says so next
  to the figure.
- With these settings QAT beats PTQ on every seed at 3 and 4 bits and on
  most seeds at 2 bits. Two-bit results vary by seed on purpose; that is
  part of the lesson.
- Memory figures are weight-only and ignore group scales, activations and
  the KV cache. GPU sizes are nominal.
- The transformer block shows one illustrative W4A8 layout. Real recipes differ in
  which tensors they quantize and at what granularity.
- Every row of the recipe table links its primary source (arXiv paper or
  the official Gemma 3 QAT announcement); the figures quoted are the ones
  those sources state.

## Tests

```bash
pnpm install
pnpm test       # Playwright against the installed Chrome, or bundled Chromium in CI
```

The tests open the page at 1440 px and 390 px and fail on any console or
page error or on horizontal overflow, then exercise the quantizer, the
number line, the heatmap outlier toggle, the loop stepper, the drift reset,
and a full training-lab run including a bit-width toggle mid-run.
