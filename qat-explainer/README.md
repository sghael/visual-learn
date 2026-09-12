# Quantization-Aware Training

An interactive explainer on how quantization-aware training (QAT) lets large
language models run at 4 bits and below without losing what they learned.
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
| 1 Why shrink a model | Weight-memory bar chart | Pick a model size; bars for bf16, int8, int4 against GPU memory lines |
| 2 Quantizing one number | Number line | Drag x; sliders for bit-width and clip range; readout of q, x̂, error |
| 3 Quantizing a matrix | Three heatmaps | Bit-width slider, per-tensor vs per-row scales, outlier toggle |
| 4 Rounding in the loop | QAT loop diagram | Six step buttons highlight each stage and explain it |
| 5 The straight-through trick | Forward/backward plots, drift animation | Toggle true derivative vs STE; play a latent weight crossing a rounding boundary |
| 6 Training lab | Live training in the browser | Choose 2, 3 or 4 bits and a seed; watch fp32 pretraining, then PTQ vs QAT |
| 7 Inside a real LLM | Transformer block map | Hover or tap each block for what precision it runs at and why |
| 8 Recipes in the wild | Table | LLM-QAT, BitNet b1.58, EfficientQAT, Gemma 3 QAT, ParetoQ |
| 9 Check yourself | Three reveal questions | |

## Simplifications and caveats

- The training lab is a 97-parameter regression network (1 → 32 tanh → 1),
  not a language model. It quantizes weights only, per tensor, with a scale
  recomputed from the current master weights every step, a clipped
  straight-through gradient, Adam, and a cosine learning-rate decay. There
  is no activation quantization and no distillation. The page says so next
  to the figure.
- With these settings QAT beats PTQ on every seed at 3 and 4 bits and on
  most seeds at 2 bits. Two-bit results vary by seed on purpose; that is
  part of the lesson.
- Memory figures are weight-only and ignore group scales, activations and
  the KV cache. GPU sizes are nominal.
- The transformer block shows one common W4A8 layout. Real recipes differ in
  which tensors they quantize and at what granularity.
- The recipe table summarizes published papers and model cards from memory
  of their headline claims; check the sources for exact numbers.

## Tests

```bash
pnpm install
pnpm test       # Playwright against the installed Chrome, or bundled Chromium in CI
```

The tests open the page at 1440 px and 390 px and fail on any console or
page error or on horizontal overflow, then exercise the quantizer, the
number line, the heatmap outlier toggle, the loop stepper, the drift reset,
and a full training-lab run including a bit-width toggle mid-run.
