# BitNet b1.58 2B4T

How a network learns to compute with minus one, zero, and plus one

Open `index.html` directly in a browser, or follow this folder from the collection index. All runtime CSS and JavaScript are local; there is no build step. Google Fonts are optional and have system fallbacks.

## What the tutorial teaches

BitNet b1.58 2B4T trained a 2-billion-parameter model whose weights take only the values −1, 0 and +1, instead of rounding a finished model afterward.

- Weights with three values
- Learning through rounding
- The model needs matching software
- Fewer bits do not prove a better model

## Figures

- **A precise latent weight, a three-valued calculation** (`ternary`): Simplified ternary rounding with a fixed scale of 0.5. In BitLinear, the scale comes from the average absolute value of each weight matrix. The model trains a precise hidden value, but its calculations see only three levels.
- **Train with the restriction already in place** (`flow`): Simplified quantization-aware training loop. It omits normalization, optimizer state and the handling of activations. Training under the restriction lets the optimizer learn to work around it.

## Accuracy and scope

Release date: 2025-04-14. Microsoft’s BitNet repository announced the model on April 14, 2025. The technical report followed on arXiv.

- The figure fixes the scale for teaching; the real model computes it from each weight matrix.
- Bits of information per weight, the storage format on disk and total runtime memory are three different quantities.
- This page computes no speed-up or accuracy estimate.

The page includes primary source links next to claims and a source list. Research cutoff: 2026-10-03. Quantitative widgets are explicitly simplified teaching models, not executed model inference or performance benchmarks.

## Maintenance

Edit `lesson.json`; run `pnpm author` from the collection directory to refresh committed static files. Run `pnpm test` there. The optional authoring utility is not required to view or deploy this page.
