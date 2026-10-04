# BitNet b1.58 2B4T

How a network learns to compute with minus one, zero, and plus one

Open `index.html` directly in a browser, or follow this folder from the collection index. All runtime CSS and JavaScript are local; there is no build step. Google Fonts are optional and have system fallbacks.

## What the tutorial teaches

Round a weight onto a ternary grid and distinguish native low-bit training from compressing a finished model.

- Three choices are enough to define a different arithmetic
- Learning needs a path through rounding
- A model and an implementation must agree
- Precision savings do not measure language quality

## Figures

- **A continuous latent weight, a discrete forward value** (`ternary`): Simplified fixed-scale ternary quantizer. Actual BitLinear scales are derived from weight statistics. The network must learn while its forward computation sees a coarse grid.
- **Train with the restriction already present** (`flow`): Simplified quantization-aware training loop; it omits normalization, optimizer state, and activation details. Native low-bit training gives the optimizer opportunities to compensate.

## Accuracy and scope

Release date: 2025-04-14. Official repository release note; technical report followed on arXiv.

- The displayed scale is fixed for teaching; actual quantization uses tensor statistics.
- Theoretical code entropy, physical storage format, and total runtime memory are different quantities.
- No hardware speedup or accuracy estimate is computed by this page.

The page includes primary source links next to claims and a source list. Research cutoff: 2026-10-03. Quantitative widgets are explicitly simplified teaching models, not executed model inference or performance benchmarks.

## Maintenance

Edit `lesson.json`; run `pnpm author` from the collection directory to refresh committed static files. Run `pnpm test` there. The optional authoring utility is not required to view or deploy this page.
