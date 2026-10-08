# Gemma 4 · E2B, E4B, 26B-A4B, 31B, later 12B

Reading a model family whose names count parameters three different ways

Open `index.html` directly in a browser, or follow this folder from the collection index. All runtime CSS and JavaScript are local; there is no build step. Google Fonts are optional and have system fallbacks.

## What the tutorial teaches

Gemma 4 mixes sparse and dense models, counts parameters in several ways, adds a 12B model without separate image and audio encoders, and offers quantization-aware checkpoints, all under Apache 2.0.

- One family, several architectures
- The 12B model reads images and audio directly
- Moving work does not remove it
- Training with the rounding in view
- A small drafter proposes several tokens

## Figures

- **A direct path from raw input to tokens** (`flow`): Simplified path for the Gemma 4 12B model added in June 2026. The other Gemma 4 models keep separate encoders. Without a separate encoder, the main model has to learn to interpret raw image and audio pieces.
- **What the bit budget buys** (`precision`): Simplified storage for 12 billion weights, not measured Gemma 4 memory. It excludes scale factors, mixed precision, the KV cache, activations and runtime overhead, and does not simulate rounding error. The storage arithmetic is exact; QAT addresses the quality loss that the arithmetic cannot show.

## Accuracy and scope

Release date: 2026-04-02. Google released the first Gemma 4 weights on April 2, 2026. The 12B model followed on June 3 and the QAT checkpoints on June 5; the technical report appeared in July.

- The page is dated by the April family launch; the 12B model and the QAT checkpoints are dated June.
- Effective, active and total parameter counts are not interchangeable.
- The storage figure is an ideal calculation and makes no claim about fitting on a device.
- Only the 12B model is encoder-free; the other Gemma 4 models keep their encoders.

The page includes primary source links next to claims and a source list. Research cutoff: 2026-10-03. Quantitative widgets are explicitly simplified teaching models, not executed model inference or performance benchmarks.

## Maintenance

Edit `lesson.json`; run `pnpm author` from the collection directory to refresh committed static files. Run `pnpm test` there. The optional authoring utility is not required to view or deploy this page.
