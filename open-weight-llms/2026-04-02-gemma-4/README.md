# Gemma 4 · E2B, E4B, 26B-A4B, 31B, later 12B

How a family separates weight memory, token computation and sensory encoding

Open `index.html` directly in a browser, or follow this folder from the collection index. All runtime CSS and JavaScript are local; there is no build step. Google Fonts are optional and have system fallbacks.

## What the tutorial teaches

Gemma 4 combines sparse and dense models, compression-aware training, and a later encoder-free 12B variant under Apache 2.0.

- One family contains different architectures
- The 12B addition removes separate sensory encoders
- Moving work does not make it disappear
- Train while seeing the rounding error
- A small drafter can propose several next tokens

## Figures

- **A direct path from sensory chunks to tokens** (`flow`): Simplified conceptual path for the June 2026 Gemma 4 12B addition. Other Gemma 4 variants retain modality encoders. Removing a separate encoder shifts representation learning into the joint model.
- **What the bit budget buys** (`precision`): Illustrative ideal weight storage for 12B numbers, not measured Gemma 4 memory. Excludes scales, mixed precision, KV cache, activations and runtime overhead; the widget does not simulate quantization error. QAT addresses the quality consequences that a storage calculator cannot show.

## Accuracy and scope

Release date: 2026-04-02. First Gemma 4 weights April 2, 2026. The 12B addition launched June 3; QAT checkpoints launched June 5. Research includes the July technical report.

- The April folder date is the family launch; 12B and QAT additions are explicitly dated June.
- Effective, active and total parameter counts are not interchangeable.
- The precision figure is an ideal calculation and makes no device-fit claim.
- Encoder-free applies to the 12B addition, not every Gemma 4 variant.

The page includes primary source links next to claims and a source list. Research cutoff: 2026-10-03. Quantitative widgets are explicitly simplified teaching models, not executed model inference or performance benchmarks.

## Maintenance

Edit `lesson.json`; run `pnpm author` from the collection directory to refresh committed static files. Run `pnpm test` there. The optional authoring utility is not required to view or deploy this page.
