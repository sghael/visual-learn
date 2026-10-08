# DeepSeek-V3

How DeepSeek trained a 671-billion-parameter model without wasting its hardware

Open `index.html` directly in a browser, or follow this folder from the collection index. All runtime CSS and JavaScript are local; there is no build step. Google Fonts are optional and have system fallbacks.

## What the tutorial teaches

DeepSeek-V3 scaled V2’s design to 671 billion parameters and trained it efficiently with bias-based expert balancing, 8-bit arithmetic and multi-token prediction.

- A sparse model can still have traffic jams
- Steering traffic without changing what experts contribute
- Training in 8-bit numbers
- Predicting more than one token ahead
- What the training-cost figure covers

## Figures

- **Separate choosing an expert from weighting its output** (`flow`): Simplified routing with a load-balancing bias. It omits normalization of the selected scores, the distributed implementation and the small per-sequence penalty. The bias changes which experts are chosen; the original scores still weight their outputs.
- **Bits set the size of an ideal weight array** (`precision`): Simplified storage for a made-up array of 20 billion weights stored uniformly. It excludes scale factors, mixed precision, activations, the KV cache, sharding and any effect on accuracy. Halving the bits halves the ideal storage; it does not automatically halve training time.

## Accuracy and scope

Release date: 2024-12-26. DeepSeek released the original V3 weights on December 26, 2024. The later V3-0324 and V3.1 checkpoints are separate revisions.

- The precision figure is a generic storage calculation, not V3’s full training recipe.
- Auxiliary-loss-free balancing still includes a very small per-sequence balancing penalty.
- The original December 2024 weights use DeepSeek’s model license; later revisions of the family may use different licenses.

The page includes primary source links next to claims and a source list. Research cutoff: 2026-10-03. Quantitative widgets are explicitly simplified teaching models, not executed model inference or performance benchmarks.

## Maintenance

Edit `lesson.json`; run `pnpm author` from the collection directory to refresh committed static files. Run `pnpm test` there. The optional authoring utility is not required to view or deploy this page.
