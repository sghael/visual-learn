# DeepSeek-V3

Make sparse training efficient all the way through the system

Open `index.html` directly in a browser, or follow this folder from the collection index. All runtime CSS and JavaScript are local; there is no build step. Google Fonts are optional and have system fallbacks.

## What the tutorial teaches

Routing biases, mixed-precision arithmetic and multi-token prediction complement the MLA and MoE inherited from V2.

- A sparse model can still have a traffic jam
- Adjust access without redefining the expert’s contribution
- Use fewer bits where they remain reliable
- Train a representation to look farther ahead
- A training report is not a company budget

## Figures

- **Separate selection from contribution** (`flow`): Simplified routing-bias sequence. The figure omits top-k normalization, distributed implementation and the small complementary sequence loss. Load feedback changes who is selected; original affinity scores still weight selected outputs.
- **Bits change the size of an ideal weight array** (`precision`): Simplified uniform weight-storage model for an invented 20B-parameter array. It excludes scales, mixed precision, activations, KV cache, sharding and accuracy. Halving bits halves ideal stored bytes; it does not automatically halve training time.

## Accuracy and scope

Release date: 2024-12-26. Original DeepSeek-V3 public release, December 26, 2024. Later V3-0324 and V3.1 checkpoints are separate revisions.

- The precision widget is a generic ideal weight-storage calculation, not the full V3 training recipe.
- Auxiliary-loss-free balancing coexists with a small sequence-level auxiliary loss.
- The original December 2024 weights use the DeepSeek Model License; later family revisions may have different licensing.

The page includes primary source links next to claims and a source list. Research cutoff: 2026-10-03. Quantitative widgets are explicitly simplified teaching models, not executed model inference or performance benchmarks.

## Maintenance

Edit `lesson.json`; run `pnpm author` from the collection directory to refresh committed static files. Run `pnpm test` there. The optional authoring utility is not required to view or deploy this page.
