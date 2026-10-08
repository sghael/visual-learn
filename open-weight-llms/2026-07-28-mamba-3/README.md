# Mamba-3 · MIMO 1.5B

How a recurrent state can remember more through richer updates

Open `index.html` directly in a browser, or follow this folder from the collection index. All runtime CSS and JavaScript are local; there is no build step. Google Fonts are optional and have system fallbacks.

## What the tutorial teaches

Mamba-3 keeps a fixed-size recurrent state but updates it more richly, blending neighboring inputs, rotating the state, and doing more arithmetic for each memory access.

- The paper and the downloadable model
- Using both ends of each step
- Rotating the state as well as shrinking it
- Doing more arithmetic while the state is loaded
- A better state is still a finite state

## Figures

- **One update, more internal operations** (`flow`): Simplified outline of a Mamba-3 update. The rotation can be applied to projections; the real block also has normalization, gates, multiple heads and a feed-forward layer. Mamba-3 changes how a fixed-size state is updated and read.
- **The state’s size does not depend on sequence length** (`state`): Made-up sizes, not Mamba-3’s dimensions or measured memory. Weights, temporary buffers and per-update data are excluded. A fixed-size state is not a complete copy of the past. Richer updates improve a fixed-size memory without turning it into a growing record.

## Accuracy and scope

Release date: 2026-07-28. The paper appeared on March 16, 2026. A commit titled “Mamba-3 public release” (bc6b5d0) added the 1.5B MIMO weights to the official repository on July 28, 2026. The repository is public, although one sentence in its README still describes it as private.

- The page is dated by the official public-release commit of the weights, not by the March paper.
- The download is a base research checkpoint; the paper’s benchmark results are not automatically results for it.
- The arithmetic examples and state sizes do not reproduce the trained model, and they do not show unlimited retrieval.
- Whether extra arithmetic fits within the same generation time depends on the workload.

The page includes primary source links next to claims and a source list. Research cutoff: 2026-10-03. Quantitative widgets are explicitly simplified teaching models, not executed model inference or performance benchmarks.

## Maintenance

Edit `lesson.json`; run `pnpm author` from the collection directory to refresh committed static files. Run `pnpm test` there. The optional authoring utility is not required to view or deploy this page.
