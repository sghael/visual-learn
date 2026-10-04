# Mamba

How a model can remember by updating a state instead of keeping every token

Open `index.html` directly in a browser, or follow this folder from the collection index. All runtime CSS and JavaScript are local; there is no build step. Google Fonts are optional and have system fallbacks.

## What the tutorial teaches

Step through input-dependent memory and compare a fixed recurrent state with a growing KV cache.

- Two ways to carry the past
- A distractor should not erase a useful value
- Sequential meaning need not mean sequential training
- Compression has a price

## Figures

- **History grows; a recurrent state stays fixed** (`state`): Simplified single-layer storage example with illustrative dimensions. It excludes weights, temporary buffers, and training activations. A fixed-size state shifts the burden from storage to learned compression.
- **Make forgetting depend on the input** (`recurrence`): Simplified scalar recurrence. The gates are hand-chosen; no network is trained here. The same amount of state can behave differently when its update depends on content.
- **One sequence, two execution settings** (`flow`): Simplified execution schematic; real kernels fuse several operations. Parallel training and recurrent generation can implement the same sequence transformation.

## Accuracy and scope

Release date: 2023-12-03. Official checkpoint history dates the first weight-bearing model release to December 3, 2023.

- The recurrence is a hand-designed teaching example and omits learned projections, multidimensional state, discretization, convolution, and gating in the actual block.
- Constant recurrent state does not mean constant total training memory or unlimited reliable context.
- No benchmark speedup is inferred from the storage chart.

The page includes primary source links next to claims and a source list. Research cutoff: 2026-10-03. Quantitative widgets are explicitly simplified teaching models, not executed model inference or performance benchmarks.

## Maintenance

Edit `lesson.json`; run `pnpm author` from the collection directory to refresh committed static files. Run `pnpm test` there. The optional authoring utility is not required to view or deploy this page.
