# Mamba

How a model can remember by updating a state instead of keeping every token

Open `index.html` directly in a browser, or follow this folder from the collection index. All runtime CSS and JavaScript are local; there is no build step. Google Fonts are optional and have system fallbacks.

## What the tutorial teaches

Mamba replaced the growing attention cache with a fixed-size memory whose updates depend on the current token, and found a fast way to train it on GPUs.

- Two ways to carry the past
- Why the update should depend on the input
- Training without stepping token by token
- What a fixed state cannot do

## Figures

- **History grows; a recurrent state stays fixed** (`state`): Simplified storage for one layer, with made-up sizes: 64 numbers per token for the attention cache and 256 for the recurrent state. Weights, temporary buffers and training memory are excluded. A fixed-size state trades storage for compression: the model must learn what to keep.
- **Make forgetting depend on the input** (`recurrence`): Simplified one-number memory. The gate values are chosen by hand; nothing is trained. The same memory behaves differently when its update depends on the input.
- **One sequence, two ways to compute it** (`flow`): Simplified schematic. The real GPU kernels fuse several of these operations. Training and generation compute the same transformation in different orders.

## Accuracy and scope

Release date: 2023-12-03. The official checkpoint history shows the first weights, for the 130M-parameter model, published on December 3, 2023.

- The one-number memory is a hand-built example. A real Mamba block adds learned projections, a many-number state, discretization, a short convolution and an output gate.
- A fixed recurrent state does not make training memory constant, and it does not give the model unlimited reliable context.
- The storage chart does not predict any speedup.

The page includes primary source links next to claims and a source list. Research cutoff: 2026-10-03. Quantitative widgets are explicitly simplified teaching models, not executed model inference or performance benchmarks.

## Maintenance

Edit `lesson.json`; run `pnpm author` from the collection directory to refresh committed static files. Run `pnpm test` there. The optional authoring utility is not required to view or deploy this page.
