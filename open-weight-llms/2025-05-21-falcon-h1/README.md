# Falcon-H1 · 0.5B to 34B

How attention and a recurrent state can work beside each other

Open `index.html` directly in a browser, or follow this folder from the collection index. All runtime CSS and JavaScript are local; there is no build step. Google Fonts are optional and have system fallbacks.

## What the tutorial teaches

Falcon-H1 runs attention and a Mamba-2 recurrent branch side by side inside each block and mixes their outputs, instead of alternating the two in separate layers.

- Six sizes of one hybrid design
- Both branches read the same input
- A hybrid keeps both kinds of memory
- The same size can be built different ways
- Choosing a checkpoint and measuring it

## Figures

- **Two branches inside one block** (`flow`): Simplified Falcon-H1 block. Attention and Mamba-2 run as parallel branches, not one after the other. Residual connections, normalization and the feed-forward layer are omitted. Both branches process the same token representation before their outputs are combined.
- **Two parts of a hybrid’s memory** (`state`): Made-up sizes for a recurrent state and an attention cache. Falcon-H1 has both, so its memory keeps growing with the input instead of following the flat line. Weights, batching and runtime overhead are excluded. A hybrid saves memory but still keeps an attention cache that grows with the input.

## Accuracy and scope

Release date: 2025-05-21. TII announced the models on May 21, 2025. The detailed technical report appeared in July.

- A hybrid’s memory includes both a fixed-size state and an attention cache that grows with the input.
- The block diagram shows the architecture; it does not mean the two branches run simultaneously on the hardware.
- The memory figure uses made-up sizes and predicts neither real memory use nor quality.
- Size labels such as 7B are rounded names, not exact parameter counts.

The page includes primary source links next to claims and a source list. Research cutoff: 2026-10-03. Quantitative widgets are explicitly simplified teaching models, not executed model inference or performance benchmarks.

## Maintenance

Edit `lesson.json`; run `pnpm author` from the collection directory to refresh committed static files. Run `pnpm test` there. The optional authoring utility is not required to view or deploy this page.
