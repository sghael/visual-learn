# Phi-4-mini-flash-reasoning

How a hybrid decoder reuses memory instead of repeatedly reading the whole past

Open `index.html` directly in a browser, or follow this folder from the collection index. All runtime CSS and JavaScript are local; there is no build step. Google Fonts are optional and have system fallbacks.

## What the tutorial teaches

Phi-4-mini-flash-reasoning stores the full history once, shares it across layers, and lets some layers reuse earlier results through gates, which speeds up long answers.

- Why long answers are slow
- A running summary of fixed size
- Keeping a summary and a full record
- Reusing an earlier result through a gate
- Speed claims need their conditions

## Figures

- **A fixed summary beside a growing record** (`state`): Made-up sizes for two separate memory components. Phi-4-mini-flash-reasoning is a hybrid that still has an attention cache, so this is not its total memory. A fixed-size state caps one part of memory by compressing the past.
- **Reuse results across the decoder** (`flow`): Simplified SambaY layout. The real model interleaves the layers and includes extra projections, feed-forward layers and a variant of attention called differential attention. Some later layers read the shared cache; others reuse a result computed earlier.

## Accuracy and scope

Release date: 2025-07-09. Microsoft released the weights and announced the SambaY paper on July 9, 2025.

- The flat line in the memory figure describes one component only; the real model still stores a full attention cache.
- The gate example shows the arithmetic idea, not the full published formula.
- The reported throughput depends on the workload and software; it is not a general speed-up.

The page includes primary source links next to claims and a source list. Research cutoff: 2026-10-03. Quantitative widgets are explicitly simplified teaching models, not executed model inference or performance benchmarks.

## Maintenance

Edit `lesson.json`; run `pnpm author` from the collection directory to refresh committed static files. Run `pnpm test` there. The optional authoring utility is not required to view or deploy this page.
