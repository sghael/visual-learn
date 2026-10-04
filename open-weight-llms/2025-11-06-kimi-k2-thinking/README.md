# Kimi K2 Thinking

How long reasoning changes the economics of precision

Open `index.html` directly in a browser, or follow this folder from the collection index. All runtime CSS and JavaScript are local; there is no build step. Google Fonts are optional and have system fallbacks.

## What the tutorial teaches

Kimi K2 Thinking pairs reasoning interleaved with tools with quantization-aware post-training for its expert weights.

- Reason, act, observe, then revise
- Long generation repeatedly reads the weights
- An early error can change all later steps
- Count the work across the whole task

## Figures

- **A loop that can change its own plan** (`flow`): Simplified tool-using reasoning loop, not an extracted K2 Thinking trace. Useful extra computation responds to evidence.
- **Count bits before predicting performance** (`precision`): Illustrative uniform weight array. Excludes scales, mixed precision, activations, KV cache, sharding and accuracy effects. Four-bit storage is one quarter of sixteen-bit storage for the same uniformly encoded weights.

## Accuracy and scope

Release date: 2025-11-06. Public release November 6, 2025.

- The weight-storage widget is deliberately a 20B toy array, not a Kimi deployment sizing calculator.
- Vendor throughput and tool-horizon reports do not establish a general speedup or reliability guarantee.

The page includes primary source links next to claims and a source list. Research cutoff: 2026-10-03. Quantitative widgets are explicitly simplified teaching models, not executed model inference or performance benchmarks.

## Maintenance

Edit `lesson.json`; run `pnpm author` from the collection directory to refresh committed static files. Run `pnpm test` there. The optional authoring utility is not required to view or deploy this page.
