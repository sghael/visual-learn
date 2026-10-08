# Kimi K2 Thinking

Why a model built for long reasoning was trained to run in four-bit precision

Open `index.html` directly in a browser, or follow this folder from the collection index. All runtime CSS and JavaScript are local; there is no build step. Google Fonts are optional and have system fallbacks.

## What the tutorial teaches

Kimi K2 Thinking interleaves reasoning with hundreds of tool calls, and was trained with four-bit expert weights so that long tasks move less data through memory.

- Reason, act, observe, revise
- Four-bit weights for long generation
- One early error changes everything after it
- Measuring the whole task

## Figures

- **A loop that can change its plan** (`flow`): Simplified reasoning loop with tools, not a real K2 Thinking transcript. Extra computation is useful when it responds to new evidence.
- **Count the bits before predicting performance** (`precision`): Simplified storage for a made-up uniform array of weights. It excludes scale factors, mixed precision, activations, the KV cache, sharding and any effect on accuracy. Four-bit storage is a quarter of 16-bit storage for the same weights.

## Accuracy and scope

Release date: 2025-11-06. Moonshot AI released the model on November 6, 2025.

- The storage figure uses a made-up array of 20 billion weights; it is not a sizing calculator for Kimi deployments.
- Moonshot’s reported speeds and tool-call counts do not establish a general speed-up or guarantee reliability.

The page includes primary source links next to claims and a source list. Research cutoff: 2026-10-03. Quantitative widgets are explicitly simplified teaching models, not executed model inference or performance benchmarks.

## Maintenance

Edit `lesson.json`; run `pnpm author` from the collection directory to refresh committed static files. Run `pnpm test` there. The optional authoring utility is not required to view or deploy this page.
