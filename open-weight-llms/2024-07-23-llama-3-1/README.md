# Llama 3.1 · 8B, 70B, 405B

How training scale, data selection, and a long context work together

Open `index.html` directly in a browser, or follow this folder from the collection index. All runtime CSS and JavaScript are local; there is no build step. Google Fonts are optional and have system fallbacks.

## What the tutorial teaches

Llama 3.1 showed how a largely conventional dense transformer could be extended through data, scale and post-training.

- A release larger than its architecture change
- Training and serving favor different choices
- A token count needs a data recipe
- Long context needs more than an allowed length
- Good demonstrations still need selection

## Figures

- **Spend a fixed training budget** (`scaling`): Simplified dense-model compute equivalence using 6ND. The seven-billion/one-trillion baseline is illustrative, not a Llama 3.1 training run; no accuracy or latency is predicted. The same training budget can buy more parameters or more examples.
- **A model recipe has several knobs** (`flow`): Simplified organization of the Llama 3.1 development problem, not a complete production data pipeline. Architecture, data and behavior are separate experimental variables.

## Accuracy and scope

Release date: 2024-07-23. Public 3.1 weight release; the earlier Llama 3 8B/70B launch was April 18, 2024.

- The compute widget is an approximation, not a fitted scaling law or hardware benchmark.
- The paper’s multimodal research models are not counted as Llama 3.1 released weights.
- The 128K context budget covers prompt and continuation and does not guarantee reliable use of every detail.

The page includes primary source links next to claims and a source list. Research cutoff: 2026-10-03. Quantitative widgets are explicitly simplified teaching models, not executed model inference or performance benchmarks.

## Maintenance

Edit `lesson.json`; run `pnpm author` from the collection directory to refresh committed static files. Run `pnpm test` there. The optional authoring utility is not required to view or deploy this page.
