# GLM-4.5

How a curriculum keeps reinforcement learning informative

Open `index.html` directly in a browser, or follow this folder from the collection index. All runtime CSS and JavaScript are local; there is no build step. Google Fonts are optional and have system fallbacks.

## What the tutorial teaches

GLM-4.5 combined reasoning, coding and tool use in one sparse model, and kept its reinforcement learning effective by raising problem difficulty as the model improved.

- Training on whole repositories
- A group needs differences to learn from
- Difficulty depends on the learner
- One model for reasoning, code and tools

## Figures

- **Show the connections before teaching actions** (`flow`): Simplified outline of GLM-4.5’s training stages. Each stage includes more data types and settings than shown. Longer training examples show dependencies that snippets leave out.
- **When the relative signal disappears** (`grpo`): Simplified GRPO-style signal. It omits probability ratios, clipping and per-token weighting. The GLM-4.5 report’s reasoning recipe also drops the drift penalty used in some GRPO versions. Complete success and complete failure both erase the comparison within a group.

## Accuracy and scope

Release date: 2025-07-28. Z.ai released the weights on July 28, 2025. The report was posted on August 8.

- The reward figure is a simplified group-relative calculation, not GLM-4.5’s full optimizer.
- Parameter counts follow the release’s own conventions; check how each family counts before comparing them.

The page includes primary source links next to claims and a source list. Research cutoff: 2026-10-03. Quantitative widgets are explicitly simplified teaching models, not executed model inference or performance benchmarks.

## Maintenance

Edit `lesson.json`; run `pnpm author` from the collection directory to refresh committed static files. Run `pnpm test` there. The optional authoring utility is not required to view or deploy this page.
