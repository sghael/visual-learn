# GLM-4.5

How a curriculum keeps reinforcement learning informative

Open `index.html` directly in a browser, or follow this folder from the collection index. All runtime CSS and JavaScript are local; there is no build step. Google Fonts are optional and have system fallbacks.

## What the tutorial teaches

GLM-4.5 joins repository-scale training, tool use, and reasoning with a difficulty-aware learning curriculum.

- Learn relationships that cross file boundaries
- A group needs differences to supply a relative signal
- Difficulty belongs to the learner, not just the question
- Reasoning and tool use share the same checkpoint

## Figures

- **Expose relationships before teaching actions** (`flow`): Simplified GLM-4.5 training progression; individual stages contain additional data and optimization choices. Longer training examples can reveal dependencies that snippets omit.
- **When relative feedback disappears** (`grpo`): Pedagogical GRPO-style signal. Omits probability ratios, clipping and token-level loss. The GLM-4.5 report’s reasoning-RL recipe excludes the KL penalty used in some GRPO formulations. Perfect success and complete failure can both erase within-group comparison.

## Accuracy and scope

Release date: 2025-07-28. Public weight release July 28, 2025; report submitted August 8.

- The reward widget is a group-relative teaching calculation, not an implementation of GLM-4.5’s complete optimizer.
- Parameter accounting follows the release; compare counting conventions before comparing families.

The page includes primary source links next to claims and a source list. Research cutoff: 2026-10-03. Quantitative widgets are explicitly simplified teaching models, not executed model inference or performance benchmarks.

## Maintenance

Edit `lesson.json`; run `pnpm author` from the collection directory to refresh committed static files. Run `pnpm test` there. The optional authoring utility is not required to view or deploy this page.
