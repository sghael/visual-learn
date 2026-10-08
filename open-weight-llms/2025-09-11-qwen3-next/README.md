# Qwen3-Next

How a running memory and a searchable history cooperate

Open `index.html` directly in a browser, or follow this folder from the collection index. All runtime CSS and JavaScript are local; there is no build step. Google Fonts are optional and have system fallbacks.

## What the tutorial teaches

Qwen3-Next makes three of every four layers update a fixed-size memory instead of a growing cache, keeps full attention in the rest, and uses about 3 billion of its 80 billion parameters per token.

- Two kinds of remembering
- Correcting a memory instead of adding to it
- Keeping a direct route to the past
- Speed depends on the serving software

## Figures

- **A fixed state beside a growing history** (`state`): Made-up sizes for one recurrent state and one attention cache. These are not Qwen3-Next’s memory figures. A fixed-size state stops memory from growing by compressing the history.
- **The update, with one number** (`flow`): Simplified teaching sequence for the delta rule. Real Gated DeltaNet uses a matrix state and learned gates. Correcting toward a target behaves differently from repeatedly adding the same observation.

## Accuracy and scope

Release date: 2025-09-11. Qwen released the model publicly on September 11, 2025. The repository shows the weights uploaded on September 9; the page uses the public release date.

- The memory figure shows how the two kinds of memory scale. It does not implement or estimate the released model’s memory.
- Gated DeltaNet came from earlier research; Qwen3-Next’s contribution is building it into a large hybrid model.

The page includes primary source links next to claims and a source list. Research cutoff: 2026-10-03. Quantitative widgets are explicitly simplified teaching models, not executed model inference or performance benchmarks.

## Maintenance

Edit `lesson.json`; run `pnpm author` from the collection directory to refresh committed static files. Run `pnpm test` there. The optional authoring utility is not required to view or deploy this page.
