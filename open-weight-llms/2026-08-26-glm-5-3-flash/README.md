# GLM-5.3-Flash

A new base model with compact memory layers and constrained mixing between layers

Open `index.html` directly in a browser, or follow this folder from the collection index. All runtime CSS and JavaScript are local; there is no build step. Google Fonts are optional and have system fallbacks.

## What the tutorial teaches

GLM-5.3-Flash is a new multimodal base model that mixes fixed-size memory layers with sparse attention and constrains how its four residual streams mix between layers.

- A new base, not just new tuning
- Three memory layers, then one lookup layer
- Mixing residual streams without runaway growth
- Configuration versus capability

## Figures

- **Two kinds of memory in one stack** (`state`): Made-up sizes; not an estimate of GLM-5.3-Flash’s memory. The hybrid saves on history in most layers while keeping a lookup path in a few.
- **Control the route between layers** (`flow`): Simplified mHC idea with two streams; the released model uses four. Constraining the mixing matrix keeps repeated mixing from amplifying the signal.

## Accuracy and scope

Release date: 2026-08-26. Z.ai released the weights on August 26, 2026; a placeholder repository appeared on August 25.

- Z.ai’s blog post about Flash had no readable text when this page was researched; the page relies on the model card, the configuration file and the mHC paper.
- The mHC arithmetic describes only the mixing between residual streams. It is not the whole block and does not guarantee stable training.

The page includes primary source links next to claims and a source list. Research cutoff: 2026-10-03. Quantitative widgets are explicitly simplified teaching models, not executed model inference or performance benchmarks.

## Maintenance

Edit `lesson.json`; run `pnpm author` from the collection directory to refresh committed static files. Run `pnpm test` there. The optional authoring utility is not required to view or deploy this page.
