# Open-weight language models

A collection of visual tutorials tracing model innovations from roughly 2023 through October 3, 2026. Open `index.html` to browse by date, model name, or mechanism. Each dated model subfolder has its own standalone page, stylesheet, script, source-backed lesson, and README.

## Open the collection

Open `index.html` directly from disk, or serve this repository with a static server bound to `0.0.0.0`. There is no application build, framework, backend, or model download. All content and calculations are local. Optional Google Fonts have system fallbacks. Internal URLs are relative, including when deployed below `/visual-learn/open-weight-llms/`.

## What it teaches

The collection covers training data, grouped-query and latent attention, expert routing, low-precision training and storage, verifiable rewards, reasoning and tool loops, recurrent and hybrid sequence models, masked diffusion, multimodality, and reproducible training pipelines. A dated tutorial may cover multiple model sizes that share a mechanism. See the index and `manifest.json` for the current coverage list.

## How the figures work

- **KV cache:** vary context length and MHA/GQA/MQA head sharing; compute bf16 cache bytes for one sequence.
- **Experts:** change toy token routing and number of selected experts; separate selected work from stored capacity.
- **Attention sink:** compare normalization with and without a learned extra score; the toy shows how probability mass can contribute no value vector.
- **Attention:** inspect causal full/local masks; the explicitly named toy hybrid is not a model's actual layer pattern.
- **Sparse indexing:** choose a query and top-k budget; the indexer scores every key while expensive attention reads the selected set.
- **Latent/cache/state:** compare a smaller per-token representation, cross-layer cache reuse, and constant recurrent state. These are separate compression axes.
- **GRPO:** compare group-relative advantages, including a group with identical rewards and no within-group signal.
- **Verifier:** expose differences between final-number and exact-string checking; neither rule is the production Tülu verifier.
- **Distillation:** change a toy softmax temperature and inspect the probability distribution. Sequence imitation is distinguished in the prose.
- **Precision/ternary:** calculate ideal packed bytes or move a latent value across a discrete quantization boundary.
- **Scaling/mixture:** hold a teaching compute or token budget fixed and change its allocation; no quality is predicted.
- **Parallel tasks:** schedule independent invented task durations across workers; no model performance is estimated.
- **Patches:** vary abstract patch size and count image tokens and dense self-attention pairs; it is not an actual image encoder.
- **Recurrence/diffusion:** manually step a scalar gated state or a scripted unmasking schedule. Example changes and Reset return to a labeled initial state.
- **Static figures:** directly labeled bars compare commensurate quantities; ordered stage diagrams explain training or inference paths.

Desktop prose uses a 54rem column. Captions move below figures below 1280px, preserving the wider reading area on tablets.

All figures start in a labeled state. Nothing animates automatically. Controls support keyboard and touch, with visible focus; wide figures scroll within their own region. Colors have accompanying labels.

## Research and limitations

[Research method and family memos](research/README.md) record source reading, date evidence, and exclusions. Every model page has primary citations and explicit caveats. The widgets are simplified calculations, not model inference, benchmarks, or accuracy predictions. The collection is a curated technical history, not a census of all checkpoints or a current model recommendation.

## Maintain and test

`lesson.json` is the editable source for each tutorial. Shared authoring files are copied into each delivered page so the model folder remains self-contained. Run `pnpm author` after changing lesson content or authoring templates. Commit the generated static files; deployment performs no build.

Install the pinned tooling with `pnpm install --frozen-lockfile`, then run `pnpm test`. Behavior checks exercise every instance of each widget, including model-specific defaults. Layout checks cover every tutorial at desktop and phone widths, plus the caption breakpoint. Tests use installed Google Chrome locally and fall back to Playwright Chromium in CI. Web fonts are stubbed for repeatable, offline tests. `CAPTURE=1 pnpm test` also writes viewport-sized QA captures under `/private/tmp/open-weight-qa` on macOS (or the system temporary directory on other platforms).
