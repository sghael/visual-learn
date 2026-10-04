# Gemma 2 · 9B and 27B

How a teacher can make each training example more informative

Open `index.html` directly in a browser, or follow this folder from the collection index. All runtime CSS and JavaScript are local; there is no build step. Google Fonts are optional and have system fallbacks.

## What the tutorial teaches

Gemma 2’s smaller models learn a teacher’s probability distribution, while alternating local and global attention limits memory.

- One observed word hides many possibilities
- Train on a distribution rather than a single label
- A small numerical example explains the objective
- Not every layer needs to inspect the whole past
- Distillation changes training cost, not the student’s identity

## Figures

- **Reveal the alternatives in a teacher’s prediction** (`distill`): Illustrative logits and temperature. This is token-distribution distillation, not sequence imitation and not Gemma 2’s disclosed temperature recipe. Soft targets communicate relationships among alternatives.
- **Compare one local layer with one global layer** (`attention`): Simplified 16-token causal mask. Local/full presets represent different layer types; the hybrid preset is a generic toy, not Gemma 2’s alternating-layer architecture. Locality restricts direct access in a layer while global layers restore long-range access.

## Accuracy and scope

Release date: 2024-06-27. June 27, 2024 public release of 9B and 27B weights; not the earlier preview or later 2B release.

- The June date refers to 9B and 27B; 2B arrived later.
- The teacher widget uses arbitrary probabilities and does not reproduce Gemma’s training hyperparameters.
- Attention masks are scaled teaching examples, not a model execution trace.

The page includes primary source links next to claims and a source list. Research cutoff: 2026-10-03. Quantitative widgets are explicitly simplified teaching models, not executed model inference or performance benchmarks.

## Maintenance

Edit `lesson.json`; run `pnpm author` from the collection directory to refresh committed static files. Run `pnpm test` there. The optional authoring utility is not required to view or deploy this page.
