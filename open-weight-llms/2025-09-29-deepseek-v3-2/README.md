# DeepSeek-V3.2 / Exp

A cheap first pass decides which earlier tokens get expensive attention

Open `index.html` directly in a browser, or follow this folder from the collection index. All runtime CSS and JavaScript are local; there is no build step. Google Fonts are optional and have system fallbacks.

## What the tutorial teaches

DeepSeek-V3.2 uses a small, fast indexer to choose which earlier tokens each query reads, so the expensive attention step touches only a selected subset.

- A smaller record still leaves a long search
- Different questions select different tokens
- Training the index before relying on it
- Cheaper reading makes long tasks practical
- Two meanings of “sparse”

## Figures

- **The indexer scores everything; attention reads the chosen few** (`sparse`): Simplified DSA selection with fixed, made-up scores. The indexer still scores every candidate, and the selection is not taken from the real model. Selecting by content can keep distant tokens that a fixed local window would drop.
- **From full reading to selective reading** (`flow`): Simplified continued-training sequence. Stage lengths and settings are omitted. A trained model needs time to adapt when its access to earlier tokens changes.

## Accuracy and scope

Release date: 2025-09-29. DeepSeek Sparse Attention first shipped in the V3.2-Exp weights on September 29, 2025. The full V3.2 and V3.2-Speciale followed on December 1.

- The selection figure uses made-up scores and does not run a model.
- The lightning indexer still does work for every earlier token, so total cost still grows with length; only the expensive attention step becomes sparse.
- V3.2-Exp and the full V3.2 share the DSA architecture but differ in post-training.

The page includes primary source links next to claims and a source list. Research cutoff: 2026-10-03. Quantitative widgets are explicitly simplified teaching models, not executed model inference or performance benchmarks.

## Maintenance

Edit `lesson.json`; run `pnpm author` from the collection directory to refresh committed static files. Run `pnpm test` there. The optional authoring utility is not required to view or deploy this page.
