# DeepSeek-V3.2 / Exp

Use a cheap search before expensive attention

Open `index.html` directly in a browser, or follow this folder from the collection index. All runtime CSS and JavaScript are local; there is no build step. Google Fonts are optional and have system fallbacks.

## What the tutorial teaches

A lightweight indexer selects relevant cached tokens before the main attention calculation.

- A smaller record can still leave a large search
- Change the question, change the selected past
- Teach the index before depending on it
- Sparse access makes longer trajectories more practical
- Keep the two meanings of sparse apart

## Figures

- **An indexer ranks all candidates; attention reads the chosen ones** (`sparse`): Simplified DSA selection with fixed illustrative scores. The indexer still scans candidates, and the selection is not a real model trace. Selecting by content can retain distant positions that a fixed local window would exclude.
- **Adapt from dense reading to sparse reading** (`flow`): Simplified continued-training sequence. Stage lengths and optimization settings are intentionally omitted. A pretrained model needs adaptation when its access to previous tokens changes.

## Accuracy and scope

Release date: 2025-09-29. DeepSeek Sparse Attention first shipped in V3.2-Exp weights on September 29, 2025. The full V3.2 and Speciale release followed December 1; this lesson distinguishes their shared architecture from later post-training.

- The sparse widget uses invented scores and does not execute a model.
- The lightning indexer still incurs work over candidate positions; attention sparsity does not make total computation independent of length.
- Exp and full V3.2 share DSA architecture but differ in their release and post-training context.

The page includes primary source links next to claims and a source list. Research cutoff: 2026-10-03. Quantitative widgets are explicitly simplified teaching models, not executed model inference or performance benchmarks.

## Maintenance

Edit `lesson.json`; run `pnpm author` from the collection directory to refresh committed static files. Run `pnpm test` there. The optional authoring utility is not required to view or deploy this page.
