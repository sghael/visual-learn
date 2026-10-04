# Qwen3-Next

How a running memory and a searchable history cooperate

Open `index.html` directly in a browser, or follow this folder from the collection index. All runtime CSS and JavaScript are local; there is no build step. Google Fonts are optional and have system fallbacks.

## What the tutorial teaches

Qwen3-Next combines Gated DeltaNet state updates with occasional full attention and sparse experts.

- Two different meanings of remembering
- Correct an association instead of accumulating it
- Keep an exact-retrieval route in the stack
- Sparse arithmetic needs a matching implementation

## Figures

- **The two memory curves diverge** (`state`): Illustrative scalar counts for one toy recurrent state and one toy KV history. Not Qwen3-Next memory measurements. Fixed-size state limits memory growth by compressing history.
- **A scalar analogy for the update** (`flow`): Simplified delta-rule teaching sequence; real Gated DeltaNet uses matrix state and learned gates. Targeted correction behaves differently from repeatedly adding the same observation.

## Accuracy and scope

Release date: 2025-09-11. Public launch September 11; repository contains staged weight uploads dated September 9. The folder uses the public release date, not the staging timestamp.

- The state widget shows scaling behavior, not a faithful implementation or memory estimate for the released model.
- Gated DeltaNet originated in earlier research; Qwen3-Next’s contribution is its model and systems integration.

The page includes primary source links next to claims and a source list. Research cutoff: 2026-10-03. Quantitative widgets are explicitly simplified teaching models, not executed model inference or performance benchmarks.

## Maintenance

Edit `lesson.json`; run `pnpm author` from the collection directory to refresh committed static files. Run `pnpm test` there. The optional authoring utility is not required to view or deploy this page.
