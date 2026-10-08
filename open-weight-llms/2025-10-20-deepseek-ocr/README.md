# DeepSeek-OCR

Can a page stand in for a long string of tokens?

Open `index.html` directly in a browser, or follow this folder from the collection index. All runtime CSS and JavaScript are local; there is no build step. Google Fonts are optional and have system fallbacks.

## What the tutorial teaches

DeepSeek-OCR compresses an image of a page into a few hundred vision tokens and decodes the text back out, testing whether images can hold text more compactly than text tokens.

- Counting positions
- Compressing before the expensive stage
- Fewer positions, less detail
- Reading text back is only the first test
- Checking what survives compression

## Figures

- **One page, two input lengths** (`bars`): Made-up position counts for one page. Text and vision positions hold different kinds of representation. This is not a measured compression result or a comparison of file sizes. A shorter input is useful, but its length says nothing about accuracy or total time.
- **Where the sequence gets shorter** (`flow`): Simplified path from DeepEncoder to the decoder. The bridge shortens the sequence before the global vision stage, and the decoder writes text tokens. The expensive global stage receives a sequence that has already been compressed.
- **Change the number of image positions** (`patches`): Patch-counting exercise on a 16 × 12 grid. The pairs readout counts full-attention pairs only; DeepEncoder actually uses local attention, a learned compression bridge and then a global stage. Fewer positions sharply reduce pairwise work but make fine detail harder to keep.

## Accuracy and scope

Release date: 2025-10-20. DeepSeek’s repository records the release of the model and weights on October 20, 2025. The paper appeared on arXiv on October 21.

- The position-count chart is made up; it is not a published quality or speed result.
- The patch figure is an abstract counting exercise, not DeepEncoder’s tokenization.
- Accurate text reconstruction does not show good reasoning over long contexts.

The page includes primary source links next to claims and a source list. Research cutoff: 2026-10-03. Quantitative widgets are explicitly simplified teaching models, not executed model inference or performance benchmarks.

## Maintenance

Edit `lesson.json`; run `pnpm author` from the collection directory to refresh committed static files. Run `pnpm test` there. The optional authoring utility is not required to view or deploy this page.
