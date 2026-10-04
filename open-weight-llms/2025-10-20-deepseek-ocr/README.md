# DeepSeek-OCR

Can a page stand in for a long string of tokens?

Open `index.html` directly in a browser, or follow this folder from the collection index. All runtime CSS and JavaScript are local; there is no build step. Google Fonts are optional and have system fallbacks.

## What the tutorial teaches

A visual encoder compresses a page into a shorter sequence from which a language decoder reconstructs its text.

- Count positions before counting savings
- Compress before the expensive global stage
- A resolution budget is a reading budget
- Reading back is only the first test
- Ask what survived the bottleneck

## Figures

- **The same page, two hypothetical input lengths** (`bars`): Illustrative input-position counts for a hypothetical page. Text and vision positions encode different representations. This is neither a measured compression result nor a comparison of file sizes. A shorter input sequence is a useful quantity, but it does not measure fidelity or total latency.
- **Where the sequence gets shorter** (`flow`): Simplified DeepEncoder-to-decoder path. The bridge compresses the sequence before the global vision stage; the decoder emits text tokens. The costly globally connected stage receives an already compressed visual sequence.
- **Change the number of visual positions** (`patches`): Illustrative patch-count exercise on a 16 × 12 grid. The N² readout counts dense vision-attention pairs only; actual DeepEncoder uses local attention, a learned compression bridge, and a global stage. Reducing spatial positions can sharply reduce pairwise work while making fine details harder to preserve.

## Accuracy and scope

Release date: 2025-10-20. The official repository records the public model and weight release on October 20, 2025; the arXiv paper appeared October 21.

- The input-position bar chart is hypothetical, not a published quality or speed result.
- The patch widget is an abstract counting exercise, not DeepEncoder tokenization.
- OCR reconstruction does not establish general long-context reasoning performance.

The page includes primary source links next to claims and a source list. Research cutoff: 2026-10-03. Quantitative widgets are explicitly simplified teaching models, not executed model inference or performance benchmarks.

## Maintenance

Edit `lesson.json`; run `pnpm author` from the collection directory to refresh committed static files. Run `pnpm test` there. The optional authoring utility is not required to view or deploy this page.
