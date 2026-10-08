# Phi-3 · mini

How a small model trained on carefully chosen and generated text kept up with much larger ones

Open `index.html` directly in a browser, or follow this folder from the collection index. All runtime CSS and JavaScript are local; there is no build step. Google Fonts are optional and have system fallbacks.

## What the tutorial teaches

Phi-3-mini trained a 3.8-billion-parameter model on heavily filtered web text and generated lessons, aiming for strong reasoning at a size that fits on a phone.

- Why training data matters more for a small model
- The idea behind “textbook-quality” data
- Phi-3’s training recipe
- How small the weights are
- Telling a skill gap from a knowledge gap

## Figures

- **Turn raw text into useful practice** (`flow`): Simplified outline of a curated-data pipeline in the spirit of the Phi research. Microsoft has not published its full pipeline at this level of detail. Filtering chooses existing material; generation rewrites a skill into a form the model can learn from.
- **Count the weight bytes** (`precision`): Simplified storage for 3.8 billion weights at each bit width. It excludes quantization metadata, the KV cache and other runtime memory, and says nothing about accuracy. Fewer parameters and fewer bits per parameter both shrink the weights, and the two savings multiply.

## Accuracy and scope

Release date: 2024-04-23. Microsoft released the Phi-3-mini weights on April 23, 2024. The small, medium and 3.5 models came later.

- This page covers mini, the first Phi-3 model with public weights.
- Careful data selection does not prove the curriculum is optimal, and it does not give the model complete factual knowledge.
- The storage figure counts weights only.

The page includes primary source links next to claims and a source list. Research cutoff: 2026-10-03. Quantitative widgets are explicitly simplified teaching models, not executed model inference or performance benchmarks.

## Maintenance

Edit `lesson.json`; run `pnpm author` from the collection directory to refresh committed static files. Run `pnpm test` there. The optional authoring utility is not required to view or deploy this page.
