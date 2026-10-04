# Qwen3.5

How visual tokens join a hybrid language model

Open `index.html` directly in a browser, or follow this folder from the collection index. All runtime CSS and JavaScript are local; there is no build step. Google Fonts are optional and have system fallbacks.

## What the tutorial teaches

Qwen3.5 combines early multimodal fusion with the recurrent-attention architecture developed in Qwen3-Next.

- The interface between pixels and language
- Three recurrent layers, then a retrieval layer
- A useful training example joins perception and consequence
- A downloadable checkpoint and a hosted product differ

## Figures

- **A shared reasoning context** (`flow`): Simplified vision-language interface. It omits image tiling, position encoding and internal encoder layers. Visual understanding requires evidence beyond transcribed words.
- **Long inputs create different memory costs** (`state`): Illustrative recurrent-state versus KV-history scaling, not a Qwen3.5 memory calculation. Multimodal input makes the cost of retaining context a practical design concern.

## Accuracy and scope

Release date: 2026-02-16. First public flagship weight release February 16, 2026; smaller members were released later.

- The diagram is conceptual; it does not specify the exact image-tokenization pipeline.
- Perception diagnostics described here are original teaching examples, not published measurements.

The page includes primary source links next to claims and a source list. Research cutoff: 2026-10-03. Quantitative widgets are explicitly simplified teaching models, not executed model inference or performance benchmarks.

## Maintenance

Edit `lesson.json`; run `pnpm author` from the collection directory to refresh committed static files. Run `pnpm test` there. The optional authoring utility is not required to view or deploy this page.
