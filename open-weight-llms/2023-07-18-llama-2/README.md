# Llama 2 · 7B, 13B, 70B

How an open chat model learned to answer, and to remember more cheaply

Open `index.html` directly in a browser, or follow this folder from the collection index. All runtime CSS and JavaScript are local; there is no build step. Google Fonts are optional and have system fallbacks.

## What the tutorial teaches

Llama 2 paired a documented chat-training pipeline with grouped-query attention in its 70B release.

- Prediction is a starting point
- Demonstrations teach a response format
- Every new token consults the past
- Count the memory yourself
- Two kinds of progress can reinforce each other

## Figures

- **From continuation to assistant** (`flow`): Simplified Llama 2-Chat training sequence; the real pipeline iterated data collection and model updates. Each stage changes what feedback the model receives.
- **Share stored keys and values** (`kv`): Illustrative bf16 KV-only calculation: 32 layers, 32 query heads, 8 grouped KV heads, width 128. These are teaching dimensions, not Llama 2 70B dimensions. Excludes weights, activations, batch, allocator and sharding. Grouping reduces the memory slope; longer context still grows the cache.

## Accuracy and scope

Release date: 2023-07-18. First public release of base and chat weights on July 18, 2023.

- Included as a July 2023 foundation slightly outside an exact three-year cutoff.
- The KV widget uses illustrative dimensions and cannot predict throughput or model quality.
- Llama 2 adopted GQA and RLHF; it did not invent them.

The page includes primary source links next to claims and a source list. Research cutoff: 2026-10-03. Quantitative widgets are explicitly simplified teaching models, not executed model inference or performance benchmarks.

## Maintenance

Edit `lesson.json`; run `pnpm author` from the collection directory to refresh committed static files. Run `pnpm test` there. The optional authoring utility is not required to view or deploy this page.
