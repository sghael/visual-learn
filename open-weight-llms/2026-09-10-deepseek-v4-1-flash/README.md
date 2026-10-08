# DeepSeek-V4.1-Flash

Share cached history across layers and do less work when reading input

Open `index.html` directly in a browser, or follow this folder from the collection index. All runtime CSS and JavaScript are local; there is no build step. Google Fonts are optional and have system fallbacks.

## What the tutorial teaches

DeepSeek-V4.1-Flash splits the model into an encoder that prepares input once and a decoder that generates, shares cached entries across layers and stores them in four bits, cutting the cost of input-heavy agent work.

- Reading input and writing output are different jobs
- Building the decoder’s memory from the encoder
- Sharing the cache across layers
- Fast memory, persistent storage and recomputation
- Several mechanisms in one release

## Figures

- **Prepare the input separately from generating** (`flow`): Simplified encoder–decoder flow. Each decoder layer’s local sliding-window work and bounded replay are left out of this outline of the global cache path. The decoder’s memory is built from the encoder’s output, so prompt tokens do not need full decoding.
- **Share across layers; use fewer bits per number** (`cache`): Simplified accounting for a uniform global cache. It omits local-window caches, indexer keys, scale factors, nonuniform sharing and allocator overhead; it is not a memory estimator for V4.1. Sharing across layers and using fewer bits reduce different factors, so their savings multiply.

## Accuracy and scope

Release date: 2026-09-10. DeepSeek released the model and linked the weights on Hugging Face on September 10, 2026. The technical report was posted on September 17.

- The cache figure uses made-up uniform sizes and cannot estimate the release’s total runtime memory.
- Bounded replay reconstructs the local state approximately; it does not exactly recompute the full history.
- The active parameter counts for prefill and decode are not the memory needed to store the weights.

The page includes primary source links next to claims and a source list. Research cutoff: 2026-10-03. Quantitative widgets are explicitly simplified teaching models, not executed model inference or performance benchmarks.

## Maintenance

Edit `lesson.json`; run `pnpm author` from the collection directory to refresh committed static files. Run `pnpm test` there. The optional authoring utility is not required to view or deploy this page.
