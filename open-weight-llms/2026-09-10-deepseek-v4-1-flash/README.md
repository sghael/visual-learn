# DeepSeek-V4.1-Flash

Share cached history across layers and do less work when reading input

Open `index.html` directly in a browser, or follow this folder from the collection index. All runtime CSS and JavaScript are local; there is no build step. Google Fonts are optional and have system fallbacks.

## What the tutorial teaches

A causal encoder–decoder and cross-layer cache reuse reduce distinct costs of input-heavy agent workloads.

- Reading the prompt and writing a reply are different jobs
- Construct decoder memory from the encoder’s output
- A third compression axis runs through depth
- Keep persistent memory distinct from working memory
- Several mechanisms contribute to the release

## Figures

- **Separate prompt preparation from full generation** (`flow`): Simplified causal encoder–decoder flow. Layer-local sliding-window work and bounded replay remain outside this abbreviated global-cache path. The model can prepare decoder history from encoder states without fully decoding every prompt token.
- **Share across layers; reduce bits per cached number** (`cache`): Simplified uniform global-cache accounting. It omits local KV, indexer keys, scales, tail states, nonuniform sharing and allocator overhead; it is not a V4.1 memory estimator. Depth sharing and bit width multiply their effects because they reduce different factors.

## Accuracy and scope

Release date: 2026-09-10. Official public release and Hugging Face weight link on September 10, 2026. The technical report was submitted September 17. Verified before the October 3, 2026 research cutoff.

- The cache widget uses invented uniform dimensions and cannot estimate the release’s complete runtime memory.
- Bounded replay approximately reconstructs local state; it is not exact full-history recomputation.
- Prefill/decode active counts do not equal total weight-storage requirements.

The page includes primary source links next to claims and a source list. Research cutoff: 2026-10-03. Quantitative widgets are explicitly simplified teaching models, not executed model inference or performance benchmarks.

## Maintenance

Edit `lesson.json`; run `pnpm author` from the collection directory to refresh committed static files. Run `pnpm test` there. The optional authoring utility is not required to view or deploy this page.
