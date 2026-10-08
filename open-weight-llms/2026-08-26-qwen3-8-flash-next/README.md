# Qwen3.8-Flash-Next

How lookup memory, sparse retrieval and depth gates divide the work

Open `index.html` directly in a browser, or follow this folder from the collection index. All runtime CSS and JavaScript are local; there is no build step. Google Fonts are optional and have system fallbacks.

## What the tutorial teaches

Qwen3.8-Flash-Next adds large n-gram lookup tables kept in host memory, a block-based sparse attention and gated residual streams to a Qwen3-Next-style hybrid.

- Four counts that one parameter number hides
- Lookup tables for short token patterns
- Selecting small blocks before expensive attention
- Gates choose what each layer reads

## Figures

- **Move only the table row that is needed** (`flow`): Simplified n-gram lookup; not Qwen’s exact hashing or data transfer. Stored capacity can grow without reading every parameter for each token.
- **Stored history versus a running state** (`state`): Made-up sizes for a recurrent state and a stored history. The figure does not implement QSA or estimate the released model’s memory. Compressing the history and reading it selectively address different parts of the cost of context.

## Accuracy and scope

Release date: 2026-08-26. Qwen released the weights on August 26, 2026. The architecture report followed on August 31.

- The 51 billion parameters of embedding tables are stored in addition to the 125-billion-parameter backbone.
- The n-gram lookup and the one-number gate examples are teaching analogies, not implementation code.

The page includes primary source links next to claims and a source list. Research cutoff: 2026-10-03. Quantitative widgets are explicitly simplified teaching models, not executed model inference or performance benchmarks.

## Maintenance

Edit `lesson.json`; run `pnpm author` from the collection directory to refresh committed static files. Run `pnpm test` there. The optional authoring utility is not required to view or deploy this page.
