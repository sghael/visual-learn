# Qwen3.8-Flash-Next

How lookup memory, sparse retrieval and depth gates divide the work

Open `index.html` directly in a browser, or follow this folder from the collection index. All runtime CSS and JavaScript are local; there is no build step. Google Fonts are optional and have system fallbacks.

## What the tutorial teaches

Qwen3.8-Flash-Next adds micro-block sparse attention, gated residual streams and offloaded n-gram embeddings.

- Different forms of memory have different prices
- A table can store capacity without evaluating every entry
- Select small blocks before expensive attention
- Gates choose what each layer reads

## Figures

- **Move only the needed table entry** (`flow`): Simplified n-gram lookup pipeline, not an exact Qwen hashing or transfer implementation. Stored capacity can grow without reading every parameter for each token.
- **Distinguish stored history from a running state** (`state`): Illustrative state and KV scalar counts. The widget does not implement QSA selection or estimate released-model memory. Compression and selective retrieval address different parts of the context cost.

## Accuracy and scope

Release date: 2026-08-26. Public weight release August 26, 2026; architecture report August 31.

- The 51B embedding table is additional stored capacity, not included in the 125B backbone count.
- The n-gram lookup and scalar-gate examples are teaching analogies, not implementation code.

The page includes primary source links next to claims and a source list. Research cutoff: 2026-10-03. Quantitative widgets are explicitly simplified teaching models, not executed model inference or performance benchmarks.

## Maintenance

Edit `lesson.json`; run `pnpm author` from the collection directory to refresh committed static files. Run `pnpm test` there. The optional authoring utility is not required to view or deploy this page.
