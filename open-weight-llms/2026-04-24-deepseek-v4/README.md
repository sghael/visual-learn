# DeepSeek-V4

Compress the history before deciding where to look

Open `index.html` directly in a browser, or follow this folder from the collection index. All runtime CSS and JavaScript are local; there is no build step. Google Fonts are optional and have system fallbacks.

## What the tutorial teaches

DeepSeek-V4 merges groups of tokens into compressed cache entries, selects among them for distant context and keeps a local path for recent detail, making million-token contexts cheaper.

- Compressing features versus compressing time
- Compress, then select
- A coarse overview alongside local detail
- Keeping the residual streams stable
- What the release contains

## Figures

- **Choose among compressed entries** (`sparse`): Simplified CSA selection step. Each of the 16 candidates is a compressed entry; the figure omits the learned compressor and the parallel local window. Compression reduces the number of candidates; selection reduces how many of them the expensive step reads.
- **Alternate two affordable views of the history** (`flow`): Simplified alternation of CSA and HCA layers, not the release’s exact layer order. Each layer combines its own view of the distant history with local attention. Full attention is affordable when its input has already been compressed.

## Accuracy and scope

Release date: 2026-04-24. DeepSeek released preview weights for V4-Pro and V4-Flash on April 24, 2026. The technical report describes the architecture; the later 0731 and 0813 updates did not change it.

- Token counts and compression ratios in the examples are made up; they are not V4’s settings.
- The selection figure starts after compression and omits the local path.
- The June report describes the architecture of the April preview; later checkpoint updates are not treated as new designs.

The page includes primary source links next to claims and a source list. Research cutoff: 2026-10-03. Quantitative widgets are explicitly simplified teaching models, not executed model inference or performance benchmarks.

## Maintenance

Edit `lesson.json`; run `pnpm author` from the collection directory to refresh committed static files. Run `pnpm test` there. The optional authoring utility is not required to view or deploy this page.
