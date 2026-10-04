# DeepSeek-V4

Compress the history before deciding where to look

Open `index.html` directly in a browser, or follow this folder from the collection index. All runtime CSS and JavaScript are local; there is no build step. Google Fonts are optional and have system fallbacks.

## What the tutorial teaches

V4 combines sequence compression, sparse retrieval and local attention to make very long contexts less expensive.

- Compression can act across features or across time
- Compress groups, then select useful groups
- A coarse global view complements local detail
- More information paths need stable mixing
- What the release establishes

## Figures

- **Choose among compressed history records** (`sparse`): Simplified CSA selection stage. Each of sixteen candidates represents a compressed record; the widget omits the learned compressor and the parallel local window. Sequence compression reduces candidates; sparse selection reduces expensive reads among them.
- **Interleave two affordable views of the global history** (`flow`): Simplified example of interleaved CSA and HCA layers, not the release’s exact layer schedule. Each layer combines its own global pattern with local attention. Dense attention can be inexpensive when its input has already been compressed.

## Accuracy and scope

Release date: 2026-04-24. Public V4-Pro and V4-Flash preview weights appeared April 24, 2026. The technical report documents the architecture; later 0731 and 0813 revisions are not new architecture introductions.

- Toy token counts and compression ratios are illustrative, not V4 configurations.
- The sparse widget models selection after compression and omits the local path.
- The June report describes the architecture after the April preview announcement; later checkpoint refreshes are not treated as separate inventions.

The page includes primary source links next to claims and a source list. Research cutoff: 2026-10-03. Quantitative widgets are explicitly simplified teaching models, not executed model inference or performance benchmarks.

## Maintenance

Edit `lesson.json`; run `pnpm author` from the collection directory to refresh committed static files. Run `pnpm test` there. The optional authoring utility is not required to view or deploy this page.
