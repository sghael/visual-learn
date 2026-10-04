# Llama 4 · Scout and Maverick

How sparse experts and image tokens share one model

Open `index.html` directly in a browser, or follow this folder from the collection index. All runtime CSS and JavaScript are local; there is no build step. Google Fonts are optional and have system fallbacks.

## What the tutorial teaches

Llama 4 brought shared-and-routed experts plus joint image/text processing to Meta’s released model family.

- Start with the weights that were released
- One specialist route plus a shared route
- Active parameters do not measure total memory
- Images become vectors before language generation
- Measure each benefit on its own terms

## Figures

- **Follow one token through a sparse layer** (`moe`): Simplified eight-expert routing toy with one always-active shared expert. Maverick actually has 128 routed experts and selects one per token. Changing active count explores the general mechanism, not a runtime setting of Llama 4. Sparse execution reduces selected work while all expert weights still exist.
- **Bring visual information into the conversation** (`flow`): Simplified conceptual path for Llama 4 image/text input; omits image tiling, projections, layer details and training losses. A shared token space permits interaction between visual evidence and language.

## Accuracy and scope

Release date: 2025-04-05. Scout and Maverick weights became available April 5, 2025; Behemoth was only previewed.

- No Behemoth public-weight release is asserted.
- The routing diagram is scaled down and does not name experts by topic.
- No speedup follows directly from dividing total by active parameters.
- The tutorial does not claim every supported-context token is used reliably.

The page includes primary source links next to claims and a source list. Research cutoff: 2026-10-03. Quantitative widgets are explicitly simplified teaching models, not executed model inference or performance benchmarks.

## Maintenance

Edit `lesson.json`; run `pnpm author` from the collection directory to refresh committed static files. Run `pnpm test` there. The optional authoring utility is not required to view or deploy this page.
