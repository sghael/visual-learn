# Gemma 3 · 1B, 4B, 12B, 27B

How mostly local attention makes room for images and long inputs

Open `index.html` directly in a browser, or follow this folder from the collection index. All runtime CSS and JavaScript are local; there is no build step. Google Fonts are optional and have system fallbacks.

## What the tutorial teaches

Gemma 3 uses many short-window attention layers and fewer global layers, alongside an image encoder, to control inference memory.

- A model’s memory bill has more than one line
- Use a small neighborhood most of the time
- A local layer can still receive distant information
- An image needs an information-preserving route
- Lower precision addresses a different memory box

## Figures

- **Which past positions can this layer see?** (`attention`): Simplified 16-token mask; local/full represent individual layer types. Hybrid is a generic toy and is not Gemma 3’s five-local/one-global layer schedule. Fewer global layers mean fewer places storing the entire growing history.
- **Preserve detail before asking a question** (`flow`): Simplified Gemma 3 image path; crop choice and view counts vary. The diagram does not imply one fixed token total per original image. Useful visual reasoning first requires useful visual evidence.
- **Weight precision is a separate knob** (`precision`): Illustrative ideal storage for twelve billion weights, not an actual Gemma 3 checkpoint footprint. Excludes scales, mixed precision, KV cache, visual components, activations and accuracy effects. Fewer bits shrink weights; attention design controls another source of memory.

## Accuracy and scope

Release date: 2025-03-12. Public Gemma 3 weight release March 12, 2025; this lesson distinguishes the 1B and larger variants.

- 1B is text-only and does not share the larger models’ full context and image feature set.
- Neither toy represents measured Gemma throughput or device fit.
- A nominal context limit does not guarantee reliable retrieval.

The page includes primary source links next to claims and a source list. Research cutoff: 2026-10-03. Quantitative widgets are explicitly simplified teaching models, not executed model inference or performance benchmarks.

## Maintenance

Edit `lesson.json`; run `pnpm author` from the collection directory to refresh committed static files. Run `pnpm test` there. The optional authoring utility is not required to view or deploy this page.
