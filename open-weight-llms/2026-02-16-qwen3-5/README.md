# Qwen3.5

How visual tokens join a hybrid language model

Open `index.html` directly in a browser, or follow this folder from the collection index. All runtime CSS and JavaScript are local; there is no build step. Google Fonts are optional and have system fallbacks.

## What the tutorial teaches

Qwen3.5 trained images and text together on top of Qwen3-Next’s hybrid architecture, whose fixed-size memory layers keep long visual inputs affordable.

- From pixels to vectors
- Three memory layers, then one attention layer
- Training examples that need the image
- The open model and the hosted product

## Figures

- **One context for images and text** (`flow`): Simplified path from image to language model. It omits image tiling, position encoding and the encoder’s internal layers. Understanding an image takes more than the words in it.
- **Long inputs, two kinds of memory cost** (`state`): Made-up sizes comparing a recurrent state with a growing attention history; not a Qwen3.5 memory calculation. Long image and document inputs make the cost of keeping context a practical concern.

## Accuracy and scope

Release date: 2026-02-16. Qwen released the flagship weights on February 16, 2026. Smaller models followed later.

- The diagram is conceptual; it does not specify Qwen3.5’s exact image tokenization.
- The perception tests described here are suggestions for teaching, not published measurements.

The page includes primary source links next to claims and a source list. Research cutoff: 2026-10-03. Quantitative widgets are explicitly simplified teaching models, not executed model inference or performance benchmarks.

## Maintenance

Edit `lesson.json`; run `pnpm author` from the collection directory to refresh committed static files. Run `pnpm test` there. The optional authoring utility is not required to view or deploy this page.
