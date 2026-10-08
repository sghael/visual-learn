# Phi-4-reasoning-vision-15B

How better visual evidence and selective reasoning solve different problems

Open `index.html` directly in a browser, or follow this folder from the collection index. All runtime CSS and JavaScript are local; there is no build step. Google Fonts are optional and have system fallbacks.

## What the tutorial teaches

Phi-4-reasoning-vision pairs a vision encoder that keeps fine detail with a language model trained to choose between answering directly and reasoning at length.

- Read the image before solving the problem
- More image tokens keep more detail
- Connecting the encoder to the language model
- Reasoning only when it helps
- Finding which step failed

## Figures

- **Count image patches and the pairs between them** (`patches`): Abstract image grid. The patch count is columns times rows, and full attention compares every pair of patches. This is not Phi’s exact tokenizer, and a patch is a learned representation, not an average of its pixels. A larger image-token budget makes attention work grow faster than the token count.
- **From image to answer** (`flow`): Simplified data path. It omits image preprocessing, layer details and training stages. The choice of response mode is learned and can be wrong. A trained projector connects what the encoder sees to what the language model reasons about.

## Accuracy and scope

Release date: 2026-03-04. The model card and Microsoft’s research post both give March 4, 2026 as the release date of the weights.

- The patch grid illustrates encoder cost in abstract units; it is not Phi’s tokenizer or a runtime simulator.
- Cutting an image into patches does not reduce each patch to its average color.
- The model’s learned choice between direct and reasoning answers is imperfect.
- Written reasoning is not guaranteed to explain the model’s internal computation.

The page includes primary source links next to claims and a source list. Research cutoff: 2026-10-03. Quantitative widgets are explicitly simplified teaching models, not executed model inference or performance benchmarks.

## Maintenance

Edit `lesson.json`; run `pnpm author` from the collection directory to refresh committed static files. Run `pnpm test` there. The optional authoring utility is not required to view or deploy this page.
