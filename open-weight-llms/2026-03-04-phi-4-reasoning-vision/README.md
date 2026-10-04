# Phi-4-reasoning-vision-15B

How better visual evidence and selective reasoning solve different problems

Open `index.html` directly in a browser, or follow this folder from the collection index. All runtime CSS and JavaScript are local; there is no build step. Google Fonts are optional and have system fallbacks.

## What the tutorial teaches

Phi-4-reasoning-vision combines dynamic-resolution visual encoding with a learned choice between direct and extended responses.

- Read the marks before solving the problem
- More visual tokens preserve opportunities to inspect detail
- A projector connects vision to the language backbone
- Train the model to spend effort selectively
- When the answer is wrong, locate the failed step

## Figures

- **Count image patches and pairwise comparisons** (`patches`): Illustrative image-encoder grid in abstract units. Patch count is ceil(width/patch) × ceil(height/patch); dense noncausal self-attention has N² ordered pairs. This is not Phi’s exact tokenizer, decoder cost or image quality. Patching does not imply all within-patch detail is averaged away. A larger visual-token budget can make attention work grow faster than the token count.
- **From image evidence to a response** (`flow`): Simplified Phi-4-reasoning-vision data path. Omits image preprocessing, detailed layer structure and training stages. The final mode is learned, not an infallible difficulty detector. Visual representations and language reasoning are connected through a trained interface.

## Accuracy and scope

Release date: 2026-03-04. Model card and official research post both identify March 4, 2026 as the public-weight release.

- The patch grid is an abstract image-encoder cost illustration, not a faithful Phi tokenizer or runtime simulator.
- Patchification does not mean every patch is reduced to its average color.
- Learned selection between direct and reasoning modes is imperfect.
- Visible reasoning text is not a guaranteed explanation of the model’s internal computation.

The page includes primary source links next to claims and a source list. Research cutoff: 2026-10-03. Quantitative widgets are explicitly simplified teaching models, not executed model inference or performance benchmarks.

## Maintenance

Edit `lesson.json`; run `pnpm author` from the collection directory to refresh committed static files. Run `pnpm test` there. The optional authoring utility is not required to view or deploy this page.
