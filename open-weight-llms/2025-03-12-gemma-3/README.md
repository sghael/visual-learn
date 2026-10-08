# Gemma 3 · 1B, 4B, 12B, 27B

How mostly local attention made room for images and long inputs

Open `index.html` directly in a browser, or follow this folder from the collection index. All runtime CSS and JavaScript are local; there is no build step. Google Fonts are optional and have system fallbacks.

## What the tutorial teaches

Gemma 3 made five of every six attention layers look only at the last 1,024 tokens, which kept the cache small enough for 128,000-token inputs and images.

- Where a running model’s memory goes
- Most layers look only nearby
- How distant information still arrives
- How images become tokens
- Fewer bits for the weights

## Figures

- **Which earlier positions can this layer see?** (`attention`): Simplified attention pattern for one layer with 16 tokens. Local and Full stand for Gemma 3’s two layer types. “Toy hybrid” is a generic pattern, not Gemma 3’s five-local, one-global schedule. Fewer global layers means fewer layers storing the whole growing history.
- **Keep the detail before asking the question** (`flow`): Simplified image path for Gemma 3. The number of crops depends on the image, so the token count per image is not fixed. A model can reason only about the visual detail that survives encoding.
- **Weight precision is a separate setting** (`precision`): Simplified storage for 12 billion weights, not an actual Gemma 3 file size. It excludes scale factors, mixed precision, the KV cache, the vision encoder, working memory and any effect on accuracy. Fewer bits shrink the weights; the attention layout controls a different part of memory.

## Accuracy and scope

Release date: 2025-03-12. Google released the Gemma 3 weights on March 12, 2025. The 1B model differs from the larger sizes, as described below.

- The 1B model reads text only and has a shorter context than the larger models.
- Neither figure measures Gemma 3’s speed or whether it fits on a particular device.
- A context limit does not guarantee reliable retrieval from every position.

The page includes primary source links next to claims and a source list. Research cutoff: 2026-10-03. Quantitative widgets are explicitly simplified teaching models, not executed model inference or performance benchmarks.

## Maintenance

Edit `lesson.json`; run `pnpm author` from the collection directory to refresh committed static files. Run `pnpm test` there. The optional authoring utility is not required to view or deploy this page.
