# Llama 4 · Scout and Maverick

How sparse experts and image tokens share one model

Open `index.html` directly in a browser, or follow this folder from the collection index. All runtime CSS and JavaScript are local; there is no build step. Google Fonts are optional and have system fallbacks.

## What the tutorial teaches

Llama 4 was Meta’s first Llama release to use a mixture of experts and to train on images and text together from the start.

- What Meta released
- One shared expert plus one routed expert
- Why “17B active” does not mean 17B to load
- Images become vectors before the model reads them
- Test each part separately

## Figures

- **Follow one token through an expert layer** (`moe`): Simplified router with 8 routed experts and 1 shared expert. Maverick has 128 routed experts and uses one per token; the other settings show the general mechanism. Each token uses a small fraction of the experts, but every expert must still be stored.
- **Bring an image into the conversation** (`flow`): Simplified path for image and text input in Llama 4. It omits image tiling, projection layers and training details. With image and text tokens in one sequence, the model can relate what it sees to what it is asked.

## Accuracy and scope

Release date: 2025-04-05. Meta released the Scout and Maverick weights on April 5, 2025. Behemoth was previewed but not released.

- Behemoth’s weights were not released.
- The routing figure is scaled down and does not name experts by topic.
- Dividing total by active parameters does not give a speed-up.
- An advertised context length does not mean the model uses every token reliably.

The page includes primary source links next to claims and a source list. Research cutoff: 2026-10-03. Quantitative widgets are explicitly simplified teaching models, not executed model inference or performance benchmarks.

## Maintenance

Edit `lesson.json`; run `pnpm author` from the collection directory to refresh committed static files. Run `pnpm test` there. The optional authoring utility is not required to view or deploy this page.
