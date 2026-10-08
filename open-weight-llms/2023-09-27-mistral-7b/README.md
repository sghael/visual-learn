# Mistral 7B

Shrinking a conversation’s memory by sharing it across heads and limiting how far back each layer looks

Open `index.html` directly in a browser, or follow this folder from the collection index. All runtime CSS and JavaScript are local; there is no build step. Google Fonts are optional and have system fallbacks.

## What the tutorial teaches

Mistral 7B shared stored keys and values across attention heads and let each layer look back only 4,096 tokens, cutting the memory a long conversation needs.

- What the KV cache stores
- Sharing keys and values across heads
- Looking back only 4,096 tokens
- The two savings multiply
- What the release made possible

## Figures

- **Two separate multipliers: length and key–value heads** (`kv`): Simplified KV-cache size for one conversation, with each number stored in two bytes (bf16), using Mistral 7B v0.1’s layer and head dimensions. It ignores the sliding window, the weights, batching and other runtime memory. Sharing shrinks the memory for each token. It does not limit how many tokens are kept.
- **Watch the window move** (`attention`): Simplified attention pattern for one layer, with 16 positions and a window of 4 standing in for Mistral’s 4,096. The “Toy hybrid” option is a generic teaching pattern, not Mistral’s design. Each layer looks only nearby. Older information arrives indirectly, through earlier layers.

## Accuracy and scope

Release date: 2023-09-27. Mistral AI released v0.1 on September 27, 2023, and published the technical paper in October. This page describes v0.1.

- The figures are simplified calculations, not measurements of the model.
- The cache figure ignores the sliding window so that the effect of head sharing can be seen on its own.
- Passing information through many layers does not guarantee that the model can retrieve a distant detail accurately.

The page includes primary source links next to claims and a source list. Research cutoff: 2026-10-03. Quantitative widgets are explicitly simplified teaching models, not executed model inference or performance benchmarks.

## Maintenance

Edit `lesson.json`; run `pnpm author` from the collection directory to refresh committed static files. Run `pnpm test` there. The optional authoring utility is not required to view or deploy this page.
