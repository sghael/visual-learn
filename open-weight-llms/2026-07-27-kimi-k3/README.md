# Kimi K3

Changing how information moves along the sequence and up through the layers

Open `index.html` directly in a browser, or follow this folder from the collection index. All runtime CSS and JavaScript are local; there is no build step. Google Fonts are optional and have system fallbacks.

## What the tutorial teaches

Kimi K3 pairs a recurrent memory that forgets different features at different rates with attention over earlier layers’ outputs, in a 2.8-trillion-parameter mixture of experts.

- Two directions of information flow
- Forgetting different features at different rates
- Letting later layers choose earlier outputs
- Experts and low precision

## Figures

- **A fixed state beside a growing history** (`state`): Made-up sizes, not the dimensions of KDA, MLA or K3’s cache. A hybrid can combine a compact recurrent summary with direct retrieval.
- **Choose information across depth** (`flow`): Simplified idea of Attention Residuals. It omits block boundaries and the exact learned scoring. Choosing across layers and retrieving across tokens solve different problems.

## Accuracy and scope

Release date: 2026-07-27. Moonshot AI announced the hosted model on July 16, 2026, and released the weights on July 27.

- KDA and Attention Residuals come from earlier component papers; the K3 report specifies the changes and settings used in this release.
- The weighted-sum example isolates mixing across layers; it is not a K3 layer.

The page includes primary source links next to claims and a source list. Research cutoff: 2026-10-03. Quantitative widgets are explicitly simplified teaching models, not executed model inference or performance benchmarks.

## Maintenance

Edit `lesson.json`; run `pnpm author` from the collection directory to refresh committed static files. Run `pnpm test` there. The optional authoring utility is not required to view or deploy this page.
