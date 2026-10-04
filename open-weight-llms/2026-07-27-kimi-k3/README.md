# Kimi K3

How memory changes across sequence length and model depth

Open `index.html` directly in a browser, or follow this folder from the collection index. All runtime CSS and JavaScript are local; there is no build step. Google Fonts are optional and have system fallbacks.

## What the tutorial teaches

Kimi K3 combines channel-controlled recurrent memory with attention over earlier layer representations.

- Sequence length and depth are different axes
- Forgetting can be selective within a memory
- Later layers can select earlier representations
- A larger model is also a systems design

## Figures

- **Bounded state beside a growing history** (`state`): Illustrative scalar counts; not KDA, MLA or K3 cache dimensions. A hybrid can combine compact recurrent summaries with direct retrieval.
- **Select information across depth** (`flow`): Simplified Attention Residuals concept. Omits block boundaries and the exact learned scoring parameterization. Depth-wise selection and token-wise retrieval solve different information-flow problems.

## Accuracy and scope

Release date: 2026-07-27. Weights released July 27, 2026; July 16 was the hosted-model announcement.

- The K3 full-report link could not be retrieved during research; architecture claims use the official model card and read component papers.
- The scalar weighted-sum example isolates depth mixing and is not a K3 layer implementation.

The page includes primary source links next to claims and a source list. Research cutoff: 2026-10-03. Quantitative widgets are explicitly simplified teaching models, not executed model inference or performance benchmarks.

## Maintenance

Edit `lesson.json`; run `pnpm author` from the collection directory to refresh committed static files. Run `pnpm test` there. The optional authoring utility is not required to view or deploy this page.
