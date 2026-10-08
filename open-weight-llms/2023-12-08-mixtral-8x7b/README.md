# Mixtral 8×7B

A model with many parameters that uses only a few of them for each token

Open `index.html` directly in a browser, or follow this folder from the collection index. All runtime CSS and JavaScript are local; there is no build step. Google Fonts are optional and have system fallbacks.

## What the tutorial teaches

In each layer, a router sends every token to two of eight feed-forward networks, so Mixtral stores 46.7 billion parameters but uses about 12.9 billion for each token.

- What an “expert” is
- How a token picks two experts
- All experts must stay in memory
- Experts do not specialize by subject
- The larger Mixtral uses the same design

## Figures

- **Follow a token through the router** (`moe`): Simplified router with eight experts. Mixtral uses two per token; the other settings show what would change. The model can store many experts without running all of them on every token.
- **Stored parameters versus parameters used per token** (`bars`): Mixtral’s reported parameter counts. Parameter counts are not measured speed. Experts that a token skips still take up memory.

## Accuracy and scope

Release date: 2023-12-08. Mistral AI posted the weights as a torrent on December 8, 2023, announced the model on December 11, and published the technical report on January 8, 2024.

- Experts are numbered, not named; the paper found no evidence of subject specialization.
- The ratio of active to total parameters is not an estimate of speed.
- The release date is the torrent posting; the official announcement is dated December 11.

The page includes primary source links next to claims and a source list. Research cutoff: 2026-10-03. Quantitative widgets are explicitly simplified teaching models, not executed model inference or performance benchmarks.

## Maintenance

Edit `lesson.json`; run `pnpm author` from the collection directory to refresh committed static files. Run `pnpm test` there. The optional authoring utility is not required to view or deploy this page.
