# Mixtral 8×7B

A large library of parameters, a small working selection

Open `index.html` directly in a browser, or follow this folder from the collection index. All runtime CSS and JavaScript are local; there is no build step. Google Fonts are optional and have system fallbacks.

## What the tutorial teaches

A router chooses two feed-forward experts for each token at each layer, separating total capacity from active computation.

- An expert is a component, not a complete chatbot
- Choose two transformations and combine them
- Unused this time does not mean absent from memory
- The word “expert” can mislead
- The larger Mixtral follows the same principle

## Figures

- **Follow a token through the expert choices** (`moe`): Simplified token router with eight available experts. The release uses two selected experts; other active counts are counterfactual teaching settings. Capacity can grow without applying every expert to every token.
- **Inventory and per-token work use the same unit** (`bars`): Illustrative comparison of the released model’s reported parameter counts. Parameter counts are not measured latency. The inactive experts still contribute to the model’s storage requirement.

## Accuracy and scope

Release date: 2023-12-08. Public torrent weights appeared December 8, 2023; Mistral’s formal announcement is dated December 11. The technical report followed January 8, 2024.

- Expert labels in this tutorial are numerical; no topic-specific specialization is assumed.
- Active/total parameter ratios are not latency or throughput estimates.
- The date records the initial torrent release; the formal official blog is dated December 11.

The page includes primary source links next to claims and a source list. Research cutoff: 2026-10-03. Quantitative widgets are explicitly simplified teaching models, not executed model inference or performance benchmarks.

## Maintenance

Edit `lesson.json`; run `pnpm author` from the collection directory to refresh committed static files. Run `pnpm test` there. The optional authoring utility is not required to view or deploy this page.
