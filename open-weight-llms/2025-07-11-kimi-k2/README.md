# Kimi K2

Training a trillion-parameter model without instability, and teaching it to use tools

Open `index.html` directly in a browser, or follow this folder from the collection index. All runtime CSS and JavaScript are local; there is no build step. Google Fonts are optional and have system fallbacks.

## What the tutorial teaches

Kimi K2 trained a trillion-parameter mixture of experts stably with the MuonClip optimizer, and learned tool use from large numbers of generated practice sessions.

- Many experts, few used at a time
- Keeping a giant training run stable
- Practicing tool use
- The model and the software around it

## Figures

- **Stored experts versus experts used** (`moe`): Simplified router with eight experts. K2 has 384 routed experts per layer, uses 8 per token, and adds 1 shared expert. Using few experts per token reduces computation, not storage.
- **Keep the next update in range** (`flow`): Simplified training step with QK-Clip. The report specifies per-head details and special handling for MLA. At this scale, keeping training stable is part of what makes an optimizer useful.

## Accuracy and scope

Release date: 2025-07-11. Moonshot AI released the Base and Instruct weights on July 11, 2025. The technical report followed later.

- Muon and mixtures of experts predate K2. The release’s own contributions are QK-Clip and making the combination work at this scale.
- The eight-expert router and the clipping arithmetic are teaching examples.

The page includes primary source links next to claims and a source list. Research cutoff: 2026-10-03. Quantitative widgets are explicitly simplified teaching models, not executed model inference or performance benchmarks.

## Maintenance

Edit `lesson.json`; run `pnpm author` from the collection directory to refresh committed static files. Run `pnpm test` there. The optional authoring utility is not required to view or deploy this page.
