# Qwen2.5

How a training corpus becomes a design choice

Open `index.html` directly in a browser, or follow this folder from the collection index. All runtime CSS and JavaScript are local; there is no build step. Google Fonts are optional and have system fallbacks.

## What the tutorial teaches

Qwen2.5 shows how data selection, longer training, and deployment memory shape a dense model.

- A pile of text is not yet a curriculum
- Training cost and serving cost pull in different directions
- Conversation length creates a separate memory bill
- Read the release as a bundle of choices

## Figures

- **The corpus is a sequence of decisions** (`flow`): Simplified data pipeline, not the complete Qwen implementation. More tokens and a better mixture are separate interventions.
- **Spend the same toy training budget** (`scaling`): Illustrative 6ND budget. These settings are not the Qwen2.5 training recipe and predict no benchmark score. A smaller model can receive more training while costing less per generated token.
- **Count only the keys and values** (`kv`): Simplified bf16, single-sequence KV estimate using the 7B model dimensions. Excludes weights, activations, allocator overhead and sharding. Data efficiency during training does not eliminate context memory during inference.

## Accuracy and scope

Release date: 2024-09-19. Official public launch date; the technical report appeared in December 2024.

- The training-budget widget approximates dense weight-matrix work; it omits attention’s dependence on sequence length and is not a scaling-law fit.
- Qwen2.5-Turbo was an API model; its million-token configuration is not attributed to the September open checkpoints.

The page includes primary source links next to claims and a source list. Research cutoff: 2026-10-03. Quantitative widgets are explicitly simplified teaching models, not executed model inference or performance benchmarks.

## Maintenance

Edit `lesson.json`; run `pnpm author` from the collection directory to refresh committed static files. Run `pnpm test` there. The optional authoring utility is not required to view or deploy this page.
