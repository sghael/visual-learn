# Qwen2.5

How the choice of training data became the main design decision

Open `index.html` directly in a browser, or follow this folder from the collection index. All runtime CSS and JavaScript are local; there is no build step. Google Fonts are optional and have system fallbacks.

## What the tutorial teaches

Qwen2.5 kept a familiar dense transformer and improved it mainly by growing and rebalancing its training data, and came in seven sizes for different budgets.

- Designing the training corpus
- Trading model size for training data
- The memory a conversation needs
- Comparing like with like

## Figures

- **A corpus is built from a series of decisions** (`flow`): Simplified outline of a corpus pipeline. The Qwen team’s full implementation has more steps. More tokens and a better mixture are separate improvements.
- **Spend the same training budget differently** (`scaling`): Simplified 6ND budget with a made-up starting point. These are not Qwen2.5’s training settings, and the figure predicts no benchmark score. A smaller model can be trained on more tokens and still cost less for each generated token.
- **Count only the keys and values** (`kv`): Simplified KV-cache size for one conversation in bf16, using Qwen2.5-7B’s dimensions. Weights and other runtime memory are excluded. Better training data does not shrink the memory a long conversation needs.

## Accuracy and scope

Release date: 2024-09-19. Alibaba’s Qwen team released the weights on September 19, 2024. The technical report followed in December.

- The training-budget figure approximates the main matrix work of a dense model. It ignores attention’s dependence on sequence length and is not a fitted scaling law.
- Qwen2.5-Turbo, which offered a one-million-token context, was available only through an API. That context length does not apply to the September open-weight models.

The page includes primary source links next to claims and a source list. Research cutoff: 2026-10-03. Quantitative widgets are explicitly simplified teaching models, not executed model inference or performance benchmarks.

## Maintenance

Edit `lesson.json`; run `pnpm author` from the collection directory to refresh committed static files. Run `pnpm test` there. The optional authoring utility is not required to view or deploy this page.
