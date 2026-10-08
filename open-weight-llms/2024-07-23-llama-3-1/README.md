# Llama 3.1 · 8B, 70B, 405B

How data, training scale and a longer context improved a conventional design

Open `index.html` directly in a browser, or follow this folder from the collection index. All runtime CSS and JavaScript are local; there is no build step. Google Fonts are optional and have system fallbacks.

## What the tutorial teaches

Llama 3.1 kept a conventional dense transformer and improved it through data curation, training scale, staged extension of the context to 128,000 tokens and preference training.

- A conventional design at a larger scale
- Spending a fixed training budget
- Choosing which tokens to train on
- Making 128,000 tokens usable
- Choosing between good answers

## Figures

- **Spend a fixed training budget** (`scaling`): Simplified training-compute trade for a dense model, using 6ND. The starting point of 7 billion parameters and 1 trillion tokens is made up; it is not a Llama 3.1 run. The figure predicts neither accuracy nor speed. The same budget buys either more parameters or more training tokens.
- **A model recipe has several separate settings** (`flow`): Simplified outline of Llama 3.1’s development stages. The production pipeline had many more steps. Data, context length and behavior can each be changed without changing the architecture.

## Accuracy and scope

Release date: 2024-07-23. Meta released the 3.1 weights on July 23, 2024. The earlier Llama 3 8B and 70B models appeared on April 18, 2024.

- The compute figure is a rough estimate, not a fitted scaling law or a hardware measurement.
- The paper’s image and speech research models were not released as Llama 3.1 weights.
- The 128,000-token window covers prompt and reply together, and fitting text into it does not guarantee that the model uses every detail.

The page includes primary source links next to claims and a source list. Research cutoff: 2026-10-03. Quantitative widgets are explicitly simplified teaching models, not executed model inference or performance benchmarks.

## Maintenance

Edit `lesson.json`; run `pnpm author` from the collection directory to refresh committed static files. Run `pnpm test` there. The optional authoring utility is not required to view or deploy this page.
