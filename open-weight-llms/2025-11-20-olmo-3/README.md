# Olmo 3

How releasing the training path turns a model into an experiment

Open `index.html` directly in a browser, or follow this folder from the collection index. All runtime CSS and JavaScript are local; there is no build step. Google Fonts are optional and have system fallbacks.

## What the tutorial teaches

Ai2 published Olmo 3’s data, code and checkpoints from every training stage, so researchers can change one step and measure its effect on the final model.

- A model is the end of a series of choices
- Changing the training mix
- Several models from one base
- Publishing is the first step toward reproducing

## Figures

- **Find a point where an experiment can branch** (`flow`): Simplified map of Olmo 3’s published training flow, with the branches condensed. An intermediate checkpoint lets an experiment change training earlier than the final model.
- **Keep the token budget fixed; change the mix** (`mixture`): Made-up budget of 100 billion tokens. It is not Olmo 3’s actual mixture or a prediction of quality. A published mixture makes a training change precise enough to test.

## Accuracy and scope

Release date: 2025-11-20. Ai2 released Olmo 3 on November 20, 2025. The technical report followed on arXiv in December.

- The mixture figure is a budgeting exercise, not a measured relationship between data and quality.
- Open data and code do not make a full reproduction cheap or numerically identical.
- This page covers the November Olmo 3 release, not the December Olmo 3.1 update.

The page includes primary source links next to claims and a source list. Research cutoff: 2026-10-03. Quantitative widgets are explicitly simplified teaching models, not executed model inference or performance benchmarks.

## Maintenance

Edit `lesson.json`; run `pnpm author` from the collection directory to refresh committed static files. Run `pnpm test` there. The optional authoring utility is not required to view or deploy this page.
