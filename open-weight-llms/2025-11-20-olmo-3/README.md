# Olmo 3

How releasing the training path turns a model into an experiment

Open `index.html` directly in a browser, or follow this folder from the collection index. All runtime CSS and JavaScript are local; there is no build step. Google Fonts are optional and have system fallbacks.

## What the tutorial teaches

Follow pretraining, midtraining, and post-training through a model family with public data and checkpoints.

- A model is the endpoint of many choices
- Changing the curriculum changes the experiment
- Reasoning and conversation can share a base
- Availability is the beginning of reproducibility

## Figures

- **Find a point where the experiment can branch** (`flow`): Simplified map of Olmo 3’s published model flow; branches are compressed into one overview. An intermediate checkpoint allows an intervention earlier than the final model.
- **Hold the token budget fixed; change the mix** (`mixture`): Illustrative 100B-token curriculum, not the measured Olmo mixture or a predicted quality curve. A transparent mixture makes the training change precise enough to test.

## Accuracy and scope

Release date: 2025-11-20. Official public launch date; the arXiv technical report followed in December.

- The mixture widget is an allocation exercise, not an empirical scaling law.
- Openness does not imply that a full training reproduction is inexpensive or numerically identical.
- This page distinguishes the November Olmo 3 release from December Olmo 3.1.

The page includes primary source links next to claims and a source list. Research cutoff: 2026-10-03. Quantitative widgets are explicitly simplified teaching models, not executed model inference or performance benchmarks.

## Maintenance

Edit `lesson.json`; run `pnpm author` from the collection directory to refresh committed static files. Run `pnpm test` there. The optional authoring utility is not required to view or deploy this page.
