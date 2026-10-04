# Tülu 3

How a checkable answer becomes a training signal

Open `index.html` directly in a browser, or follow this folder from the collection index. All runtime CSS and JavaScript are local; there is no build step. Google Fonts are optional and have system fallbacks.

## What the tutorial teaches

Explore verifiable rewards, checker failures, and the stages of an open post-training recipe.

- Different stages teach different behavior
- A reward is only as good as its checker
- Why reward can change future attempts
- An open recipe makes the comparison inspectable

## Figures

- **From imitation to preferences to checked outcomes** (`flow`): Simplified view of the published Tülu 3 post-training recipe. The recipe changes both the data and the signal used to update the weights.
- **Inspect the reward, not just the answer** (`verifier`): Simplified checker demonstration. These string rules are not the production Tülu verifier. An exact reward can still be an incomplete description of success.

## Accuracy and scope

Release date: 2024-11-21. Date of the initial official model and recipe release.

- The checker is a pedagogical example, not a copy of the released verifier.
- RLVR is a source of rewards, not a synonym for one optimizer: Tülu 3 uses PPO.
- This lesson covers the original 8B/70B recipe; the later 405B extension is not the November release.

The page includes primary source links next to claims and a source list. Research cutoff: 2026-10-03. Quantitative widgets are explicitly simplified teaching models, not executed model inference or performance benchmarks.

## Maintenance

Edit `lesson.json`; run `pnpm author` from the collection directory to refresh committed static files. Run `pnpm test` there. The optional authoring utility is not required to view or deploy this page.
