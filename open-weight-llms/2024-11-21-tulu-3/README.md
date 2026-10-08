# Tülu 3

How a checkable answer becomes a training signal

Open `index.html` directly in a browser, or follow this folder from the collection index. All runtime CSS and JavaScript are local; there is no build step. Google Fonts are optional and have system fallbacks.

## What the tutorial teaches

Tülu 3 published a complete post-training recipe that ends with reinforcement learning on rewards from automatic checks, such as whether a math answer is correct.

- Three stages of post-training
- A reward is only as good as its checker
- How a reward changes future answers
- What an open recipe makes possible

## Figures

- **From imitation to preferences to checked outcomes** (`flow`): Simplified view of the published Tülu 3 post-training recipe. Each stage changes both the training data and the kind of signal that updates the weights.
- **Inspect the reward, not just the answer** (`verifier`): Simplified checker. These rules are not Tülu 3’s actual verifier. A reward can be exact and still miss part of what makes an answer good.

## Accuracy and scope

Release date: 2024-11-21. Ai2 released the first models and the recipe on November 21, 2024.

- The checker in the figure is a teaching example, not the released verifier.
- RLVR describes where the reward comes from, not which algorithm uses it; Tülu 3 used PPO.
- This page covers the original 8B and 70B models from November 2024. The 405B model came later.

The page includes primary source links next to claims and a source list. Research cutoff: 2026-10-03. Quantitative widgets are explicitly simplified teaching models, not executed model inference or performance benchmarks.

## Maintenance

Edit `lesson.json`; run `pnpm author` from the collection directory to refresh committed static files. Run `pnpm test` there. The optional authoring utility is not required to view or deploy this page.
