# Phi-4-reasoning · reasoning-plus

How well-chosen worked examples and a short round of rewards trained a small reasoning model

Open `index.html` directly in a browser, or follow this folder from the collection index. All runtime CSS and JavaScript are local; there is no build step. Google Fonts are optional and have system fallbacks.

## What the tutorial teaches

Phi-4-reasoning fine-tuned the 14B Phi-4 on long worked solutions to carefully chosen problems; reasoning-plus added reinforcement learning on checked answers.

- Starting from Phi-4
- Choosing problems at the edge of ability
- Comparing several attempts
- The reward defines what gets practiced
- Longer answers cost time

## Figures

- **Examples first, then rewards** (`flow`): Simplified development sequence. Only the plus model has the final reinforcement-learning stage. Demonstrations establish good habits before rewards refine them.
- **Compute a group-relative learning signal** (`grpo`): Simplified GRPO advantage with made-up rewards of 0 or 1, using the population standard deviation; a group with equal rewards gets zero. It omits clipping, the drift penalty, per-token weighting and Phi’s length, format and repetition terms. A relative signal needs the attempts to differ.

## Accuracy and scope

Release date: 2025-04-30. Microsoft released the reasoning and reasoning-plus weights on April 30, 2025. The underlying Phi-4 model was released earlier.

- The figure computes only a simplified advantage, not Phi’s full training objective.
- Imitating written solutions is different from matching a teacher’s token probabilities.
- Longer reasoning does not guarantee a better answer to any particular question.

The page includes primary source links next to claims and a source list. Research cutoff: 2026-10-03. Quantitative widgets are explicitly simplified teaching models, not executed model inference or performance benchmarks.

## Maintenance

Edit `lesson.json`; run `pnpm author` from the collection directory to refresh committed static files. Run `pnpm test` there. The optional authoring utility is not required to view or deploy this page.
