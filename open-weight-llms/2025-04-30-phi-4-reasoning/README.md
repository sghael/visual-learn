# Phi-4-reasoning · reasoning-plus

How teachable examples and verifiable rewards shape a compact reasoner

Open `index.html` directly in a browser, or follow this folder from the collection index. All runtime CSS and JavaScript are local; there is no build step. Google Fonts are optional and have system fallbacks.

## What the tutorial teaches

Phi-4-reasoning adds carefully selected demonstrations to Phi-4, then reasoning-plus learns from group-relative rewards.

- Begin with an already trained model
- Choose problems near the model’s boundary
- Several attempts create a local comparison
- Reward design determines what is being practiced
- More generated work has a cost

## Figures

- **A curriculum before a reward loop** (`flow`): Simplified Phi-4-reasoning development sequence. The plus checkpoint adds the final reinforcement-learning stage. Demonstrations establish useful behavior before outcome-based refinement.
- **Compute a group-relative learning signal** (`grpo`): Simplified pedagogical GRPO signal with illustrative binary rewards. Uses population standard deviation and returns zero for an equal-reward group. Omits clipping, KL, entropy, token loss and Phi’s actual length/format/repetition reward. A relative signal needs differences among sampled outcomes.

## Accuracy and scope

Release date: 2025-04-30. Public reasoning and reasoning-plus weights launched April 30, 2025. The underlying Phi-4 was a prior release.

- The widget is only a simplified advantage calculation, not Phi’s full training loss.
- Sequence imitation is not the same as token-distribution distillation.
- Longer reasoning is not guaranteed to improve an individual answer.

The page includes primary source links next to claims and a source list. Research cutoff: 2026-10-03. Quantitative widgets are explicitly simplified teaching models, not executed model inference or performance benchmarks.

## Maintenance

Edit `lesson.json`; run `pnpm author` from the collection directory to refresh committed static files. Run `pnpm test` there. The optional authoring utility is not required to view or deploy this page.
