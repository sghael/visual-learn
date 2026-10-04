# DeepSeek-R1

Reward successful attempts, then teach a smaller model to imitate them

Open `index.html` directly in a browser, or follow this folder from the collection index. All runtime CSS and JavaScript are local; there is no build step. Google Fonts are optional and have system fallbacks.

## What the tutorial teaches

Group-relative reinforcement learning turns verifiable outcomes into a training signal for extended reasoning.

- A checker can be cheaper than a teacher
- Compare attempts at the same problem
- A reward changes probabilities, not a line of code
- R1 adds a curriculum around reinforcement learning
- A smaller model can learn from generated solutions

## Figures

- **A reward becomes meaningful relative to alternatives** (`grpo`): Simplified pedagogical GRPO signal using population standard deviation. It omits clipping, KL regularization, token-level loss and release-specific training details. A group supplies both examples and a baseline; identical rewards provide no relative ranking.
- **The released R1 recipe has several stages** (`flow`): Simplified R1 post-training pipeline. R1-Zero is a separate pure-RL post-training experiment from a pretrained base. The final checkpoint combines examples, outcome feedback and further alignment.

## Accuracy and scope

Release date: 2025-01-20. Public R1, R1-Zero and distilled checkpoint release on January 20, 2025; the first paper version followed on January 22.

- R1-Zero starts from a pretrained base; final R1 additionally uses supervised stages.
- The GRPO widget demonstrates relative advantage only, not the complete optimization objective.
- Distilled checkpoints inherit obligations from their Qwen or Llama base licenses.

The page includes primary source links next to claims and a source list. Research cutoff: 2026-10-03. Quantitative widgets are explicitly simplified teaching models, not executed model inference or performance benchmarks.

## Maintenance

Edit `lesson.json`; run `pnpm author` from the collection directory to refresh committed static files. Run `pnpm test` there. The optional authoring utility is not required to view or deploy this page.
