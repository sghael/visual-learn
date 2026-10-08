# DeepSeek-R1

How rewarding correct final answers trained long step-by-step reasoning, and how smaller models learned from it

Open `index.html` directly in a browser, or follow this folder from the collection index. All runtime CSS and JavaScript are local; there is no build step. Google Fonts are optional and have system fallbacks.

## What the tutorial teaches

DeepSeek-R1 used reinforcement learning on automatically checked answers to train long step-by-step reasoning, then taught smaller models to imitate its solutions.

- Checking an answer is easier than writing one
- Comparing attempts at the same problem
- What the update changes
- How the final R1 was trained
- Smaller models trained on R1’s solutions

## Figures

- **A reward means something only next to the alternatives** (`grpo`): Simplified GRPO advantage, using the population standard deviation. It omits clipping, the penalty for drifting from a reference model, per-token weighting and R1’s training settings. The group supplies its own baseline. When every reward is the same, that group gives nothing to learn from.
- **R1 was trained in several stages** (`flow`): Simplified R1 training recipe. The curated examples fine-tune a fresh copy of V3-Base, so this is not one continuous chain of updates. R1-Zero was a separate experiment. The final model combines imitation of examples, checked outcomes and further tuning for general use.

## Accuracy and scope

Release date: 2025-01-20. DeepSeek released R1, R1-Zero and the distilled models on January 20, 2025. The first version of the paper followed on January 22.

- R1-Zero starts from a pretrained model; the final R1 also uses supervised fine-tuning stages.
- The GRPO figure shows only the relative advantage, not the full training objective.
- The distilled models carry obligations from their Qwen or Llama base licenses.

The page includes primary source links next to claims and a source list. Research cutoff: 2026-10-03. Quantitative widgets are explicitly simplified teaching models, not executed model inference or performance benchmarks.

## Maintenance

Edit `lesson.json`; run `pnpm author` from the collection directory to refresh committed static files. Run `pnpm test` there. The optional authoring utility is not required to view or deploy this page.
