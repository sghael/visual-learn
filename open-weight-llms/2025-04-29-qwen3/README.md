# Qwen3

How one model learns when to deliberate

Open `index.html` directly in a browser, or follow this folder from the collection index. All runtime CSS and JavaScript are local; there is no build step. Google Fonts are optional and have system fallbacks.

## What the tutorial teaches

Qwen3 put a switchable thinking mode into one model, trained it in four stages for the largest models, and distilled it into the smaller ones.

- Thinking costs tokens
- Four training stages, then distillation
- What a teacher’s probabilities add
- Switching modes in practice

## Figures

- **One model, an optional detour** (`flow`): Simplified flow for the original Qwen3 release, in which one model handles both modes. Thinking mode spends extra tokens before the answer.
- **See what a soft target contains** (`distill`): Made-up scores for four tokens. Temperature is a teaching control here, not a reconstruction of Qwen3’s training. A teacher’s distribution shows how the alternatives compare, not only which one is best.

## Accuracy and scope

Release date: 2025-04-29. Qwen announced the models on April 29, 2025; the weight uploads are dated April 28 UTC. The page uses the announcement date. The technical report followed on May 14.

- The written reasoning is not a complete or faithful account of the model’s internal computation.
- The softmax figure shows what information distillation carries. It does not train anything or predict a student’s quality.

The page includes primary source links next to claims and a source list. Research cutoff: 2026-10-03. Quantitative widgets are explicitly simplified teaching models, not executed model inference or performance benchmarks.

## Maintenance

Edit `lesson.json`; run `pnpm author` from the collection directory to refresh committed static files. Run `pnpm test` there. The optional authoring utility is not required to view or deploy this page.
