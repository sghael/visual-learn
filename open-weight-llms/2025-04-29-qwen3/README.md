# Qwen3

How one model learns when to deliberate

Open `index.html` directly in a browser, or follow this folder from the collection index. All runtime CSS and JavaScript are local; there is no build step. Google Fonts are optional and have system fallbacks.

## What the tutorial teaches

Qwen3 combines explicit thinking control with a teacher-to-student training route.

- Extra tokens are extra computation
- Teach the behavior before transferring it
- A teacher can provide more than a winning token
- Mode control must survive the chat template

## Figures

- **A conditional path through the same model** (`flow`): Simplified inference sequence for the original hybrid-thinking Qwen3 release. Thinking mode spends additional generated tokens before the answer.
- **Inspect the information in a soft target** (`distill`): Illustrative token distribution. Temperature is a teaching control, not a reconstruction of Qwen3 training. The teacher can communicate relative alternatives as well as the top choice.

## Accuracy and scope

Release date: 2025-04-29. Official announcement date April 29, 2025; weight uploads are dated April 28 UTC. The folder follows the announcement calendar date. Technical report submitted May 14.

- Generated reasoning is not a faithful explanation of every internal computation.
- The softmax widget teaches distillation information; it does not implement training or predict student quality.

The page includes primary source links next to claims and a source list. Research cutoff: 2026-10-03. Quantitative widgets are explicitly simplified teaching models, not executed model inference or performance benchmarks.

## Maintenance

Edit `lesson.json`; run `pnpm author` from the collection directory to refresh committed static files. Run `pnpm test` there. The optional authoring utility is not required to view or deploy this page.
