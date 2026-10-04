# gpt-oss-20b and gpt-oss-120b

How sparse weights, low precision, and a conversation protocol make reasoning deployable

Open `index.html` directly in a browser, or follow this folder from the collection index. All runtime CSS and JavaScript are local; there is no build step. Google Fonts are optional and have system fallbacks.

## What the tutorial teaches

Separate storage, active computation, and reasoning effort in OpenAI’s open-weight models.

- Stored weights and active weights are different
- Store expert weights in fewer bits
- A tool call needs a boundary
- More reasoning changes the inference budget

## Figures

- **Visit a few experts; keep the whole bank** (`moe`): Simplified eight-expert router. Actual gpt-oss counts are given above; the displayed routes are illustrative. Sparse activation reduces selected work, not the need to store the expert bank.
- **A lower bound for storing 21 billion numbers** (`precision`): Simplified uniform storage calculation. The release is mixed precision, and MXFP4 includes scale metadata. Fewer bits and fewer active experts save different resources.
- **One model turn can contain a tool round trip** (`flow`): Simplified host/model protocol. No tool is executed by this tutorial. Tool use belongs to a model-and-runtime loop.

## Accuracy and scope

Release date: 2025-08-05. Official public launch date.

- The two sizes share this tutorial because the mechanism is shared; their parameter and expert counts differ.
- The four-bit chart is an ideal lower bound, not a claim that the whole model uses four bits or fits in a particular device.
- This tutorial does not infer an undisclosed full training-data recipe from the released weights.

The page includes primary source links next to claims and a source list. Research cutoff: 2026-10-03. Quantitative widgets are explicitly simplified teaching models, not executed model inference or performance benchmarks.

## Maintenance

Edit `lesson.json`; run `pnpm author` from the collection directory to refresh committed static files. Run `pnpm test` there. The optional authoring utility is not required to view or deploy this page.
