# gpt-oss-20b and gpt-oss-120b

How expert routing, attention, weight precision, and tool handoffs divide the work

Open `index.html` directly in a browser, or follow this folder from the collection index. All runtime CSS and JavaScript are local; there is no build step. Google Fonts are optional and have system fallbacks.

## What the tutorial teaches

Separate storage, active computation, and reasoning effort in OpenAI’s open-weight models.

- Stored weights and active weights are different
- Attention can use less history—or less weight
- Store expert weights in fewer bits
- A tool call needs a boundary
- More reasoning changes the inference budget

## Figures

- **Visit a few experts; keep the whole bank** (`moe`): Simplified eight-expert router with four experts initially selected. Real gpt-oss selects four from 32 or 128; the displayed routes are illustrative. Sparse activation reduces selected work, not the need to store the expert bank.
- **Leave some attention weight unused** (`sink`): Simplified single-head softmax with invented scalar values. The slider varies a hypothetical learned sink score; it is not a user setting in gpt-oss. Real attention operates on vectors. A zero-contribution option lets attention reduce its total weight on the visible tokens.
- **A lower bound for storing 21 billion numbers** (`precision`): Simplified uniform storage calculation. The release is mixed precision, and MXFP4 includes scale metadata. Fewer bits and fewer active experts save different resources.
- **One model turn can contain a tool round trip** (`flow`): Simplified host/model protocol. No tool is executed by this tutorial. A tool reply returns control to the model; it does not itself finish the assistant turn.

## Accuracy and scope

Release date: 2025-08-05. Official public launch date.

- The two sizes share this tutorial because the mechanism is shared; their parameter and expert counts differ.
- The four-bit chart is an ideal lower bound, not a claim that the whole model uses four bits or fits in a particular device.
- This tutorial does not infer an undisclosed full training-data recipe from the released weights.
- The attention sink changes normalized value mixing; it is not an extra cached token, a context compressor, or a guarantee of faithful reasoning.
- Reported checkpoint sizes, ideal packed-weight bytes, and peak serving memory describe different quantities.

The page includes primary source links next to claims and a source list. Research cutoff: 2026-10-03. Quantitative widgets are explicitly simplified teaching models, not executed model inference or performance benchmarks.

## Maintenance

Edit `lesson.json`; run `pnpm author` from the collection directory to refresh committed static files. Run `pnpm test` there. The optional authoring utility is not required to view or deploy this page.
