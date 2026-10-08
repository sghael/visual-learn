# gpt-oss-20b and gpt-oss-120b

How experts, attention, four-bit weights and a tool protocol each address a different cost

Open `index.html` directly in a browser, or follow this folder from the collection index. All runtime CSS and JavaScript are local; there is no build step. Google Fonts are optional and have system fallbacks.

## What the tutorial teaches

OpenAI’s first open-weight language models since GPT-2 combine sparse experts, mixed local and full attention with an attention sink, four-bit expert weights and a structured format for tool calls.

- Stored weights and active weights
- Two attention savings, and a sink
- Four-bit expert weights
- Handing work to a tool
- Choosing how much to reason

## Figures

- **Visit a few experts; store them all** (`moe`): Simplified router with eight experts, four of them selected to start. gpt-oss selects 4 of 32 or 4 of 128; the routes shown are made up. Using a few experts per token saves computation, not the memory to store them all.
- **Leave some attention weight unused** (`sink`): Simplified single head with made-up numbers. The slider sets a hypothetical learned sink score; it is not a user setting in gpt-oss. Real attention works on vectors. A sink that contributes nothing lets a head put less total weight on the visible tokens.
- **A lower bound for storing 21 billion numbers** (`precision`): Simplified storage if all 21 billion weights used one format. The real model mixes precisions, and MXFP4 adds scale factors. Fewer bits and fewer active experts save different resources.
- **One assistant turn can include a tool call** (`flow`): Simplified exchange between application and model. This page runs no tools. A tool result returns control to the model; it does not end the assistant’s turn.

## Accuracy and scope

Release date: 2025-08-05. OpenAI released both models on August 5, 2025.

- Both sizes share a page because they share the same mechanisms; their parameter and expert counts differ.
- The four-bit chart is an ideal lower bound. It does not claim that the whole model uses four bits or that it fits on a particular device.
- This page does not guess at the full training data, which OpenAI has not published.
- The attention sink changes how attention mixes values. It is not an extra cached token, a way to compress context, or a guarantee of faithful reasoning.
- Reported checkpoint sizes, ideal packed-weight sizes and peak serving memory are three different quantities.

The page includes primary source links next to claims and a source list. Research cutoff: 2026-10-03. Quantitative widgets are explicitly simplified teaching models, not executed model inference or performance benchmarks.

## Maintenance

Edit `lesson.json`; run `pnpm author` from the collection directory to refresh committed static files. Run `pnpm test` there. The optional authoring utility is not required to view or deploy this page.
