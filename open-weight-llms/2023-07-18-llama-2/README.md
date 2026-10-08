# Llama 2 · 7B, 13B, 70B

How a text predictor was trained into an assistant, and how the largest version saved memory

Open `index.html` directly in a browser, or follow this folder from the collection index. All runtime CSS and JavaScript are local; there is no build step. Google Fonts are optional and have system fallbacks.

## What the tutorial teaches

Meta released Llama 2 as both a base model and a chat model, described how the chat model was trained, and gave the 70B version a cheaper way to store a conversation.

- What a base model learns
- How the chat model was trained
- Why a long conversation uses memory
- Working out the cache size
- Which saving matters depends on the job

## Figures

- **From continuation to assistant** (`flow`): Simplified training order for Llama 2-Chat. Meta repeated the last two stages over several rounds, collecting new comparisons each time. Each stage gives the model a different kind of feedback.
- **Sharing keys and values shrinks the cache** (`kv`): Simplified size of the KV cache for one conversation, with each number stored in two bytes (bf16). The model shape is made up for teaching: 32 layers, 32 query heads, 8 shared key–value heads, 128 numbers per head. Llama 2 70B is larger. Weights and other runtime memory are excluded. Sharing makes each token cheaper to store; a longer conversation still needs more memory.

## Accuracy and scope

Release date: 2023-07-18. Meta released the base and chat weights on July 18, 2023.

- The memory figure uses a made-up model shape and counts only the KV cache. It does not predict speed or answer quality.
- The training description follows the paper’s overview. It omits details such as Ghost Attention, the paper’s method for keeping a system instruction in effect over many turns.
- Llama 2 used grouped-query attention and reinforcement learning from human feedback; it invented neither.

The page includes primary source links next to claims and a source list. Research cutoff: 2026-10-03. Quantitative widgets are explicitly simplified teaching models, not executed model inference or performance benchmarks.

## Maintenance

Edit `lesson.json`; run `pnpm author` from the collection directory to refresh committed static files. Run `pnpm test` there. The optional authoring utility is not required to view or deploy this page.
