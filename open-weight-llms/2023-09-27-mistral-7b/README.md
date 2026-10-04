# Mistral 7B

Share the memory; move the window

Open `index.html` directly in a browser, or follow this folder from the collection index. All runtime CSS and JavaScript are local; there is no build step. Google Fonts are optional and have system fallbacks.

## What the tutorial teaches

Grouped-query attention and a rolling attention window reduce the cost of remembering a conversation.

- Why yesterday’s tokens cost memory
- Many questions can share the same records
- A local window changes the pattern of work
- Savings multiply, while quality still needs testing
- What became easier to build

## Figures

- **Two independent multipliers: length and KV heads** (`kv`): Simplified bf16 KV-only calculation, using v0.1 head and layer dimensions. It omits sliding-window eviction, weights, batch size, allocator overhead and sharding. GQA reduces memory per retained token; it does not by itself limit how many tokens are retained.
- **Watch the neighborhood move** (`attention`): Simplified single-layer causal mask. Four positions stand in for the release’s much larger window. The generic hybrid preset is only a teaching pattern, not Mistral v0.1’s architecture. Direct access is local; information can also travel indirectly through successive layers.

## Accuracy and scope

Release date: 2023-09-27. Initial Mistral 7B v0.1 public release on September 27, 2023; the technical paper followed in October. This lesson describes v0.1.

- All widgets are teaching calculations, not benchmarks.
- The GQA widget intentionally omits v0.1 sliding-window eviction so head sharing can be inspected independently.
- Reachability through stacked local layers does not guarantee long-range retrieval quality.

The page includes primary source links next to claims and a source list. Research cutoff: 2026-10-03. Quantitative widgets are explicitly simplified teaching models, not executed model inference or performance benchmarks.

## Maintenance

Edit `lesson.json`; run `pnpm author` from the collection directory to refresh committed static files. Run `pnpm test` there. The optional authoring utility is not required to view or deploy this page.
