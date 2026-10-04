# Phi-4-mini-flash-reasoning

How a hybrid decoder reuses memory instead of repeatedly reading the whole past

Open `index.html` directly in a browser, or follow this folder from the collection index. All runtime CSS and JavaScript are local; there is no build step. Google Fonts are optional and have system fallbacks.

## What the tutorial teaches

Phi’s SambaY architecture combines recurrent state, local attention, shared global memory and gated reuse of earlier representations.

- Long generation repeats the memory problem
- A running summary has a fixed number of slots
- Keep both a summary and an addressable record
- A gate selects channels from an earlier representation
- Efficiency claims need the workload beside them

## Figures

- **A fixed summary versus an expanding record** (`state`): Illustrative scalar counts for isolated memory components. Phi-mini-flash is hybrid and still has attention memory; this is not its total runtime memory or a quality comparison. A fixed-state component bounds one memory term by compressing the past.
- **Reuse representations across the decoder** (`flow`): Simplified SambaY organization; the real layers interleave and include additional projections, feedforward computation and differential attention. Some later layers read global memory; others gate a representation already computed.

## Accuracy and scope

Release date: 2025-07-09. Public weights and SambaY paper announced July 9, 2025.

- The fixed-state curve describes only a toy component; the real hybrid still stores global attention memory.
- The gate example is arithmetic intuition, not the full published GMU formula.
- Reported throughput is conditional on workload and runtime, not a universal multiplier.

The page includes primary source links next to claims and a source list. Research cutoff: 2026-10-03. Quantitative widgets are explicitly simplified teaching models, not executed model inference or performance benchmarks.

## Maintenance

Edit `lesson.json`; run `pnpm author` from the collection directory to refresh committed static files. Run `pnpm test` there. The optional authoring utility is not required to view or deploy this page.
