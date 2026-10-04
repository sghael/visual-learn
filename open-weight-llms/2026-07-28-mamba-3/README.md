# Mamba-3 · MIMO 1.5B

How a recurrent state can remember more through richer updates

Open `index.html` directly in a browser, or follow this folder from the collection index. All runtime CSS and JavaScript are local; there is no build step. Google Fonts are optional and have system fallbacks.

## What the tutorial teaches

Mamba-3 improves the recurrence, adds input-dependent rotations, and performs multiple input/output operations within a fixed-size state.

- Separate the research date from the weight release
- Mix information from both ends of an interval
- Remember direction as well as magnitude
- Do more computation while the state is already available
- A better state still has finite capacity

## Figures

- **One recurrent computation, with richer internal operations** (`flow`): Simplified conceptual Mamba-3 update. Rotations can be implemented on projections; the actual block also includes normalization, gates, multiple heads and an MLP. The architecture changes how a fixed-size state is updated and read.
- **Keep the state size independent of sequence length** (`state`): Illustrative scalar counts, not actual Mamba-3 dimensions or measured runtime memory. Excludes weights, temporary buffers and constant update metadata. A fixed-size state is not a lossless copy of the past. Richer updates improve a bounded memory without turning it into a growing token archive.

## Accuracy and scope

Release date: 2026-07-28. Official MIMO 1.5B repository commit bc6b5d0f7994fe4cb3478242e92da8daf9ee29ec is titled Mamba-3 public release and dated July 28, 2026. The paper appeared March 16. Public API access was verified despite a stale private-repository sentence in the README.

- Folder date is the official weight-repository public-release commit, not the March paper date.
- The download is a base research checkpoint; paper benchmark results are not automatically results for this artifact.
- Toy arithmetic and state sizes do not reproduce the trained model or prove unlimited retrieval.
- More arithmetic at similar decode latency is workload-dependent.

The page includes primary source links next to claims and a source list. Research cutoff: 2026-10-03. Quantitative widgets are explicitly simplified teaching models, not executed model inference or performance benchmarks.

## Maintenance

Edit `lesson.json`; run `pnpm author` from the collection directory to refresh committed static files. Run `pnpm test` there. The optional authoring utility is not required to view or deploy this page.
