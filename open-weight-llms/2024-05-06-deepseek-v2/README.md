# DeepSeek-V2

Remember a compact description instead of every expanded view

Open `index.html` directly in a browser, or follow this folder from the collection index. All runtime CSS and JavaScript are local; there is no build step. Google Fonts are optional and have system fallbacks.

## What the tutorial teaches

Multi-head latent attention compresses each token’s cached representation while fine-grained experts keep feed-forward work sparse.

- Several large views can share a small description
- Make the storage bottleneck visible
- Position is information that compression must preserve
- Share routine work and route the remainder
- Cache efficiency and weight size remain separate

## Figures

- **How many numbers must one token leave behind?** (`latent`): Simplified dimensional bottleneck with invented dimensions. This is not a faithful MLA cache implementation and omits the separate rotary-position component. A smaller learned coordinate system reduces cached numbers before numerical precision is considered.
- **One shared path plus selected paths** (`moe`): Simplified, scaled-down DeepSeekMoE teaching model. Eight routed experts and one shared expert are illustrative counts, not V2’s configuration. Shared work remains active even when routed work is sparse.

## Accuracy and scope

Release date: 2024-05-06. The official repository records the public release on May 6, 2024. The paper was submitted May 7; folder dates track weights rather than paper submission.

- The latent-width slider does not model quality, RoPE storage or exact MLA implementation.
- The expert widget uses scaled-down counts and cannot predict throughput.
- The weights use the release’s custom DeepSeek Model License; code and weights have different licenses.

The page includes primary source links next to claims and a source list. Research cutoff: 2026-10-03. Quantitative widgets are explicitly simplified teaching models, not executed model inference or performance benchmarks.

## Maintenance

Edit `lesson.json`; run `pnpm author` from the collection directory to refresh committed static files. Run `pnpm test` there. The optional authoring utility is not required to view or deploy this page.
