# DeepSeek-V2

Caching a small compressed vector for each token instead of full keys and values

Open `index.html` directly in a browser, or follow this folder from the collection index. All runtime CSS and JavaScript are local; there is no build step. Google Fonts are optional and have system fallbacks.

## What the tutorial teaches

DeepSeek-V2 cached one small learned vector per token instead of full keys and values, and paired it with many small experts, cutting both memory and computation.

- Storing a compact summary instead of full keys and values
- How much smaller the cache gets
- Why position needs separate handling
- Shared experts plus many small routed ones
- Two savings for two different costs

## Figures

- **How many numbers must each token leave behind?** (`latent`): Simplified storage comparison with made-up sizes. It omits MLA’s separate position key and the cost of rebuilding keys and values. Storing fewer numbers per token shrinks the cache before any reduction in bits.
- **One shared path plus selected paths** (`moe`): Simplified router with 8 routed experts and 1 shared expert. DeepSeek-V2 has 160 routed and 2 shared experts in each layer. Shared experts run on every token, however sparse the routed experts are.

## Accuracy and scope

Release date: 2024-05-06. DeepSeek’s repository records the weight release on May 6, 2024. The paper was posted on May 7.

- The latent-width figure does not model quality, the separate position key or the exact MLA computation.
- The expert figure uses scaled-down counts and cannot predict speed.
- The weights use DeepSeek’s own model license; the code uses the MIT license.

The page includes primary source links next to claims and a source list. Research cutoff: 2026-10-03. Quantitative widgets are explicitly simplified teaching models, not executed model inference or performance benchmarks.

## Maintenance

Edit `lesson.json`; run `pnpm author` from the collection directory to refresh committed static files. Run `pnpm test` there. The optional authoring utility is not required to view or deploy this page.
