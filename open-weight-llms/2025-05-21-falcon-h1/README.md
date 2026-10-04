# Falcon-H1 · 0.5B to 34B

How attention and a recurrent state can work beside each other

Open `index.html` directly in a browser, or follow this folder from the collection index. All runtime CSS and JavaScript are local; there is no build step. Google Fonts are optional and have system fallbacks.

## What the tutorial teaches

Falcon-H1 combines attention and Mamba-2 branches in parallel within a block, then concatenates their outputs.

- A hybrid family, rather than one fixed size
- Both branches receive the same input
- The hybrid retains two different memory terms
- Equal parameter counts can hide different paths
- Check the checkpoint and workload you actually need

## Figures

- **Two branches inside one mixer** (`flow`): Simplified Falcon-H1 block. The attention and Mamba-2 operations shown together run as parallel branches, not as consecutive stages; residual, normalization and MLP details are omitted. Combine branch outputs after both have processed the same token representation.
- **Two components contribute to hybrid memory** (`state`): Illustrative recurrent-state and attention-cache scalar counts. Falcon-H1 contains both terms, so its total sequence memory is not the flat recurrent line. Excludes weights, batch and runtime overhead. A hybrid reduces some costs while retaining a sequence-length-dependent attention cache.

## Accuracy and scope

Release date: 2025-05-21. TII’s public launch announcement is dated May 21, 2025. The detailed report appeared in July; repository creation alone was not used as proof of public release.

- Hybrid memory includes both fixed-state and growing attention-cache components.
- The block diagram is architectural; it does not promise simultaneous kernel execution.
- The state widget uses invented scalar counts and does not predict real memory or quality.
- Model-family size labels are rounded names, not exact parameter inventories.

The page includes primary source links next to claims and a source list. Research cutoff: 2026-10-03. Quantitative widgets are explicitly simplified teaching models, not executed model inference or performance benchmarks.

## Maintenance

Edit `lesson.json`; run `pnpm author` from the collection directory to refresh committed static files. Run `pnpm test` there. The optional authoring utility is not required to view or deploy this page.
