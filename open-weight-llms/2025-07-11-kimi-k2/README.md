# Kimi K2

How stable training and practiced tool use meet

Open `index.html` directly in a browser, or follow this folder from the collection index. All runtime CSS and JavaScript are local; there is no build step. Google Fonts are optional and have system fallbacks.

## What the tutorial teaches

Kimi K2 combines a sparse model, MuonClip optimization, and training trajectories that include actions and feedback.

- Many stored experts, a few consulted at a time
- An optimizer needs numerical guardrails
- A tool trajectory is a lesson with consequences
- Separate the model from the machinery around it

## Figures

- **Storage capacity and selected computation differ** (`moe`): Simplified eight-expert teaching model. Actual K2 has 384 routed experts with eight selected and one shared expert. Sparse activation reduces the work performed per token without removing the stored experts.
- **Stabilize the update before the next batch** (`flow`): Simplified optimizer cycle. K2’s report specifies per-head and MLA-specific details. Training stability is part of making an optimizer useful at scale.

## Accuracy and scope

Release date: 2025-07-11. Official first public Base and Instruct release July 11, 2025; report appeared later.

- Muon and mixture-of-experts predate K2; MuonClip and their scaled integration are the release-specific focus.
- The eight-expert router and scalar clipping arithmetic are teaching models.

The page includes primary source links next to claims and a source list. Research cutoff: 2026-10-03. Quantitative widgets are explicitly simplified teaching models, not executed model inference or performance benchmarks.

## Maintenance

Edit `lesson.json`; run `pnpm author` from the collection directory to refresh committed static files. Run `pnpm test` there. The optional authoring utility is not required to view or deploy this page.
