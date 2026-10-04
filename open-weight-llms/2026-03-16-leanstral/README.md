# Leanstral / 1.5

Turn a proposed proof into a checked artifact

Open `index.html` directly in a browser, or follow this folder from the collection index. All runtime CSS and JavaScript are local; there is no build step. Google Fonts are optional and have system fallbacks.

## What the tutorial teaches

Formal proof tools turn code-agent attempts into inspectable feedback and verifiable training outcomes.

- Begin with something a checker can mean
- Learn to work inside the feedback loop
- Reward a completed trajectory
- More attempts create more opportunities
- A passed check has a precise boundary

## Figures

- **The model proposes; tools make the remaining work visible** (`flow`): Simplified proof-engineering loop. Rejected candidates return to editing; acceptance depends on the task’s statement and allowed assumptions. Reliable feedback changes both how an agent searches and which trajectories can teach it.
- **Separate the amount of work from elapsed time** (`parallel`): Illustrative independent-attempt durations in arbitrary units. The generic scheduler is not the Leanstral inference implementation, and it does not predict proof success. Parallel attempts can spend more compute within a shorter time budget; verification identifies acceptable outputs.

## Accuracy and scope

Release date: 2026-03-16. First Leanstral public weights: March 16, 2026. The July 2 Leanstral 1.5 announcement and report are explicitly identified where their training details are discussed.

- March 2603 checkpoint facts and July 1.5 training details are labeled separately.
- A checked proof establishes its formal statement under its assumptions; it does not automatically validate the English specification.
- The parallel widget models independent scheduling only, not actual serving performance or coordinated agents.

The page includes primary source links next to claims and a source list. Research cutoff: 2026-10-03. Quantitative widgets are explicitly simplified teaching models, not executed model inference or performance benchmarks.

## Maintenance

Edit `lesson.json`; run `pnpm author` from the collection directory to refresh committed static files. Run `pnpm test` there. The optional authoring utility is not required to view or deploy this page.
