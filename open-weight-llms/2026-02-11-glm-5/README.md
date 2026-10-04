# GLM-5

How an agent can learn while other agents are still working

Open `index.html` directly in a browser, or follow this folder from the collection index. All runtime CSS and JavaScript are local; there is no build step. Google Fonts are optional and have system fallbacks.

## What the tutorial teaches

GLM-5 uses asynchronous agent reinforcement learning and sparse attention to reduce different sources of wasted computation.

- The slowest rollout can determine the next update
- Overlap reduces waiting, not the amount of work
- Old experience came from an older policy
- Reduce reading cost separately from training waits

## Figures

- **Experience and learning run on different schedules** (`flow`): Simplified asynchronous RL pipeline. Real rollout generation continues while the learner updates. Decoupling removes a mandatory wait for every ongoing task.
- **See the cost of an uneven job mix** (`parallel`): Illustrative independent durations and greedy scheduling. No model throughput or GLM training speedup is inferred. Overlapping independent jobs can shorten elapsed time without reducing total work.

## Accuracy and scope

Release date: 2026-02-11. Public weight uploads February 11, 2026; official blog dated February 12 and report February 17.

- The scheduling widget illustrates independent job overlap; it does not reproduce slime or asynchronous policy optimization.
- Efficiency gains from sparse attention and asynchronous RL cannot be multiplied into an overall speedup without measurements.

The page includes primary source links next to claims and a source list. Research cutoff: 2026-10-03. Quantitative widgets are explicitly simplified teaching models, not executed model inference or performance benchmarks.

## Maintenance

Edit `lesson.json`; run `pnpm author` from the collection directory to refresh committed static files. Run `pnpm test` there. The optional authoring utility is not required to view or deploy this page.
