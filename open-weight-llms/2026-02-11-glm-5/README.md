# GLM-5

How an agent can learn while other agents are still working

Open `index.html` directly in a browser, or follow this folder from the collection index. All runtime CSS and JavaScript are local; there is no build step. Google Fonts are optional and have system fallbacks.

## What the tutorial teaches

GLM-5 lets rollout generation and learning run on separate schedules so slow agent tasks do not stall training, and adopts sparse attention to cut the cost of long contexts.

- The slowest attempt sets the pace
- Overlap removes waiting, not work
- Learning from an older policy’s experience
- A separate saving: sparse attention

## Figures

- **Generation and learning on separate schedules** (`flow`): Simplified asynchronous reinforcement-learning pipeline. Rollout generation continues while the learner updates. Decoupling generation from learning removes the forced wait for every running task.
- **The cost of an uneven mix of jobs** (`parallel`): Made-up independent job durations with greedy scheduling. Nothing here estimates GLM-5’s training speed. Overlapping independent jobs shortens elapsed time without reducing total work.

## Accuracy and scope

Release date: 2026-02-11. The weights were uploaded on February 11, 2026. Z.ai’s blog post is dated February 12 and the report February 17.

- The scheduling figure shows independent jobs overlapping. It does not reproduce GLM-5’s training framework, called slime, or its asynchronous optimization.
- The gains from sparse attention and from asynchronous training cannot be multiplied into one overall speed-up without measurements.

The page includes primary source links next to claims and a source list. Research cutoff: 2026-10-03. Quantitative widgets are explicitly simplified teaching models, not executed model inference or performance benchmarks.

## Maintenance

Edit `lesson.json`; run `pnpm author` from the collection directory to refresh committed static files. Run `pnpm test` there. The optional authoring utility is not required to view or deploy this page.
