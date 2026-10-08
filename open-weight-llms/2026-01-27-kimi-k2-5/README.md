# Kimi K2.5

How seeing and delegating change an agent’s work

Open `index.html` directly in a browser, or follow this folder from the collection index. All runtime CSS and JavaScript are local; there is no build step. Google Fonts are optional and have system fallbacks.

## What the tutorial teaches

Kimi K2.5 added image and video input to Kimi K2, and trained an orchestrator to split suitable tasks among helper agents working in parallel.

- Seeing changes what an agent can fix
- Independent work can overlap
- Rewarding useful delegation
- The released model versus the full system

## Figures

- **Acting on what it sees produces new evidence** (`flow`): Illustrative loop for fixing a page from screenshots; not a recorded K2.5 session. Seeing the result closes a feedback loop that source code alone cannot.
- **Total work and elapsed time are different** (`parallel`): Made-up independent tasks in arbitrary time units, scheduled greedily with no coordination cost. Not a measurement of Kimi. Parallel work shortens elapsed time only when the dependencies allow it.

## Accuracy and scope

Release date: 2026-01-27. Moonshot AI released the weights on January 27, 2026. The report appeared on arXiv on February 2, 2026.

- Agent Swarm results come from Moonshot’s hosted system; downloading the weights alone does not reproduce them.
- The scheduler adds up made-up task durations; it does not call any model or simulate reasoning.

The page includes primary source links next to claims and a source list. Research cutoff: 2026-10-03. Quantitative widgets are explicitly simplified teaching models, not executed model inference or performance benchmarks.

## Maintenance

Edit `lesson.json`; run `pnpm author` from the collection directory to refresh committed static files. Run `pnpm test` there. The optional authoring utility is not required to view or deploy this page.
