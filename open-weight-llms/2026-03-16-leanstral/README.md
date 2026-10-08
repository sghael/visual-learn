# Leanstral / 1.5

How a proof checker turns a model’s attempts into reliable feedback

Open `index.html` directly in a browser, or follow this folder from the collection index. All runtime CSS and JavaScript are local; there is no build step. Google Fonts are optional and have system fallbacks.

## What the tutorial teaches

Leanstral is trained to write formal Lean proofs inside real projects, using the proof checker’s feedback to guide its work and the checker’s verdict as its training reward.

- A checker needs an exact statement
- Working inside the feedback loop
- Rewarding a completed session
- More attempts, more chances
- What a passed check proves

## Figures

- **The model proposes; the tools show what remains** (`flow`): Simplified proof-engineering loop. Rejected attempts go back to editing; acceptance depends on the task’s statement and permitted assumptions. Reliable feedback changes how an agent searches and which attempts it can learn from.
- **Total work versus elapsed time** (`parallel`): Made-up durations for independent attempts, in arbitrary units. This generic scheduler is not Leanstral’s serving system and does not predict proof success. Parallel attempts spend more computation in less time; the checker picks out the valid ones.

## Accuracy and scope

Release date: 2026-03-16. Mistral released the first Leanstral weights on March 16, 2026. Training details from the Leanstral 1.5 announcement and report of July 2 are labeled where they appear.

- Facts about the March checkpoint and training details from the July 1.5 report are labeled separately.
- A checked proof establishes its formal statement under its assumptions; it does not show that the statement matches the English requirement.
- The scheduling figure models independent attempts only, not real serving performance or cooperating agents.

The page includes primary source links next to claims and a source list. Research cutoff: 2026-10-03. Quantitative widgets are explicitly simplified teaching models, not executed model inference or performance benchmarks.

## Maintenance

Edit `lesson.json`; run `pnpm author` from the collection directory to refresh committed static files. Run `pnpm test` there. The optional authoring utility is not required to view or deploy this page.
