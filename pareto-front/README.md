# Intelligence versus cost

This chart shows the plotted Pareto frontier of language models with a reported Artificial Analysis Intelligence Index score and a cost per Index task. A point is on the frontier when no plotted model has both a higher score and a lower cost.

Open `index.html` directly or visit the [published page](https://sghael.github.io/visual-learn/pareto-front/). The page is one HTML file with its styles, code, and data embedded.

## Controls

- Drag the date slider to show models released by that date. Play and Pause move through the dates automatically.
- Click a dot to see its model, creator, score, cost per task, release date, and frontier status. The selected dot is outlined on the chart.
- Use the model menu to inspect an overlapping dot or navigate by keyboard. The menu lists models visible at the chosen date.

## Reading the dates

The dates control when models enter the chart. Every point uses its **September 2026** score and cost, so this is a release chronology, not a record of how prices and scores changed at each date. The axes and positions stay fixed.

DeepSeek V3 appears first because it has the earliest release date among models measured on the September 2026 scoring method. It was not necessarily the most intelligent model available then. [Artificial Analysis revised the Index](https://artificialanalysis.ai/methodology/intelligence-benchmarking) in September 2026, and scores across that boundary cannot be compared directly. The generator uses observations from September 5 onward. The archive contains 36 retired models with no observation on that basis, including GPT-5 and GPT-5.1 from 2025, so they are omitted. This is the main reason the early curve is sparse. The [archive methodology](https://llm-frontier.catalystneuro.com/methodology/) explains the scoring periods.

A separate gap exists on the live Artificial Analysis site: some models have a score but no reported Index-task cost. [Claude 4 Opus](https://artificialanalysis.ai/models/comparisons/claude-4-opus-vs-claude-4-opus-thinking) and [GPT-5.2](https://artificialanalysis.ai/models/comparisons/gpt-5-2-medium-vs-claude-opus-4-5) are examples. They are outside this archived data set. The page explains both limits below the chart. Its frontier describes only the plotted models.

Data: [Artificial Analysis](https://artificialanalysis.ai/#intelligence-comparison-tabs), as preserved in the [CatalystNeuro archive](https://github.com/catalystneuro/llm-frontier). `upstream-history.json` is the archived input snapshot, last updated September 23, 2026. `build.py` filters it to models with both values, fixes each plotted position to its September 2026 observation, and writes `data.json`, the self-contained `index.html`, and `pareto-front.gif` from `template.html`. Keep the generated files with the source so this page works without a build step.

To rebuild, install Pillow 12.3.0 and run `python3 build.py` in this folder. The GIF uses locally available system fonts, so its exact lettering can differ between macOS and Linux.

## Testing

`pnpm install --frozen-lockfile && pnpm test` runs the browser interaction and layout checks.
