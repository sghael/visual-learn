# Intelligence versus cost

This chart shows the plotted Pareto frontier of language models with a reported Artificial Analysis Intelligence Index score and a cost per Index task. A point is on the frontier when no plotted model has both a higher score and a lower cost.

Open `index.html` directly or visit the [published page](https://sghael.github.io/visual-learn/pareto-front/). The page is one HTML file with its styles, code, and data embedded.

## Controls

- Drag the date slider to show models released by that date. Play and Pause move through the dates automatically.
- Click a dot to see its model, creator, score, cost per task, release date, and frontier status. The selected dot is outlined on the chart.
- Use the model menu to inspect an overlapping dot or navigate by keyboard. The menu lists models visible at the chosen date.

## Reading the dates

The dates control when models enter the chart. Every point uses its **September 2026** score and cost, so this is a release chronology, not a record of how prices and scores changed at each date. The axes and positions stay fixed. Models without a published cost per Index task are omitted. That leaves gaps in 2025, including some models with higher scores than those shown. The frontier describes only the models plotted here.

DeepSeek V3 appears first because it has the earliest release date among models in the archive with both values. It was not necessarily the most intelligent model available then. Artificial Analysis reports a score but no Index-task cost for examples such as [Claude 4 Opus](https://artificialanalysis.ai/models/comparisons/claude-4-opus-vs-claude-4-opus-thinking) and [GPT-5.2](https://artificialanalysis.ai/models/comparisons/gpt-5-2-medium-vs-claude-opus-4-5). The page explains this below the chart.

Data: [Artificial Analysis](https://artificialanalysis.ai/#intelligence-comparison-tabs), as preserved in the [CatalystNeuro archive](https://github.com/catalystneuro/llm-frontier). `upstream-history.json` is the archived input snapshot, last updated September 23, 2026. `build.py` filters it to models with both values, fixes each plotted position to its September 2026 observation, and writes `data.json`, the self-contained `index.html`, and `pareto-front.gif` from `template.html`. Keep the generated files with the source so this page works without a build step.

To rebuild, install Pillow 12.3.0 and run `python3 build.py` in this folder. The GIF uses locally available system fonts, so its exact lettering can differ between macOS and Linux.

## Testing

`pnpm install --frozen-lockfile && pnpm test` runs the browser interaction and layout checks.
