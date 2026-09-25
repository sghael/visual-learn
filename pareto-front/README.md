# Intelligence versus cost

This explainer plots the Artificial Analysis Intelligence Index against cost per Index task for 162 language models and draws their Pareto frontier. A model is on the frontier when every cheaper plotted model scores lower. The page teaches how to read the frontier as a budget (the best score available at or below a given cost), shows how the frontier filled in as models were released from December 2024 to September 2026, and explains which models the data leaves out and why.

Open `index.html` directly from disk, serve the folder with `python3 -m http.server`, or visit the [published page](https://sghael.github.io/visual-learn/pareto-front/). The page is one HTML file with its styles, code and data embedded; web fonts load from Google Fonts and fall back to system fonts.

## Figures

- **Figure 1, the frontier at one date.** Every model with both an Index score and a cost per task, as a grey dot; the frontier as a blue step line with blue dots and direct labels. The resting state is the latest archived date, September 23, 2026. Controls:
  - **Play** steps the date forward one week at a time from December 26, 2024, adding models in order of release; it stops at the latest date. **Pause** holds the current date. **Reset** returns to the latest date and clears the selection.
  - The **Released by** slider picks any weekly date directly. Moving it stops playback.
  - Clicking or tapping a dot, or choosing a model from the **Model** list, selects that model. The line under the chart gives its creator, release date, score and cost. For a model off the frontier it names the cheapest frontier model that scores at least as high, and the chart joins the two with a dashed line. The list holds the models visible at the chosen date, so keyboard users and readers facing overlapping dots can reach every model. Clicking empty chart space clears the selection.
  - The chart is drawn at the container's real width. On narrow screens each run of frontier points from one model gets a single label with the model name; on wide screens each point is labeled, with the effort setting only after the first point of a run.
- **Figure 2, quarter by quarter.** Small multiples of the same chart at the last day of each quarter from December 31, 2024 to the latest date, on identical axes. Each panel shows that date's frontier in blue, other released models in grey, and the September 23, 2026 frontier as a thin grey reference step. The panels are static.

## Accuracy caveats

- **Positions are fixed.** Every model is plotted at its last observation between September 5 and September 23, 2026. Dates only decide when a model appears. Neither figure reconstructs past scores or prices, and prices can change after release.
- **Index revision.** [Artificial Analysis revised the Intelligence Index](https://artificialanalysis.ai/methodology/intelligence-benchmarking) in September 2026; the [archive methodology](https://llm-frontier.catalystneuro.com/methodology/) dates the change to September 5, 2026 (version 4.3). Scores across that boundary cannot be compared, so the build uses observations from September 5 onward only.
- **Missing models.** The archive contains 36 retired models with no observation on that basis, including GPT-5 and GPT-5.1 from 2025. They are omitted, which is the main reason the early frontier is sparse. DeepSeek V3 appears first because it has the earliest release date among the models that remain, not because it led the field in December 2024. `build.py` asserts the count of 36 so that the page text is reviewed if it changes.
- **Score without cost.** On the live Artificial Analysis site some models, for example [Claude 4 Opus](https://artificialanalysis.ai/models/comparisons/claude-4-opus-vs-claude-4-opus-thinking) and [GPT-5.2](https://artificialanalysis.ai/models/comparisons/gpt-5-2-medium-vs-claude-opus-4-5), have a score but no reported cost per Index task. They are outside the archived data set.
- **Plotted models only.** The frontier describes the plotted models. A model outside the data could lie above it.
- **Names.** The build shortens Artificial Analysis effort suffixes, for example "(Adaptive Reasoning, Max Effort, Default Fallback)" becomes "(max)" and a non-default fallback is kept, as in "Claude Fable 5 (max, Opus 4.8 fallback)".

## Data and build

Data: [Artificial Analysis](https://artificialanalysis.ai/#intelligence-comparison-tabs), as preserved in the [CatalystNeuro archive](https://github.com/catalystneuro/llm-frontier). `upstream-history.json` is the archived input snapshot, last updated September 23, 2026.

`build.py` filters the snapshot to models with both values, fixes each position, computes the frontier for every weekly date and every quarter end, and writes:

- `data.json`: `models` (id, display name, creator, release date, cost, score), `frames` (weekly dates with model count and frontier ids, cheapest first), `snapshots` (the quarter ends used by Figure 2), `excluded`, and the fixed axes;
- `index.html`: `template.html` with that data embedded;
- `pareto-front.gif`: an animation of Figure 1 over the weekly dates, in the page's palette.

Keep the generated files in the repository so the page works without a build step. To rebuild, install Pillow 12.3.0 and run `python3 build.py` in this folder; `python3 build.py --no-gif` skips the slower GIF while editing the template. The GIF uses locally available system fonts (Palatino and Helvetica Neue on macOS, DejaVu on Linux), so its lettering can differ between machines.

## Testing

`pnpm install --frozen-lockfile && pnpm test` runs the browser tests with Playwright. They check the resting state at 1440 px and 390 px (latest date, no autoplay, labeled frontier, no horizontal overflow, chart text at least 11 px), recompute every stored frontier from the model list, check the numbers quoted in the prose against the data, and drive the slider, dot selection, the model list, Play, Pause and Reset, including fake-clock tests that a cancelled run never moves the date afterwards.
