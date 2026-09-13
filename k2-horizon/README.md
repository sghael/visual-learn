# K2 Horizon Explainer

An interactive, visual walkthrough of the K2 Horizon model fleet released by
MBZUAI's Institute of Foundation Models on 3 September 2026: six Apache 2.0
models from 0.9B to 375B parameters, the Mixture-of-Value Attention (MoVA)
used in the 36B-A4B model, the Uno diffusion adapter that speeds up decoding,
the staged training recipe, and what "fully open" adds over open weights.
Built in the spirit of the [Polo Club](https://poloclub.github.io/) explainers.

## Open it

It is a standalone page. Either double-click `index.html`, or serve the folder:

```bash
python3 -m http.server 8000 --bind 0.0.0.0
```

No build step. D3 7.9 loads from cdnjs and the fonts from Google Fonts; with
no network the charts do not draw but the prose still reads.

## What is inside

| Chapter | Widget | What it shows |
|---|---|---|
| 1 The fleet | log-scale bar chart + spec sheet | Total versus active parameters for all six models; click a model for layers, hidden size, heads, vocabulary, context and expert counts from its `config.json` |
| 2 Inside a layer | switchable block diagram | Dense, MoE (375B) and MoE + MoVA (36B-A4B) layers; "Route a token" lights the experts one token uses |
| 3 MoVA attention | routed value pool | Sliders for pool size and active experts; shows why capacity grows while value compute stays fixed |
| 4 Uno decoding | two-lane simulation | Autoregressive decoding versus diffusion-drafted blocks verified by the base model; block size and agreement sliders, forward-pass counter |
| 5 Training pipeline | stage track + stage buttons | Pretraining, four midtraining context extensions (8K to 512K), SFT and RL-then-merge for the 375B model; pick a stage |
| 6 The data | two-ring donut (static) | Roughly 20T tokens, half synthetic, about 17% reasoning trajectories |
| 7 Benchmarks | grouped bars | Each of the six models against the competitors IFM chose in its model card |
| 8 Fully open | table (static) | Weights, checkpoints, code, corpus, evaluation protocols |

The page supports light and dark themes (system, or the toggle in the rail).

## Accuracy caveats

- Benchmark numbers are transcribed from the Hugging Face model cards and
  IFM's announcement as of September 2026 and are self-reported by IFM. The
  375B Terminal-Bench 2.1 score uses IFM's corrected 66.9%, not the launch
  figure of 70.2%.
- The 32B benchmarks are for the `K2-Horizon-32B-Stage1` checkpoint; the model
  card says the final checkpoint and Stage 2 results are forthcoming. The page
  labels them as such.
- IFM has not published MoVA's routing function or Uno's block schedule. The
  MoVA diagram shows the structure (a routed pool of value experts) with
  random routing, and the Uno widget is a toy: its "draft agreement" slider
  stands in for a rate that depends on the text.
- Training-stage token counts and step counts come from the 375B model card.
  The RL stage has no published token count; its bar width is illustrative
  and labelled as such.
- Layer counts and dimensions come from each model's `config.json`. Parameter
  names are the official ones; Hugging Face file sizes run a little higher
  because embeddings are counted separately.
- ifm.ai itself sits behind a bot check, so the page cites the model cards,
  the arXiv Uno paper and repository, and press coverage; see the footer.

## Tests

Playwright drives the page in headless Chrome (the installed Google Chrome, or
Playwright's own Chromium as a fallback). D3 is fetched from cdnjs once and
cached in `test/.d3.min.js` (gitignored), so later runs are offline. The tests check that every stage renders
at 1440 px and 390 px with no console or page errors and no horizontal
overflow, that switching layer variants and models redraws, that the Uno
simulation survives reset mid-run and honours block-size changes, and that the
theme toggle redraws every chart.

```bash
pnpm install
pnpm exec playwright install chromium   # only if Google Chrome is not installed
pnpm test
```
