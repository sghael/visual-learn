# K2 Horizon

An interactive guide to the six K2 Horizon language models released by
MBZUAI's Institute of Foundation Models on 3 September 2026. It covers model
sizes, Mixture-of-Value Attention (MoVA), Uno decoding, the 375B model's
training stages, reported benchmarks, and the status of IFM's promised release
artifacts. Built in the spirit of the
[Polo Club](https://poloclub.github.io/) explainers.

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
| Models | log-scale bar chart + spec sheet | Total and active parameters for all six models; model dimensions from each published `config.json` |
| Transformer layers | switchable block diagram | Dense, MoE (375B), and MoE + MoVA (36B-A4B) layers |
| MoVA | routed value pool | Stored value experts and active value experts per token |
| Uno decoding | two-lane simulation | Autoregressive decoding and a simplified diffusion-draft/verification loop |
| Training stages | stage track + stage buttons | Pretraining, four context-extension stages, SFT, and RL merging for the 375B model |
| Training data | two-ring donut (static) | IFM's reported corpus composition: roughly 20T tokens, half synthetic, about 17% reasoning trajectories |
| Benchmarks | grouped bars | Self-reported results against the competitors IFM chose for each model card |
| Release artifacts | table (static) | Available, partial, and announced artifacts as of 10 September 2026 |

The page supports light and dark themes (system, or the toggle in the rail).

## Accuracy caveats

- Benchmark numbers are transcribed from IFM's Hugging Face model cards and
  are not independent evaluations. The 375B Terminal-Bench 2.1 score uses
  IFM's reward-hacking-audited 66.9%; its current model-card table still shows
  the raw 70.2%.
- The 32B benchmarks are for the `K2-Horizon-32B-Stage1` checkpoint; the model
  card says the final checkpoint and Stage 2 results are forthcoming. The page
  labels them as such.
- IFM has not documented MoVA's routing function in enough detail to reproduce
  it. The
  MoVA diagram shows the structure (a routed pool of value experts) with
  random routing. The Uno widget is a simplified simulation: its "draft
  agreement" slider is an invented probability, and its pass counter omits
  adapter overhead, batching, and hardware utilization.
- Training-stage token counts and step counts come from the 375B model card.
  The RL stage has no published token count; its bar width is illustrative
  and labelled as such.
- Layer counts and dimensions come from each model's `config.json`. The model
  names use rounded parameter counts; Hugging Face reports slightly different
  exact sizes for the stored tensors.
- IFM describes K2 Horizon as fully open, but its 10 September artifact
  inventory still listed the code repository as in progress and checkpoints
  as a partial rollout. The page reports status rather than treating the
  announced release plan as complete.

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
