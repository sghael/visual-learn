# Accelerator Atlas

An interactive, visual tour of every GPU and TPU you can rent on Google Cloud,
as of September 2026. Built in the spirit of the
[Polo Club](https://poloclub.github.io/) explainers.

Live: https://sghael.github.io/visual-learn/gcp-accelerators/

## Open it

It is a standalone page. Either double-click `index.html`, or serve the folder:

```bash
python3 -m http.server 8000 --bind 0.0.0.0
```

No build step. D3 v7 loads from cdnjs (pinned), fonts from Google Fonts with
system fallbacks. Nothing is tracked and no data leaves the browser.

## What is inside

| Chapter | Widget | What it shows |
|---|---|---|
| Hero | canvas | One tile per accelerator model, sized by memory per chip |
| Two ways to multiply | two canvases | GPU streaming multiprocessors taking warps versus a systolic-array wavefront |
| The lineup | cards + spec sheet | Every rentable chip grouped by family; click for machine series, fast-domain size, host shape and bars against the lineup maximum |
| Memory vs. compute | D3 scatter | Log-log map of memory per chip against dense FP8/INT8 or BF16 peak, optional sizing by price |
| Where the bottleneck is | D3 roofline | Pick a chip, drag arithmetic intensity, read whether you are memory- or compute-bound |
| How chips talk | four canvases | 8-GPU NVSwitch island, NVL72 rack, 2D torus pod, rotating 3D torus pod |
| Will my model fit? | calculator + table | Parameters × precision × headroom to chips needed, fast-domain fit and rough $/hr per chip type |
| How you buy it | cards + table | On-demand, Spot, Dynamic Workload Scheduler, reservations; approximate zone footprint per chip |
| Glossary | static | HBM, NVLink, ICI, MXU, SparseCore, slices, dense vs. sparse FLOPS |

## Data and caveats

All numbers are in one array (`CHIPS`) near the top of the script in
`index.html`, with sources in the page footer.

- **Chips covered.** GB300 (A4X Max), GB200 (A4X), B200 (A4), H200 (A3 Ultra),
  H100 (A3 Mega, High, Edge), A100 80/40 GB (A2), RTX PRO 6000 (G4), L4 (G2),
  T4, V100, P100, P4 (N1), TPU v5e, v5p, v6e Trillium and TPU7x Ironwood.
  TPU 8t and 8i are shown as announced (Cloud Next, April 2026), not rentable.
- **Throughput** is the vendor's peak *dense* figure, marked approximate.
  NVIDIA's sparse peaks are not used. Chips without FP8 (A100, V100, T4,
  P100, P4) are plotted at INT8 or FP16 peak, which the page says.
- **Prices** are approximate on-demand list per chip-hour in us-central1,
  from the Google Cloud pricing pages and a Thunder Compute survey dated
  11 Sep 2026; VM-level prices are divided by GPU count. B200, H200 and the
  A4X series are sold via reservations, Spot or DWS, so their figures are
  indicative. GB300 has no public price and is shown without one.
- **The fit calculator** is a memory rule of thumb (weights = parameters ×
  bytes; full fine-tune about 16 bytes per parameter, plus a headroom
  percentage), and it treats 90% of each chip's memory as usable. It ignores
  interconnect speed, so a "fits" over PCIe cards is a capacity statement,
  not a performance recommendation.
- **The roofline** uses dense FP8/INT8 peak over HBM bandwidth. Real kernels
  reach a fraction of either roof.
- **Zone counts** are approximate, from the GPU regions page on 11 Sep 2026.
- The anatomy and topology animations are simplified and say so on the page.

## Tests

The page has no runtime dependencies. The tests do: Playwright drives the
page in headless Chrome (the installed Google Chrome, or Playwright's own
Chromium in CI) and checks rendering at 1440 px and 390 px with zero errors
and no horizontal overflow, that the canvases paint at rest, the card spec
sheet, the scatter toggle, the fit arithmetic and the roofline verdict.

```bash
pnpm install
pnpm test
```
