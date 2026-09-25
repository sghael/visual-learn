# GPUs and TPUs on Google Cloud

A visual guide to the GPUs and TPUs that Google Cloud listed on
11 September 2026. It explains how the two kinds of chip do matrix
arithmetic, compares the whole lineup on memory, bandwidth, compute,
interconnect and price, uses a roofline model to show which of those numbers
limits a workload, and estimates how many chips a model needs.

Live: https://sghael.github.io/visual-learn/gcp-accelerators/

## Open it

It is a standalone page: `index.html`, `styles.css` and `app.js`, with no
runtime dependencies and no build step. Double-click `index.html`, or serve
the folder:

```bash
python3 -m http.server 8000 --bind 0.0.0.0
```

Fonts come from Google Fonts with system fallbacks. Nothing is tracked and no
data leaves the browser. `styles.css` starts with a copy of the Visual Learn
house stylesheet; page-specific rules follow it.

## What is inside

Seven sections, about 15 minutes of reading.

| Section | Figure | What it shows |
|---|---|---|
| How GPUs and TPUs multiply matrices | Fig. 1, static schematics | An H100's 132 SMs with one enlarged (warps issuing or waiting, registers, arithmetic units, Tensor Cores, shared memory), beside one TPU v4 TensorCore (four MXUs, vector and scalar units, HBM) |
| | Fig. 2, three small multiples | A 6 × 6 systolic array at cycles 1, 4 and 11: how it fills along a diagonal and then keeps every cell busy |
| Every accelerator Google Cloud rents | Fig. 3, table | One row per chip, grouped by architecture: memory (with an inline bar), bandwidth, dense low-precision and BF16 peaks, fast-domain size, price, machine series. Selecting a chip's name opens its machine type, host shape, chips per VM, price basis and notes |
| Two speed limits | Fig. 4, interactive roofline | Pick a chip and a workload (three presets or a slider for arithmetic intensity); the chart and the sentence under it show the ridge point, whether the workload is memory- or compute-bound, and the share of peak it can reach |
| | Fig. 5, static scatter | Dense peak against bandwidth for every chip, with diagonal lines of equal ridge point (100, 300, 1,000 FLOP/byte) |
| Fast domains | Fig. 6, four small multiples | 8-GPU NVSwitch, 72-GPU NVL72 rack, 2D torus, 3D torus |
| How many chips a model needs | Fig. 7, calculator + table | Parameters, weight precision, a KV-cache/activation allowance and mode give the memory needed, the cheapest on-demand option that fits, the fewest chips, and a per-chip table sorted by hourly cost |
| Buying capacity and finding it | two tables | Purchasing models (on demand, Spot, Dynamic Workload Scheduler, reservations) and approximate zone footprints |
| Glossary | | Arithmetic intensity, dense vs. sparse FLOPS, HBM, ICI, MXU, Multislice, NVLink, RoCE/GPUDirect, slice, SparseCore, vWS |

Nothing animates. The roofline redraws when a control changes; the other
figures are static.

## Data and caveats

All numbers are in one array (`CHIPS`) at the top of `app.js`. Sources are
listed in the page's colophon.

- **Chips covered.** GB300 (A4X Max), GB200 (A4X), B200 (A4), H200 (A3 Ultra),
  H100 (A3 Mega, High, Edge), A100 80/40 GB (A2), RTX PRO 6000 (G4), L4 (G2),
  T4, V100, P100, P4 (N1), TPU v2, v3, v4, v5e, v5p, v6e Trillium and TPU7x
  Ironwood. TPU 8t and 8i are described as announced (Cloud Next, April 2026),
  not rentable. v2 and v3 are legacy but still priced and zoned by Google.
- **Throughput** is the vendor's peak *dense* figure, marked approximate on
  the page. NVIDIA's sparse peaks are divided by two. Each record carries a
  `peakBasis` (FP8, INT8, FP16 or BF16) naming the precision behind its
  low-precision number. The P4 has no fast 16-bit mode, so its "BF16 or FP16"
  cell is its FP32 figure and says so.
- **Prices** are list per chip-hour in us-central1 (or the first region
  Google lists), from the Google Cloud pricing pages and a Thunder Compute
  survey dated 11 September 2026; VM-level prices are divided by GPU count.
  A3 Ultra, A3 Mega, A4 and A4X are not sold on demand, so their prices are
  marked `*` as indicative and never win "cheapest on demand". GB300 and GB200
  have no public price.
- **The roofline** uses dense low-precision vendor peak and peak memory
  bandwidth. Real kernels reach a fraction of either. The workload presets
  count weight traffic only, with one-byte weights: 2 FLOP per byte per
  sequence in the batch (decode at batch 1 ≈ 2, batch 64 ≈ 128) and ≈ 2,048
  for a 1,024-token prefill. KV-cache reads, cache reuse and kernel fusion
  change real intensities; the page says so.
- **The fit calculator** is a capacity estimate: parameters × bytes plus a
  user-selected allowance (not a computed KV cache), or 16 bytes per parameter
  for a mixed-precision Adam full fine-tune, against 90% of each chip's
  memory. A "fits" result is not a performance recommendation.
- **Host shapes** (vCPU, RAM, network) are the largest documented VM per
  chip; for TPUs the network figure is the per-VM NIC. Google publishes no VM
  shape for v2, v3 and v4, and the page says so.
- **Zone counts** are approximate, from the GPU regions page on
  11 September 2026.
- **Schematics** in Figures 1, 2 and 6 are simplified and say so on the page.

## Tests

The page has no runtime dependencies. The tests do: Playwright drives the page
in headless Chrome (the installed Google Chrome, or Playwright's own Chromium
in CI). They check, at 1440 px and 390 px, that every section and figure
renders with no errors, no horizontal page overflow and no chart text under
11 px; that charts redraw at their container's width; the lineup's
expandable rows and price markers; the roofline verdict, the preset values
against what they compute, and that batch-1 decode is memory-bound on every
chip; the fit arithmetic, including indicative prices and the fine-tune mode;
and data invariants for every chip record against Google's and NVIDIA's
published tables.

```bash
pnpm install --frozen-lockfile
pnpm test
```
