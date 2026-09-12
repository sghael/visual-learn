# Register Tokens Explainer

An interactive walkthrough of *Vision Transformers Need Registers* (Darcet,
Oquab, Mairal and Bojanowski, ICLR 2024): why large ViTs grow a handful of
high-norm "artifact" tokens in featureless background patches, what those
tokens actually contain, and how appending a few learned register tokens to
the input sequence removes them.

Live: https://sghael.github.io/visual-learn/register-tokens/

## Open it

```bash
open index.html                       # from disk
python3 -m http.server 8000 --bind 0.0.0.0   # or served locally
```

One HTML file, no build step. The only external resources are Google Fonts,
with system fallbacks.

## What each widget does

| Section | Widget | Controls |
|---|---|---|
| Hero | Input image beside the last-layer `[CLS]` attention map | 0 / 4 registers toggle (auto-flips three times on load, then stops) |
| 01 Tokens | 14×14 patch grid and the 197-token sequence it becomes | Hover a patch or a token to see its position, content and edge density |
| 02 Artifacts | Token-norm heatmap plus a norm histogram and a probe panel | Layer slider, ViT-S/B/L selector, image overlay; click a patch to probe it |
| 03 Hypothesis | Neighbour-similarity map with future artifacts circled | Static |
| 04 Registers | Input-sequence diagram, norm or attention map, token-norm strip, norm budget | 0 to 8 register slider, norm/attention view toggle |
| 05 Results | Table of the paper's ViT-L results with and without 4 registers | Static |
| 06 FAQ | Register count, placement, relation to LLM attention sinks, retraining | Expandable |

## Accuracy caveats

- **Every heatmap is a simulation.** A small deterministic model, seeded so
  the page is identical on every load, reproduces the paper's qualitative
  findings: artifacts appear mid-network, only in ViT-B and larger, only in
  redundant patches, and vanish once any register exists. No real network
  runs in the page, and the page says so in each caption and the footer.
- **Probe numbers** (position top-1, reconstruction L2, ImageNet linear
  probe) are the paper's reported values for DINOv2 ViT-g, shown for both the
  clicked token kinds regardless of which simulated layer is selected.
- **Results table** values were transcribed from the paper's Tables 2 and 3.
  Verify against the original before quoting them.
- The "ViT-S never develops artifacts" statement follows the paper's
  observation at the scales it studied, not a theorem.

## Tests

```bash
pnpm install
pnpm test
```

Playwright drives `index.html` in headless Chrome and checks: no console or
page errors and no horizontal overflow at 1440 px and 390 px; the hero
toggle, layer slider, model selector, register slider and view toggle all
change what the simulation shows in the direction the prose promises.
