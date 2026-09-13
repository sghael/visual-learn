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
| Hero | Input image beside the last-layer `[CLS]` attention map | 0 / 4 registers toggle (auto-flips three times on load unless reduced motion is requested; a click cancels it) |
| 01 Tokens | 14×14 patch grid and the 197-token sequence it becomes | Hover a patch or a token to see its position, content and edge density |
| 02 Artifacts | Token-norm heatmap plus a norm histogram and a probe panel | Layer slider, DINOv2 ViT-S/B/L selector, image overlay; click or focus-and-Enter a patch to probe it |
| 03 Hypothesis | Neighbour-similarity map with future artifacts circled | Static |
| 04 Registers | Input-sequence diagram, norm or attention map, token-norm strip, norm budget | 0 to 8 register slider, norm/attention view toggle |
| 05 Results | Table of the paper's results (DeiT-III ViT-B, OpenCLIP ViT-B, DINOv2 ViT-L) with and without 4 registers | Static |
| 06 FAQ | Register count, placement, relation to LLM attention sinks, retraining | Expandable |

## Accuracy caveats

- **Every heatmap is a simulation.** A small deterministic model, seeded so
  the page is identical on every load, reproduces the paper's qualitative
  findings for DINOv2: artifacts appear mid-network, only at ViT-L and
  larger, only in redundant patches, and vanish once any register exists.
  The size threshold is recipe-dependent; the paper finds artifacts in
  DeiT-III and OpenCLIP at ViT-B, which the S/B/L selector does not model. No real network
  runs in the page, and the page says so in each caption and the footer.
- **Probe numbers** (position top-1, reconstruction L2, ImageNet linear
  probe) are the paper's reported values for DINOv2 ViT-g, shown for both the
  clicked token kinds regardless of which simulated layer is selected.
- **Results table** values were checked against the paper's Tables 2 and 3
  (ICLR 2024 version) and are asserted by the tests.
- **Register order** follows the official DINOv2 implementation: `[CLS]`,
  registers, then patch tokens. The paper's text says "appended"; the
  page notes both.
- The "DINOv2 ViT-S and ViT-B never develop artifacts" statement follows the
  paper's size study, not a theorem.

## Tests

```bash
pnpm install
pnpm test
```

Playwright drives `index.html` in headless Chrome and checks: no console or
page errors and no horizontal overflow at 1440 px and 390 px; the hero
toggle, layer slider, model selector, register slider and view toggle all
change what the simulation shows in the direction the prose promises; patch
cells and tokens are keyboard operable; the results table matches the paper;
and the hero autoplay flips exactly three times, not at all under reduced
motion, and stops on a click.
