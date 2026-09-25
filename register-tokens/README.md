# Register tokens in Vision Transformers

An illustrated explanation of *Vision Transformers Need Registers* (Darcet,
Oquab, Mairal and Bojanowski, ICLR 2024,
[arXiv:2309.16588](https://arxiv.org/abs/2309.16588)). The page teaches:

1. how a ViT turns an image into a sequence of `[CLS]` plus patch tokens;
2. what the high-norm "artifact" tokens are, where they appear, and in which
   layers and model sizes;
3. what linear probes show those tokens contain (less local information, more
   global information);
4. how learned register tokens take over that role, and what they cost;
5. what changes downstream, from the paper's tables;
6. related questions: registers versus `[CLS]`, which models have artifacts,
   attention sinks in language models, and training-free registers.

Live: https://sghael.github.io/visual-learn/register-tokens/

## Open it

```bash
open index.html                               # from disk
python3 -m http.server 8000 --bind 0.0.0.0    # or served locally
```

`index.html`, `styles.css` (the Visual Learn house style plus page rules) and
`main.js`. No build step and no third-party scripts; the only external
resource is Google Fonts, with system fallbacks.

## Figures

The page is static. Every comparison the old interactive widgets hid behind a
toggle or slider is drawn as small multiples on a shared scale.

| Figure | What it shows |
|---|---|
| 1 | Input image beside last-layer `[CLS]` attention maps with 0 and 4 registers, one color scale. The caption computes the share of attention the artifacts take and the bird's share with and without registers. |
| 2 | The 14 × 14 patch grid and the token strip it becomes: `[CLS]` at position 0, then rows 0, 1, 6 and 13 in row-major order. Four patches carry the same position number in the grid and the strip. |
| 3 | Last-layer norm map (grey up to 150, orange above) beside a dot plot of the same 196 norms. The caption states the two groups' ranges, computed from the data. |
| 4 | Norm maps at 1/4, 1/2, 3/4 and full depth for DINOv2 ViT-S, ViT-B and ViT-L, each with its artifact count. |
| 5 | Linear-probe results for normal and artifact tokens (paper Fig. 5b and Table 1). |
| 6 | Diagram of a ViT-L with four registers: 201 tokens in, register outputs discarded. |
| 7 | Dot plots of every output norm for models with 0, 1 and 4 registers: `[CLS]`, registers and patch tokens on one axis. |
| 8 | Downstream results with and without four registers (paper Tables 2a and 3), with ▲/▼/= marks for better, worse and unchanged. |

Colors are fixed across the page: `[CLS]` purple, registers green, artifact
tokens orange, ordinary patch tokens grey. Attention uses a blue ramp and
norms a grey ramp.

## Accuracy caveats

- **Every map and dot plot is a simulation.** A small deterministic model in
  `main.js` reproduces the paper's qualitative findings: artifacts from the
  middle layers of DINOv2 ViT-L (onset at layers 9 to 13 of 24, in proportion
  to the paper's layer 15 of 40 for ViT-g), none in DINOv2 ViT-S or ViT-B,
  only in redundant background patches, and none once one or more registers
  are present. Each caption on the page says "simulated". The norms are
  illustrative, not measured.
- **One register removes the artifacts** in the simulation. This matches the
  paper's Fig. 8, which reports that visible artifacts disappear with at least
  one register; the paper uses four for its main results.
- **Grid size.** The simulation uses a 14 × 14 grid (a 224-pixel image with
  16-pixel patches) everywhere, including where it is labeled DINOv2, whose
  14-pixel patches give 16 × 16 at 224 pixels. The page says so in a sidenote.
- **The size threshold is recipe-dependent.** The paper finds artifacts at
  ViT-B for DeiT-III and OpenCLIP, which the DINOv2 size figure does not
  model; the page says so in the prose.
- **Tables** (probe results and downstream results) are transcribed from the
  ICLR 2024 version (arXiv v2, April 2024) and asserted by the tests.
- **Register order** follows the official DINOv2 implementation
  (`prepare_tokens_with_masks`): `[CLS]`, registers, then patch tokens, with
  registers inserted after the positional embeddings. The paper's text says
  "appended"; the page notes both.
- The 2024 method trains models with registers. The page separately describes
  a 2025 paper that edits activations of existing models to get a
  training-free register-like effect.

## Tests

```bash
pnpm install --frozen-lockfile
pnpm test
```

Playwright drives `index.html` in headless Chrome and checks, at 1440 px and
390 px: no console or page errors, no horizontal overflow, every figure
drawn, measured SVGs drawn at their displayed width, and no chart text below
11 px. It also checks that figures redraw when the viewport shrinks; that the
simulation shows what the prose claims (artifacts only without registers,
only in sky patches, only in ViT-L from mid-depth, absorbed by one register);
that computed captions and labels match the drawn data (position numbers
equal 1 + 14 × row + column, artifact counts, norm ranges); that the register
order is `[CLS]`, registers, patches; that both tables match the paper and
regressions carry ▼; that the top bar tracks the current section; and that
every source link is still present.
