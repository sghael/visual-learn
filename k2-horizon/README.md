# K2 Horizon

A guide to the six K2 Horizon language models that MBZUAI's Institute of
Foundation Models (IFM) released on 3 September 2026. It explains how the six
differ (read from each model's `config.json`), what sparse Mixture-of-Experts
layers and IFM's Mixture-of-Value Attention (MoVA) store and compute, how the
Uno adapters draft and verify several tokens per round, how the 375B model was
trained, what IFM's reported benchmark numbers show, and which promised release
artifacts were available on 25 September 2026.

## Open it

The page is self-contained: `index.html`, `styles.css` and `k2.js`, no build
step and no third-party scripts. Double-click `index.html`, or serve the folder:

```bash
python3 -m http.server 8000 --bind 0.0.0.0
```

Fonts load from Google Fonts and fall back to system serif and sans faces
offline.

## Figures

| # | Figure | What it shows | Interaction |
|---|---|---|---|
| 1 | Stored versus active parameters | All six models on one log scale: grey ring for stored parameters, orange dot for parameters used per token | none |
| 2 | Configuration table | Layers, width, heads, experts, vocabulary and context from each `config.json`, plus stored tensor counts from the Hugging Face API | none |
| 3 | Three layer types, small multiples | Dense, MoE (375B-A23B) and MoE + MoVA (36B-A4B) side by side; every expert drawn as a square, the ones a token uses in orange | none |
| 4 | 36B-A4B parameter ledger | Stored and used parameters by component: routed FFN experts, value experts, attention, shared and dense FFNs | none |
| 5 | Uno decoding, simulated | The same sentence written by autoregressive decoding and by Uno's draft-and-verify rounds, with tokens against forward passes below | Play/Pause, Step, Reset, block size (4, 8, 16), draft agreement slider |
| 6 | 7B-Uno efficiency table | Tokens per pass, system and per-request throughput, and two accuracy rows from the 7B-Uno card | none |
| 7 | 375B training stages | Every stage's tokens (with bars on one linear scale), steps and sequence length | none |
| 8 | Reported benchmark scores | Dot plot on one 0–100 scale (GDPVal-AA Elo on its own axis): K2 in blue, each comparison model in grey, K2 versus the best other score as text; a disclosure holds the full table | model selector |
| 9 | Release artifacts | Available versus announced artifacts, checked 25 September 2026 | none |

The Uno figure rests on a finished run at the default settings (block 4, 70%
agreement), so the key comparison is visible before any interaction. Changing
either setting stops any run and redraws a finished run for the new setting.

## How the numbers were derived

- **Configurations** come from `config.json` in each repository under
  <https://huggingface.co/IFM>, read 25 September 2026. Stored parameter counts
  come from the Hugging Face API (`safetensors.total`).
- **Model names** count parameters outside the vocabulary tables. For example
  the 3.7B checkpoint stores 5.06B parameters, of which 2 × 250,624 × 2,560 =
  1.28B are the input embedding and output matrices.
- **The 36B-A4B ledger** (Figure 4) is computed from `config.json` and
  `modeling_k2_horizon.py`: 45 sparse layers (48 minus `mlp_only_layers`
  0–2), 100 SwiGLU experts of 3 × 2,560 × 768 each, 64 value experts of
  2,560 × 1,024 each, Q/K/O and a softplus output gate in every layer, one
  shared expert, and dense FFNs of width 6,144 in the first three layers.
  Adding routers, norms and the two 250,624 × 2,560 vocabulary matrices gives
  37,444,792,020 parameters, which equals the checkpoint's tensor count.
  About 4.7B non-embedding parameters run per token; IFM describes the model
  as activating approximately 4B.
- **Benchmarks** are transcribed from the "Full results" tables on IFM's
  technical blog (<https://ifm.ai/blog/k2/>), which use one fixed comparison
  set per model. The model cards differ in places: the 7B card chooses
  reference models row by row, and the 32B card labels its checkpoint
  "Stage1".

## Accuracy caveats

- Every benchmark number is self-reported by IFM; IFM also chose the
  comparison models. For the 32B and 36B-A4B, IFM says the baseline scores come
  from Artificial Analysis. The page presents them as IFM's comparison, not an
  independent ranking.
- The 375B Terminal-Bench 2.1 row plots the 70.2% on the model card, because
  the comparison scores were not audited the same way, and marks IFM's
  reward-hacking-audited 66.9% beside it.
- MoVA is described from the released modeling code. IFM's technical report was
  due at the end of September 2026 and was not available when this page was
  written. Which experts light up in Figure 3 is illustrative.
- **The Uno figure is a simplified simulation.** Each word stands for one
  token. The "draft agreement" slider is an invented probability that each
  drafted token after the first passes verification. The simulation follows the
  paper's accounting (two forward passes per round; the first drafted token is
  always accepted; one token sampled by the base model per round), so tokens
  per pass stay between 1 and (B + 1)/2. It ignores adapter overhead, batching
  and hardware utilization, and treats a pass over a block as costing the same
  as a one-token pass.
- The 7B-Uno numbers in Figure 6 are from the model card. The paper's own
  Table 1 reports slightly different accuracies for some rows (for example
  68.4% versus 70.1% on SWE-bench Verified); the page uses the card.
- The training table is for the 375B-A23B only. Its RL stage has no published
  token or step count. The smaller models' cards describe different recipes.
- Release status changes quickly. The table records what the model cards, the
  `ifm-ai` GitHub organization and the IFM Hugging Face organization showed on
  25 September 2026.

## Tests

Playwright drives the page in headless Chrome (the installed Google Chrome, or
Playwright's own Chromium as a fallback). The tests check that every figure
renders at 1440 px and 390 px with no console or page errors, no horizontal
overflow and no chart text under 11 px; that the configuration table and layer
multiples carry the configured values; that the ledger labels match the prose;
that the Uno simulation rests on a finished run, stops cleanly on Reset, Pause
and a mid-run block-size switch (with a fake clock), and never leaves the
paper's 1 ≤ tokens-per-pass ≤ (B + 1)/2 range; that every benchmark "best
other" label equals the maximum in the data table; and that the controls are
keyboard reachable.

```bash
pnpm install --frozen-lockfile
pnpm exec playwright install chromium   # only if Google Chrome is not installed
pnpm test
```
