# Reinforcement learning for language-model agents

An explainer on how reinforcement learning trains language models and
tool-using agents: where RL sits in training, policy gradients and PPO's
clipped objective, GRPO's group-normalized advantages, reward models learned
from preferences, the KL penalty, verifiable rewards, credit assignment in a
multi-turn coding episode, reward hacking, and a sourced table of what AI
laboratories have published through mid-2026.

Ten sections, about 30 minutes. Part of
[Visual Learn](https://sghael.github.io/visual-learn/); it follows the
collection's house style (Source Serif 4 / Source Sans 3 / IBM Plex Mono,
margin notes and captions, one light theme).

## Open it

A single standalone page with no build step and no script dependencies.
Double-click `index.html`, or serve the folder:

```bash
python3 -m http.server 8000 --bind 0.0.0.0
```

Fonts load from Google Fonts with system fallbacks. Every chart is drawn in
plain SVG at the container's CSS width, so labels keep their size on a phone.
Nothing animates.

## Figures

| # | Figure | What it shows | Interaction |
|---|---|---|---|
| 1 | The loop (`loop`) | Classic RL and a language-model agent as two side-by-side diagrams, each with one episode drawn as a strip: rewards at every step versus one reward at the end | Static small multiples |
| 2 | Policy-gradient update (`policy-gradient`) | A five-answer softmax policy; each update applies the exact gradient with advantage = reward − baseline, and a tick marks the previous probabilities | Pick the sampled answer and its reward, apply updates, reset |
| 3 | PPO clipped objective (`ppo-clip`) | The clipped objective against the ratio r for positive and negative advantage, ε = 0.2 | Static small multiples |
| 4 | GRPO group (`grpo`) | Six answers to 17 × 24 with verifier rewards, group mean, population std and per-response advantages | Swap any response for its correct or wrong version; toggle the format reward; reset |
| 5 | GRPO advantage by solve count (`grpo-curve`) | Advantage of each correct and each wrong response for 0–6 correct of 6 | Static |
| 6 | Bradley–Terry reward model (`rlhf`, `rlhf-curve`) | Two scores per pair trained by clicking preferences; the current point on the logistic curve with earlier positions in grey | Prefer A or B, next pair, reset |
| 7 | KL-regularized optimum (`kl`) | The exact optimum π_ref · exp(r/β) / Z over ten tokens, with expected reward, KL and objective | One β slider, log scale 0.05–5 |
| 8 | Coding-agent trajectory (`trajectory`) | Model turns (trained), tool output (masked) and the hidden grader, with the advantage each model turn receives | Pass/fail outcome; outcome-only versus per-turn shaping |
| 9 | Proxy versus true reward (`goodhart`) | Three graders of increasing quality, simulated with the functional form from Gao, Schulman and Hilton (2022) | Static small multiples |

Tables: training phases, the RL vocabulary in both settings, agent reward
sources, published methods by organization (`labs`), and algorithms in
chronological order.

## Accuracy caveats

- Figures 3, 4, 5 and 7 evaluate the published formulas exactly. Figure 7 uses
  the closed-form optimum of the KL-regularized objective (equation 4 of the
  DPO paper) on a made-up ten-token distribution.
- Figure 2 is simplified: the baseline is an exponential moving average of
  past rewards (weight 0.5), where real systems use a critic or group
  statistics. Figure 6 is simplified: the reward model has one free score per
  response instead of a network.
- Figure 9 is a simulation. It uses the RL functional form Gao et al. fit to
  their experiments, R(d) = d(α − β ln d) with d = √KL, but the coefficients
  are chosen for illustration.
- Figure 8's token counts are approximate and its baseline (0.45) is an
  assumed group solve rate. The design (outcome reward, masked tool output) is
  representative, not any one lab's recipe.
- The published-methods table summarizes public statements through mid-2026
  and links each source. Complete recipes and compute allocations are not
  public. Rows without a date or link restate claims from the earlier version
  of this page that had none.
- The R1-Zero AIME figures (15.6% to 71.0% pass@1) are from section 2.2.4 of
  the DeepSeek-R1 paper.

## Tests

Playwright drives the page in headless Chrome (the installed Google Chrome, or
Playwright's own Chromium) and checks: every figure renders at 1440 px and
390 px with no console or page errors, no horizontal overflow, and no chart
label under 11 px; the top bar's links fit at 1440 px and track the current
section; GRPO's zero-variance case, the exact ±1 case, and that every
verdict and reward matches its text; keyboard operation of the verdict
buttons and the β slider; a policy-gradient update and the shrinking second
update; the trajectory's sign flip, masking and shaping; the reward model's
shrinking steps and reset; the KL readouts against the exact solution; that
every organization links a source and no source link from the previous
version was dropped; and that the page loads no external scripts.

```bash
pnpm install --frozen-lockfile
pnpm test
```
