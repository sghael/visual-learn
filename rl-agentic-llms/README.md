# Agentic RL Explainer

An interactive, visual tutorial on how reinforcement learning turns a language
model into an agent: policy gradients, PPO, GRPO, RLHF, verifiable rewards,
multi-turn tool-use training, reward hacking, and what the frontier labs have
said in public about their RL work through mid-2026. Built in the spirit of the
[Polo Club](https://poloclub.github.io/) explainers.

## Open it

It is a single standalone page. Either double-click `index.html`, or serve the
folder:

```bash
python3 -m http.server 8000 --bind 0.0.0.0
```

No build step. D3 v7 loads from cdnjs (pinned) and fonts from Google Fonts,
both with graceful fallbacks; the page needs network for D3. Animations respect
`prefers-reduced-motion`: the hero shows its resting state, the episode runner
and the training run render their finished state immediately.

## What is inside

| Section | Widget | What it shows |
|---|---|---|
| Hero | `hero` | A reference distribution over 12 tokens and a policy that sharpens toward the rewarded token, with live KL |
| 01 Pipeline | `pipeline` | Pretraining, SFT, preference RL and verifiable/agentic RL as clickable stages with an illustrative compute share |
| 02 The loop | `loop` | The agent/environment loop, toggled between classic RL and the LLM mapping; a run animates one episode and the growing context |
| 03 Policy gradient | `policy-gradient`, `ppo-clip` | A five-action softmax policy you update by hand with reward and learning-rate sliders; the PPO clipped objective versus the ratio |
| 04 GRPO | `grpo` | Six rollouts to one prompt; toggle correctness and a format bonus to watch group-normalized advantages, including the zero-variance case |
| 05 RLHF | `rlhf`, `kl` | A Bradley–Terry reward model trained by clicking preferences; a KL-leash view with optimization pressure and β |
| 06 Verifiable rewards | `rlvr` | An animated, illustrative R1-Zero-style run: accuracy and response length over RL steps, with a length-penalty slider |
| 07 Agentic RL | `trajectory`, `reward-sources` | A coding-agent episode with loss-masked tool output; flip the outcome or add per-turn shaping to see the credit assignment |
| 08 Reward hacking | `goodhart` | Proxy versus true reward under optimization, with a grader-robustness slider |
| 09 Frontier labs | `labs`, `timeline` | Filterable cards per lab with primary-source links, and a keyboard-reachable timeline from REINFORCE to 2026 |
| 10 Scaling | `scaling` | Illustrative RL share of training compute by model generation |
| 11–12 | static | Open problems, glossary, primary sources |

## Accuracy caveats

- Every chart labeled **illustrative** on the page is hand-shaped to show the
  concept and is not a measurement: the compute shares in sections 01 and 10,
  the RLVR training curves, the reward-source mix, and the Goodhart curves.
- The GRPO advantage uses the population standard deviation of the group. The
  toy policy-gradient widget uses a running-mean baseline; real systems use a
  critic or group statistics.
- The KL widget's "optimization pressure" is a single exponential tilt of the
  reference distribution, softened by β for intuition. It is not a simulation
  of an RL run.
- The lab cards summarize public statements through mid-2026 and link their
  primary sources at the bottom of each card. Where a claim is an inference
  from those sources rather than something a lab stated, the card says so.
  Frontier labs disclose little; treat each card as partial.
- Dates are given to the month where the source is public; later releases are
  described generically ("and successors").

## Tests

The page has no runtime dependencies. The tests do: Playwright drives the page
in headless Chrome (the installed Google Chrome, or Playwright's own Chromium)
at 1440 px and 390 px and checks for page errors, console errors, horizontal
overflow, and the behaviors that are easy to break: zero-variance GRPO groups,
a policy-gradient update, flipping the trajectory outcome, reset during the
RLVR run, and the reward-model click loop.

```bash
pnpm install
pnpm test
```
