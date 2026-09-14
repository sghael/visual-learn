# Agentic RL Explainer

An interactive explanation of reinforcement learning for language models:
policy gradients, PPO, GRPO, RLHF, verifiable rewards, multi-turn tool-use
training, reward hacking, and publicly documented training methods through
mid-2026. Built in the spirit of the [Polo Club](https://poloclub.github.io/)
explainers.

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
| 01 Pipeline | `pipeline` | Pretraining, supervised fine-tuning, preference training, and outcome-based RL as clickable stages |
| 02 The loop | `loop` | The agent/environment loop, toggled between classic RL and the LLM mapping; a run animates one episode and the growing context |
| 03 Policy gradient | `policy-gradient`, `ppo-clip` | A five-action softmax policy you update by hand with reward and learning-rate sliders; the PPO clipped objective versus the ratio |
| 04 GRPO | `grpo` | Six rollouts to one prompt; toggle correctness and a format bonus to watch group-normalized advantages, including the zero-variance case |
| 05 RLHF | `rlhf`, `kl` | A Bradley–Terry reward model trained by clicking preferences; a simplified view of KL regularization |
| 06 Verifiable rewards | `rlvr` | An animated, illustrative R1-Zero-style run: accuracy and response length over RL steps, with a length-penalty slider |
| 07 Agentic RL | `trajectory`, `reward-sources` | A coding-agent episode with loss-masked tool output; flip the outcome or add per-turn shaping to see the credit assignment |
| 08 Reward hacking | `goodhart` | Proxy versus true reward under optimization, with a grader-robustness slider |
| 09 Published methods | `labs`, `timeline` | Filterable cards with primary-source links and a keyboard-reachable timeline from REINFORCE to 2026 |
| 10 Scaling | `scaling` | OpenAI's reported relative increase in RL training compute from o1 to o3/o4-mini |
| 11–12 | static | Open problems, glossary, primary sources |

## Accuracy caveats

- The RLVR training curves and Goodhart curves are hand-shaped teaching
  simulations, not measurements. The reward-source graphic lists possible
  signals without assigning percentages. The scaling chart shows a relative
  claim reported by OpenAI; it does not show absolute compute.
- The GRPO advantage uses the population standard deviation of the group. The
  toy policy-gradient widget uses a running-mean baseline; real systems use a
  critic or group statistics.
- The KL widget's "optimization pressure" is a single exponential tilt of the
  reference distribution, softened by β for intuition. It is not a simulation
  of an RL run.
- The organization cards summarize public statements through mid-2026 and
  link their sources. They are partial because complete training recipes and
  compute allocations are not public.
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
