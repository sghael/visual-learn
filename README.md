# Visual Learn

Interactive explanations of machine-learning systems, with diagrams, worked
examples, and browser experiments. Inspired by the [Polo Club](https://poloclub.github.io/) explainers.

Live site: https://sghael.github.io/visual-learn/

## Explainers

| Folder | Explainer | What it covers |
|---|---|---|
| [`jax-vs-pytorch/`](jax-vs-pytorch/) | [JAX and PyTorch](https://sghael.github.io/visual-learn/jax-vs-pytorch/) | How JAX, XLA and TPUs run your code, side by side with PyTorch and CUDA |
| [`qat-explainer/`](qat-explainer/) | [Quantization-aware training](https://sghael.github.io/visual-learn/qat-explainer/) | Training with low-precision weights: rounding and clipping error, the straight-through estimator, and a PTQ versus QAT lab that trains in the browser |
| [`gcp-accelerators/`](gcp-accelerators/) | [Google Cloud accelerators](https://sghael.github.io/visual-learn/gcp-accelerators/) | GPUs and TPUs on Google Cloud: how they multiply matrices, the lineup, the roofline, interconnects, and a chip-count estimate |
| [`register-tokens/`](register-tokens/) | [Register tokens](https://sghael.github.io/visual-learn/register-tokens/) | High-norm patch tokens in vision transformers and the effect of adding register tokens |
| [`rl-agentic-llms/`](rl-agentic-llms/) | [Reinforcement learning for LLM agents](https://sghael.github.io/visual-learn/rl-agentic-llms/) | Policy gradients, GRPO, reward models and KL penalties, verifiable rewards, tool-use training, and reward hacking |
| [`k2-horizon/`](k2-horizon/) | [K2 Horizon](https://sghael.github.io/visual-learn/k2-horizon/) | Model configs, mixture-of-experts and MoVA layers, Uno diffusion drafting, training stages, and reported benchmarks |
| [`pareto-front/`](pareto-front/) | [Intelligence versus cost](https://sghael.github.io/visual-learn/pareto-front/) | The Pareto frontier of Intelligence Index score against cost per task, and how it filled in quarter by quarter |

## How the repo works

- One explainer per top-level folder, fully self-contained: its own
  `index.html`, CSS and JS, no build step, no framework, no code shared between
  folders. Open any folder from disk or serve it locally.
- The root `index.html` is the landing page that links the explainers.
- Every explainer follows the [house style](house-style/): Tufte-style
  figures, sidenotes, one shared stylesheet copied into each folder, and plain
  writing.
- Pushing to `main` publishes the whole repo through GitHub Pages, so a folder
  named `foo/` is live at `https://sghael.github.io/visual-learn/foo/` with no
  per-folder deploy step.
- A folder that ships tests has a `package.json` with a `test` script. CI
  discovers those folders automatically and runs each one in headless Chrome.

```bash
cd jax-vs-pytorch
pnpm install
pnpm test      # Playwright against the installed Chrome
pnpm serve     # http://0.0.0.0:8000/
```
