# Visual Learn

Interactive explanations of machine-learning systems, with diagrams, worked
examples, and browser experiments. Inspired by the [Polo Club](https://poloclub.github.io/) explainers.

Live site: https://sghael.github.io/visual-learn/

## Explainers

| Folder | Explainer | What it covers |
|---|---|---|
| [`jax-vs-pytorch/`](jax-vs-pytorch/) | [JAX and PyTorch](https://sghael.github.io/visual-learn/jax-vs-pytorch/) | How JAX, XLA and TPUs run your code, side by side with PyTorch and CUDA |
| [`qat-explainer/`](qat-explainer/) | [Quantization-Aware Training](https://sghael.github.io/visual-learn/qat-explainer/) | Training with low-precision weights: fake quantization, the straight-through estimator, and a live PTQ vs QAT lab |
| [`gcp-accelerators/`](gcp-accelerators/) | [Google Cloud Accelerators](https://sghael.github.io/visual-learn/gcp-accelerators/) | GPU and TPU hardware on Google Cloud: anatomy, lineup, roofline, interconnects, and a model-fit calculator |
| [`register-tokens/`](register-tokens/) | [Register Tokens Explainer](https://sghael.github.io/visual-learn/register-tokens/) | High-norm patch tokens in vision transformers and the effect of adding register tokens |
| [`rl-agentic-llms/`](rl-agentic-llms/) | [Reinforcement Learning for LLM Agents](https://sghael.github.io/visual-learn/rl-agentic-llms/) | Policy gradients, preference learning, verifiable rewards, and tool-use training |
| [`k2-horizon/`](k2-horizon/) | [K2 Horizon](https://sghael.github.io/visual-learn/k2-horizon/) | Model sizes, mixture-of-experts layers, attention, diffusion decoding, and reported benchmarks |
| [`pareto-front/`](pareto-front/) | [Intelligence versus cost](https://sghael.github.io/visual-learn/pareto-front/) | Model release dates and the plotted Intelligence Index versus task cost frontier |

## How the repo works

- One explainer per top-level folder, fully self-contained: its own
  `index.html`, CSS and JS, no build step, no framework, no code shared between
  folders. Open any folder from disk or serve it locally.
- The root `index.html` is the landing page that links the explainers.
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
