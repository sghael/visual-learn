# Visual Learn

Interactive, single-page explainers on how machine-learning systems work, in
the spirit of the [Polo Club](https://poloclub.github.io/) explainers.

Live site: https://sghael.github.io/visual-learn/

## Explainers

| Folder | Explainer | What it covers |
|---|---|---|
| [`jax-vs-pytorch/`](jax-vs-pytorch/) | [JAX Explainer](https://sghael.github.io/visual-learn/jax-vs-pytorch/) | How JAX, XLA and TPUs run your code, side by side with PyTorch and CUDA |
| [`qat-explainer/`](qat-explainer/) | [Quantization-Aware Training](https://sghael.github.io/visual-learn/qat-explainer/) | How LLMs are trained to survive 4-bit weights: fake quantization, the straight-through estimator, and a live PTQ vs QAT lab |

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
