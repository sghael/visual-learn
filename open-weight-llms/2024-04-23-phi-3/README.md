# Phi-3 · mini

How choosing the curriculum can matter as much as increasing the model

Open `index.html` directly in a browser, or follow this folder from the collection index. All runtime CSS and JavaScript are local; there is no build step. Google Fonts are optional and have system fallbacks.

## What the tutorial teaches

Phi-3-mini emphasizes educationally filtered and synthetic data to develop a useful model with few stored weights.

- Choose the student before choosing the curriculum
- A worked example supplies intermediate structure
- Phi-3 uses both filtering and synthesis
- A compact model changes what can be stored locally
- Test the curriculum’s blind spots

## Figures

- **Turn raw material into useful practice** (`flow`): Simplified conceptual curriculum inspired by the Phi research direction; this is not Microsoft’s complete internal data pipeline. Filtering chooses the material; generation changes how a skill is demonstrated.
- **Count the weight bytes** (`precision`): Illustrative ideal storage for 3.8B weights. Excludes quantization metadata, mixed precision, activations, KV cache and runtime overhead; no accuracy or phone-fit claim. Small parameter counts and fewer bits reduce different factors in the same product.

## Accuracy and scope

Release date: 2024-04-23. Phi-3-mini public weights were announced April 23, 2024. Later small, medium and 3.5 variants are not assigned this date.

- The lesson focuses on mini, the first public Phi-3 release.
- Educational-data selection does not prove a universal optimum or complete factual knowledge.
- The memory calculator models ideal weights only.

The page includes primary source links next to claims and a source list. Research cutoff: 2026-10-03. Quantitative widgets are explicitly simplified teaching models, not executed model inference or performance benchmarks.

## Maintenance

Edit `lesson.json`; run `pnpm author` from the collection directory to refresh committed static files. Run `pnpm test` there. The optional authoring utility is not required to view or deploy this page.
