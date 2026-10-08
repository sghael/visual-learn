# Gemma 2 · 9B and 27B

How a larger teacher model gave a smaller student more to learn from each token

Open `index.html` directly in a browser, or follow this folder from the collection index. All runtime CSS and JavaScript are local; there is no build step. Google Fonts are optional and have system fallbacks.

## What the tutorial teaches

Gemma 2 trained its 9B model on a larger teacher’s next-token probabilities instead of only the observed text, and alternated local and global attention layers to limit memory.

- What one observed word leaves out
- Softening the teacher’s prediction
- Why matching the whole distribution helps
- Alternating local and global attention
- Two meanings of “distillation”

## Figures

- **See the alternatives in a teacher’s prediction** (`distill`): Made-up scores for four candidate words. The figure shows distillation from token probabilities; it does not reproduce Gemma 2’s training settings. A soft target shows how the alternatives compare, not only which one won.
- **Compare a local layer with a global layer** (`attention`): Simplified attention pattern for one layer, with 16 tokens. Local and Full show Gemma 2’s two layer types; “Toy hybrid” is a generic teaching pattern, not Gemma 2’s layer order. Local layers limit what each layer sees directly; the global layers between them keep long-range access.

## Accuracy and scope

Release date: 2024-06-27. Google released the 9B and 27B weights on June 27, 2024. The date excludes the earlier preview and the 2B model, which came later.

- The release date covers the 9B and 27B models; the 2B model came later.
- The temperature figure uses made-up scores and does not reproduce Gemma 2’s training settings.
- The attention figure is a scaled-down illustration, not a trace of the model running.

The page includes primary source links next to claims and a source list. Research cutoff: 2026-10-03. Quantitative widgets are explicitly simplified teaching models, not executed model inference or performance benchmarks.

## Maintenance

Edit `lesson.json`; run `pnpm author` from the collection directory to refresh committed static files. Run `pnpm test` there. The optional authoring utility is not required to view or deploy this page.
