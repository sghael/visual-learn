# LLaDA 8B

How to generate text by repeatedly filling blanks

Open `index.html` directly in a browser, or follow this folder from the collection index. All runtime CSS and JavaScript are local; there is no build step. Google Fonts are optional and have system fallbacks.

## What the tutorial teaches

LLaDA generates text by filling in a reply made of blanks over several rounds instead of writing left to right, and shows why that is not automatically faster.

- Training by filling in blanks
- Generating several tokens per round
- Why parallel prediction is not automatically faster
- What LLaDA showed, and when it appeared

## Figures

- **Hide part of a sequence; learn to restore it** (`flow`): Simplified training outline. The real loss also weights each example by its masking fraction, as described above. Training on blanks teaches the model to use context on both sides of a gap.
- **Fill several blanks, then use them as context** (`diffusion`): Scripted, simplified unmasking. It does not run LLaDA or model its confidence. The order in which words are generated can differ from the order in which they are read.

## Accuracy and scope

Release date: 2025-02-20. February 20, 2025 is the first recorded commit of the Instruct weights. The repository later dated the work February 14; when the files became public is uncertain.

- The release date is the first recorded commit containing weights, not an independently verified public announcement.
- The sentence and unmasking order in the figure are scripted by hand, not produced by a model.
- The original LLaDA release did not include its full training data or training framework, only the weights and the code to run them.

The page includes primary source links next to claims and a source list. Research cutoff: 2026-10-03. Quantitative widgets are explicitly simplified teaching models, not executed model inference or performance benchmarks.

## Maintenance

Edit `lesson.json`; run `pnpm author` from the collection directory to refresh committed static files. Run `pnpm test` there. The optional authoring utility is not required to view or deploy this page.
