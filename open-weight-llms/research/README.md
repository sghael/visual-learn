# Research method and coverage

Research cutoff: **October 3, 2026**. The collection covers roughly three years of open-weight language-model development, with Llama 2 and Mistral 7B included as nearby 2023 foundations. It selects releases for distinct, documented mechanisms rather than attempting a checkpoint census. Closely related model sizes can share a tutorial; each named tutorial has a dated subfolder.

## How claims were checked

1. Discover releases through official model-family announcements, papers, and model repositories.
2. Open and read the relevant paper sections and official release material. A search excerpt is a discovery aid, not the final evidence.
3. Verify public-weight availability, named variants, and license provenance. Separate an announcement date from a paper date and a repository staging date. When visibility history cannot be recovered, disclose the uncertainty beside the date.
4. Explain inherited techniques separately from a release's contribution. Parameter counts, activation counts, precision, cache storage, and measured latency are different quantities.
5. Build a small, transparent calculation around the mechanism. Label teaching dimensions and scripted data directly on the page. Do not turn them into predicted accuracy or hardware speedups.
6. Link model-specific claims to primary sources in each tutorial. Source lists say which portions establish the claims; the memos below record qualifications and omissions.

## Research memos

- [Llama, Gemma, Phi](llama-gemma-phi.md)
- [DeepSeek, Mistral, Mixtral](deepseek-mistral.md)
- [Kimi, GLM, Qwen](kimi-glm-qwen.md)
- [OpenAI, Mamba, Tülu, Olmo, LLaDA, BitNet](additional-families.md)
- [Mamba-3 and Falcon-H1](mamba3-falcon.md)
- [Optical compression and Mistral additions](optical-mistral-additions.md)

## What the collection does not claim

This is not an exhaustive list of every public model, fine-tune, modality, quantization, or unpublished training innovation. It concentrates on language-model mechanisms: data and post-training, attention and state, sparsity, numerical representation, reasoning and tool use, multimodal context, and reproducibility. Adjacent image/video generation, embedding-only models, safety classifiers, and minor checkpoint updates are outside this collection.

Coverage is historical, not a recommendation that a particular model is currently best. The collection does not combine benchmarks from incompatible evaluation setups into a league table. Where the full report or a dynamic official page could not be retrieved, the corresponding memo identifies the limitation and the actual primary evidence used.

## Reproducibility of the tutorials

The committed `index.html`, `styles.css`, and `tutorial.js` in each dated folder are the delivered tutorial. They run without a build and without model downloads. Optional web fonts have local fallbacks. `lesson.json` contains the authored prose and figure configurations; `authoring/` maintains the static-page generator and browser code. The test suite checks browser behavior, formulas, navigation, and narrow layouts. It does not validate model benchmark claims by running the models.
