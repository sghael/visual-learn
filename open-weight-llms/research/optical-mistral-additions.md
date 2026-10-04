# Optical compression and formal proof agents — research additions

Research cutoff: 2026-10-03. These are additions to the DeepSeek/Mistral lessons, selected for distinct mechanisms rather than benchmark rank. Both lessons contain 673 prose words, five sections, multiple figures and an interactive exercise. All diagrams are explicitly simplified or illustrative.

## DeepSeek-OCR — 2025-10-20

The [official repository](https://github.com/deepseek-ai/DeepSeek-OCR) dates its public model release October 20, 2025 and links downloadable weights. The [model card](https://huggingface.co/deepseek-ai/DeepSeek-OCR) labels the weights MIT. The paper's October 21 publication date is a different event. The lesson uses the earlier weight-release date.

Read the full [paper HTML](https://arxiv.org/html/2510.18234v1), including DeepEncoder, resolution/token settings, compression experiments and limitations; the repository's supported modes and prompts; and the model card. The tutorial attributes inherited components and keeps the model-specific summary concise.

The local worked examples are original teaching constructions: a hypothetical 1,000-to-100 input-position comparison; an abstract 16-by-12 grid; and a table whose characters survive while row associations do not. These must not become claimed benchmark outcomes. The patch widget counts N² dense attention pairs, whereas the actual encoder includes local attention and learned compression. Input-position reduction is not byte compression, losslessness, an end-to-end latency measurement, or a guarantee about output-token cost.

Read the independent primary research [Optical Context Compression Is Just (Bad) Autoencoding](https://arxiv.org/html/2512.03643v1). The lesson summarizes its reconstruction-versus-language-modeling result in a short, explicitly setup-dependent paragraph. It does not treat that result as a universal rejection of visual representations. The original OCR paper itself distinguishes OCR reconstruction from future general long-context validation.

The official repository also records DeepSeek-OCR2 on January 27, 2026. It was not added as a separate lesson in this increment: the assigned addition is the first optical-compression mechanism, and the lesson makes no claim to cover OCR2's altered encoder.

## Leanstral / 1.5 — 2026-03-16

The [March release post](https://mistral.ai/news/leanstral/) explicitly announces Apache 2.0 public weights on March 16, 2026. The [2603 model card](https://huggingface.co/mistralai/Leanstral-2603) specifies 119B total and 6.5B active parameters; the lesson uses those precise values rather than the blog's rounded 120B/6B shorthand. Its [documentation](https://docs.mistral.ai/models/leanstral-26-03) provides the same release date. API endpoint retirement is not weight withdrawal; the lesson promises no current hosted endpoint.

The [July 2 update post](https://mistral.ai/news/leanstral-1-5/) links the [technical report](https://github.com/mistralai/LeanstralSafeVerify/blob/main/LeanstralReport.pdf). The PDF was read in full after downloading the public raw file and extracting its text with the bundled Python PDF reader. Read sections on training, LeanGym, verification, CISPO, context compaction, inference evaluations and future work. The lesson labels these as **1.5** details; it does not backdate the later training recipe to March.

The distinguishing contribution is formal repository agent training with tool feedback and verifiable task outcomes. The local mathematical examples explain statement specification before discussing training. [Lean's proof-validation reference](https://lean-lang.org/doc/reference/latest/ValidatingProofs/) and [axiom reference](https://lean-lang.org/doc/reference/latest/Axioms/) establish the checker's boundaries. In particular, compilation with an unfinished proof placeholder is not sufficient, and accepted assumptions remain part of what a theorem means.

The lesson distinguishes current inference from future multi-agent work. Its scheduler and success-probability example are explicitly hypothetical; neither measures Leanstral throughput or represents a communicating agent swarm.

## Mistral scan and exclusions

Scanned Mistral's official news and model documentation through the cutoff. Leanstral adds a distinct lesson about formal proof-engineering feedback beyond the existing MoE and multimodal lessons.

- [Shieldstral, August 4, 2026](https://mistral.ai/news/shieldstral/) is a verified Apache 2.0 3B open-weight classifier. Read the announcement and mechanism: policy-conditioned binary questions with yes/no-logit normalization. Excluded from this increment because the bounded request is for one additional Mistral lesson and Leanstral develops the generative reasoning theme more directly. This is a scope decision, not a claim that Shieldstral has no contribution.
- [August 11 infrastructure announcement](https://mistral.ai/news/regional-inference-open-models-new-compute/) concerns regional inference, access to third-party models and compute capacity. It is not a new model architecture and is excluded.
- General Mistral multimodal releases found during discovery were not automatically given lessons merely for being newer. This memo does not claim Leanstral is Mistral's newest model overall.

No source artwork is copied. Model-specific prose is distributed across the primary sources and kept concise; original examples and widget arithmetic carry the explanation. No headline leaderboard values are used as timeless model-quality statements.
