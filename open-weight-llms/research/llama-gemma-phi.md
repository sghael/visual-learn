# Llama, Gemma and Phi research memo

Research cutoff: **2026-10-03**. Ten tutorials, each with five sections, two or more figures, at least one interactive figure, and 717–778 prose words excluding figures and sources. This is a mechanism-oriented selection of releases, not a list of every checkpoint.

## Release-date evidence and scope

| Folder | Date evidence | Scope boundary |
|---|---|---|
| `2023-07-18-llama-2` | [Meta announcement](https://about.fb.com/news/2023/07/llama-2/) | Historical foundation just outside an exact three-year window. Original 7B/13B/70B release; no public 34B claim. |
| `2024-04-23-phi-3` | [Microsoft announcement](https://news.microsoft.com/source/features/ai/the-phi-3-small-language-models-with-big-potential/) | Mini only; later family variants not backdated. |
| `2024-06-27-gemma-2` | [Google announcement](https://blog.google/innovation-and-ai/technology/developers-tools/google-gemma-2/) | Original 9B/27B release; later 2B not backdated. |
| `2024-07-23-llama-3-1` | [Meta announcement](https://ai.meta.com/blog/meta-llama-3-1/) | Text 8B/70B/405B; separate from April Llama 3 release. |
| `2025-03-12-gemma-3` | [Google announcement](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-3/) | 1B text-only distinguished from larger image/text models. |
| `2025-04-05-llama-4` | [Meta announcement](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) and official model card | Scout/Maverick, excluding preview-only Behemoth. |
| `2025-04-30-phi-4-reasoning` | [Microsoft announcement](https://azure.microsoft.com/en-us/blog/one-year-of-phi-small-language-models-making-big-leaps-in-ai/) and official card | 14B reasoning and plus; prior Phi-4 covered as foundation, not assigned this date. |
| `2025-07-09-phi-4-mini-flash-reasoning` | [Microsoft announcement](https://azure.microsoft.com/en-us/blog/reasoning-reimagined-introducing-phi-4-mini-flash-reasoning/) | SambaY hybrid architecture, not the earlier mini-reasoning checkpoint. |
| `2026-03-04-phi-4-reasoning-vision` | [Microsoft post](https://www.microsoft.com/en-us/research/blog/phi-4-reasoning-vision-and-the-lessons-of-training-a-multimodal-reasoning-model/) and official card | Public 15B image/text model. |
| `2026-04-02-gemma-4` | [Google announcement](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/) | Family date; later [12B, June 3](https://blog.google/innovation-and-ai/technology/developers-tools/introducing-gemma-4-12b/) and [QAT, June 5](https://blog.google/innovation-and-ai/technology/developers-tools/quantization-aware-training-gemma-4/) additions are dated explicitly. |

## Papers and primary technical sources read

Each lesson contains the exact citation links and claim-specific anchors. The sections below record which parts were inspected. Research used official model-owner posts/model cards, paper HTML, and paper abstracts as discovery entrypoints. Search snippets alone were not the basis of lesson claims.

- **Llama 2:** [report](https://arxiv.org/html/2307.09288v2), Sections 2, 3.1–3.2, Appendix A.2.1; [GQA](https://arxiv.org/html/2305.13245v3), Section 2 and Figure 2; [official 70B card](https://huggingface.co/meta-llama/Llama-2-70b-hf). Anchors: original released variants, GQA applicability, chat-training pipeline. Widget uses explicitly invented dimensions rather than a false 70B architecture match.
- **Llama 3.1:** [report](https://arxiv.org/html/2407.21783v3), Table 1, Sections 3.1–3.4 and 4.1; [Chinchilla](https://arxiv.org/html/2203.15556v1), Sections 2–3; [DPO](https://arxiv.org/html/2305.18290v3), Sections 3–4. Anchors: data preparation, context curriculum, dense-model compute budgeting, post-training. Paper’s unreleased vision/speech experiments are explicitly excluded.
- **Llama 4:** [official card](https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct), model-information table and license; announcement’s architecture section; [sparse MoE foundational paper](https://arxiv.org/abs/1701.06538); [MetaCLIP](https://arxiv.org/html/2309.16671v1). Anchors: total/active counts, shared and routed branches, early fusion with a vision encoder. The abstract entrypoint is used only for generic prior-work attribution to sparse gating; model-specific topology comes from Meta.
- **Gemma 2:** [report](https://arxiv.org/html/2408.00118v3), Sections 2, 3.2 and 5; [distillation paper](https://arxiv.org/html/1503.02531v1), Section 2; [official card](https://huggingface.co/google/gemma-2-9b). Anchors: teacher-probability objective versus observed-token objective, local/global layers, chronology. The later report version includes 2B; the tutorial does not assign it the June launch date.
- **Gemma 3:** [report](https://arxiv.org/html/2503.19786v1), Sections 2, 2.1, 5.2–5.3; [SigLIP](https://arxiv.org/html/2303.15343v1), Sections 2–3; [official card](https://huggingface.co/google/gemma-3-4b-it). Anchors: local/global schedule, variant capabilities, visual representations and crop processing. Local/full masks are views of individual layer types, not the whole architecture.
- **Gemma 4:** [July report](https://arxiv.org/html/2607.02770v1), Sections 2, 2.3, 2.5–2.6; April launch, June 12B post and June QAT post above. Anchors: effective versus total accounting, 12B sensory projection, quantization preparation and drafter. The 12B architecture is not attributed to all variants. The paper postdates the releases, and its date is not used as the folder date.
- **Phi-3:** [report](https://arxiv.org/html/2404.14219v4), Sections 2 and 6; [Textbooks Are All You Need](https://arxiv.org/html/2306.11644v2), Section 2; [official mini card](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct). Anchors: curriculum design, filtering/synthesis distinction, mini architecture, scope of factual knowledge. The updated paper includes Phi-3.5 material; the mini tutorial does not backdate those features.
- **Phi-4 reasoning:** [reasoning report](https://arxiv.org/html/2504.21318v1), Sections 2–4; [base Phi-4 report](https://arxiv.org/html/2412.08905v1), Sections 2–4; [plus card](https://huggingface.co/microsoft/Phi-4-reasoning-plus). Anchors: teachable prompts, sequence demonstrations, actual reward ingredients, context/variant distinction. The GRPO widget is only arithmetic intuition and does not claim to reproduce Phi’s full objective.
- **Phi-mini-flash:** [SambaY](https://arxiv.org/html/2507.06607v1), Section 2 and Figure 1, inference discussion; [Mamba](https://arxiv.org/html/2312.00752v2), Section 3; [official card](https://huggingface.co/microsoft/Phi-4-mini-flash-reasoning). Anchors: state/readout distinction, shared global KV, GMUs, conditional throughput evaluation. The constant-state line is never labeled total Phi memory.
- **Phi vision:** [report](https://arxiv.org/html/2603.03975v1), Sections 2.1–2.3 and 4; [SigLIP 2](https://arxiv.org/html/2502.14786v1), Sections 2–3 including NaFlex; [official card](https://huggingface.co/microsoft/Phi-4-reasoning-vision-15B). Anchors: dynamic resolution, encoder/projector/backbone path, learned response modes. The patch toy has no fabricated image-accuracy curve.

## Teaching models and arithmetic

All widgets are labeled illustrative or simplified. Numerical examples were independently constructed for teaching and are not reported experiments. The prose explicitly excludes omitted memory components and distinguishes parameters from active computation, prompts from generated tokens, local-layer masks from whole-model context, and token-distribution distillation from generated-sequence imitation.

No third-party benchmark ranking is reproduced. No excerpt is copied verbatim. Model-specific summaries are kept short and distributed among report, announcement and card; longer exposition consists of original worked examples and general mathematical reasoning.

## Selection boundaries

Llama 3.2/3.3, Gemma 1, Gemma 3n, Phi-3.5 and Phi-4-multimodal are not given separate pages in this ten-lesson assignment. Gemma 3n’s per-layer-embedding lineage is acknowledged in the Gemma 4 tutorial. Phi-4’s base data/post-training ideas are covered in the Phi-4-reasoning lesson. These are editorial coverage boundaries, not claims that excluded releases had no innovations. No later unverified Llama release or Behemoth weight download is asserted.

## Validation completed by author

Parsed all ten JSON files; verified section counts, figure counts, interactive presence, caption caveats, source-ID resolution, and prose length. Central page generation, layout/browser verification and tests are owned by the root agent.
