# Research notes: Kimi, GLM and Qwen

Research cutoff: **2026-10-03**. This is an innovation-oriented selection, not an inventory of every checkpoint, size, quantization or hosted model. The twelve lessons distinguish inherited components from release-specific integration. Model-specific facts use primary sources. Worked examples and arithmetic are original teaching constructions, labeled illustrative or simplified.

## Release-date policy and verification

Dates describe public weight releases where identifiable, not paper submission dates. A repository's creation or staged upload can predate public availability, so it is not sufficient alone. Official announcement chronology, model cards and weight commit histories were cross-checked. The Hugging Face API endpoint `/api/models/<organization>/<model>/commits/main?limit=100` was read for the cases below. These observations should not be represented as a claim that a private repository was public at its initial timestamp.

| Folder | Date evidence and qualification |
| --- | --- |
| `2024-09-19-qwen2-5` | Official launch article dated September 19, 2024. December report is later. |
| `2025-04-29-qwen3` | Official launch uses April 29; weight uploads are April 28 UTC. The lesson uses the announcement's calendar date. |
| `2025-07-11-kimi-k2` | Moonshot official research chronology and July 11 weight uploads. |
| `2025-07-28-glm-4-5` | Public release July 28 and corresponding uploads; initial placeholder July 20 excluded. |
| `2025-09-11-qwen3-next` | Public launch September 11; weights were staged September 9, with release metadata and license finalized September 11. Explicit uncertainty note retained. |
| `2025-11-06-kimi-k2-thinking` | Moonshot official research chronology dates release November 6. |
| `2026-01-27-kimi-k2-5` | Official announcement January 27; January 1 staged repository upload excluded. |
| `2026-02-11-glm-5` | Weight release history February 11; official blog is dated February 12; report February 17. Difference stated in lesson. |
| `2026-02-16-qwen3-5` | Official checkpoint's first complete weight uploads February 16. |
| `2026-07-27-kimi-k3` | July 16 announcement explicitly promises weights by July 27; official weight history begins July 27. |
| `2026-08-26-glm-5-3-flash` | Weight uploads August 26; August 25 placeholder excluded. |
| `2026-08-26-qwen3-8-flash-next` | Official weight upload August 26; architecture report August 31. |

## Sources actually inspected and claim anchors

### Qwen2.5

- [Official launch](https://qwenlm.github.io/blog/qwen2.5/): date, dense family sizes, Apache exceptions for 3B and 72B; specialization and structured-output emphasis.
- [Technical report](https://arxiv.org/html/2412.15115v1): sections 2–4, especially 3.1 quality filtering, synthetic-data filtering, domain mixtures; table 1 architecture/licenses; 3.3 context-training stages and inference extension.
- [7B-Instruct card](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct): 28 layers, 28 query heads, 4 KV heads; default context versus YaRN extension. KV example is an explicitly incomplete bf16 calculation, not GPU-sizing advice.
- The `6ND` budget figure uses original illustrative numbers; it is not the Qwen2.5 training recipe or an accuracy prediction.

### Qwen3

- [Official launch](https://qwenlm.github.io/blog/qwen3/): hybrid thinking modes, dense and MoE open checkpoints, license and model counts.
- [Technical report](https://arxiv.org/html/2505.09388v1): sections 4.1–4.5 and 4.7, cold-start reasoning, RL, mode fusion, general RL, on-policy teacher-logit distillation. The toy distribution does not imply an actual training temperature.
- [8B model card](https://huggingface.co/Qwen/Qwen3-8B): `enable_thinking`, `/think`, `/no_think` and template behavior for the original hybrid checkpoint.

### Qwen3-Next

- [Official Instruct model card](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct): three Gated DeltaNet layers per gated-attention layer; 48 layers, 512 routed experts, 10 selected, one shared; native context and extension; direct-answer Instruct versus separate Thinking checkpoint; MTP runtime caveat.
- [Gated Delta Networks paper](https://arxiv.org/html/2412.06464v1): sections 2.3–3.1, delta update, matrix associative state and decay gate. This component predates Qwen3-Next.
- [Weight history](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/commits/main): staged and release metadata dates.
- State-memory and scalar delta-update figures are independent pedagogical examples. They make no model-memory or throughput claims.

### Qwen3.5

- [Official 397B-A17B card](https://huggingface.co/Qwen/Qwen3.5-397B-A17B): early multimodal fusion, vision encoder, 60-layer hybrid layout, 512 routed experts with 10 selected and one shared, native context and separate hosted Plus features.
- [Official blog](https://qwen.ai/blog?id=qwen3.5): indexed official text establishes asynchronous multimodal/multi-turn RL. Direct web and HTTP extraction returned an empty dynamic shell; the lesson's detailed architecture claims rely on the readable official model card, not an imagined full blog.
- [Weight history](https://huggingface.co/Qwen/Qwen3.5-397B-A17B/commits/main): release February 16.
- Counterfactual-image examples are suggested diagnostics, not reported benchmark experiments.

### Qwen3.8-Flash-Next

- [Official model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next): 125B backbone / 6B activated **plus** 51B n-gram embeddings and 4B MTP; QSA 512-block / 2,048-token budget; custom Qwen Community License 1.0.
- [Architecture report](https://arxiv.org/html/2608.30320v1): sections 2.1.2 (compressed micro-block indexer, QSA), 2.2 (four-branch Gated Residual, elementwise read, per-branch write, removal of residual mixing operator), 2.3 (host-memory n-gram tables), section 3 training/optimizer/stability considerations.
- [Weight history](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/commits/main): August 26 weights, August 31 paper.
- Total parameter categories must not be collapsed into “125B total” or “180B active.” The table example is invented arithmetic, not the implementation's hashing/indexing scheme.

### Kimi K2

- [Official technical blog](https://www.kimi.com/en/blog/kimi-k2): Base/Instruct release, initially non-thinking and text-only; simulated tool-use pipeline and task rubrics; MuonClip overview.
- [Technical report](https://arxiv.org/html/2507.20534v1): 2.1 QK-Clip and per-head MLA treatment; 2.5 training recipe; 3.1 multi-agent data synthesis, real and simulated environments.
- [Official model card](https://huggingface.co/moonshotai/Kimi-K2-Instruct): 384 routed experts / eight selected / one shared; approximate total and active parameters; modified MIT terms.
- QK-Clip's scalar example rescales a product; it is explicitly not the full MLA-specific clipping implementation. The router is a downscaled eight-expert toy and is never called a speed measurement.

### Kimi K2 Thinking

- [Official announcement](https://www.kimi.com/en/blog/kimi-k2-thinking): interleaved reasoning and tool use; reported long-horizon traces, not a general reliability guarantee.
- [Official card](https://huggingface.co/moonshotai/Kimi-K2-Thinking): section 4 specifically says INT4 **weight-only quantization of MoE components** and QAT during post-training. Avoid saying all model tensors are INT4.
- [Official chronology](https://www.kimi.com/en/blog/): November 6, 2025.
- Precision widget is a uniformly encoded imaginary 20B-weight array; it does not estimate the full checkpoint footprint or infer the vendor's speed result.

### Kimi K2.5

- [Technical report](https://arxiv.org/html/2602.02276v1): 2.2 zero-vision SFT occurs **after multimodal pretraining**; 2.3 visual outcome RL; 3 PARL, frozen subagents, critical steps; 4.2 MoonViT-3D and temporal pooling.
- [Official blog](https://www.kimi.com/en/blog/kimi-k2-5): PARL reward shaping, serial collapse and spurious parallelism; Agent Swarm research-preview product status; benchmark appendix, task-specific thinking modes and context treatment.
- [Official card](https://huggingface.co/moonshotai/Kimi-K2.5): downloadable checkpoint and deployment boundary.
- Do not attribute a hosted Agent Swarm result to a checkpoint alone. The parallel scheduler is generic arithmetic and includes no claimed model intelligence or measured runtime.

### Kimi K3

- [Official July 16 announcement](https://www.kimi.com/en/blog/kimi-k3): KDA, AttnRes and promised July 27 weight release.
- [Official card](https://huggingface.co/moonshotai/Kimi-K3): 2.8T / 104B, 69 KDA + 24 gated-MLA layers, 896 routed experts / 16 selected / two shared; MXFP4 weights and MXFP8 activations in QAT; custom Kimi K3 license.
- [Weight history](https://huggingface.co/moonshotai/Kimi-K3/commits/main): first weight commit July 27.
- [Kimi Linear component report](https://arxiv.org/html/2510.26692v1): KDA channel-wise gating, chunkwise computation, hybrid recurrence/MLA. Component work and earlier open Kimi Linear checkpoint predate K3.
- [Attention Residuals component paper](https://arxiv.org/pdf/2603.15031): learned softmax attention over preceding layer outputs; block-level form and memory/communication rationale.
- Second-pass update: the full K3 report was retrieved from https://raw.githubusercontent.com/MoonshotAI/Kimi-K3/main/k3_tech_report.pdf. Architecture §§2.1–2.3, Table 1 and training §3 support the added bounded KDA decay, Block AttnRes, 3,584-dimensional expert latent space and post-SFT quantization details.

### GLM-4.5

- [Technical report](https://arxiv.org/html/2508.06471v1): 2.3 repository-scale mid-training; 3.2 difficulty curriculum and zero reward-variance issue; architecture table counting caveats. Group-normalization widget is not asserted to reproduce GLM's complete optimization algorithm.
- [Official model card](https://huggingface.co/zai-org/GLM-4.5): 355B/32B and Air 106B/12B; hybrid thinking; MIT weights.
- [Official repository](https://github.com/zai-org/GLM-4.5): separate version sections; preserved and turn-level thinking introduced in later GLM-4.7 must not be retroactively attributed to 4.5.

### GLM-5

- [Technical report](https://arxiv.org/html/2602.15763v1): section 4.1 asynchronous generation/learning separation, periodic synchronization, policy-lag issue, task orchestration and token-in/token-out trace fidelity.
- [Official announcement](https://z.ai/blog/glm-5): 744B/40B, adopted DeepSeek Sparse Attention and MIT weights.
- [Family repository](https://github.com/zai-org/GLM-5): GLM-5.2 IndexShare across sparse-attention layers; GLM-5.3 same base and post-training; Flash new base.
- [Weight history](https://huggingface.co/zai-org/GLM-5/commits/main): February 11 uploads versus February 12 blog. The page explicitly records the date difference.
- Parallel widget is an independent scheduling analogy, not a model of slime or a measured training speedup. A guessed `zai-org/slime` URL returned 404 and is not cited.

### GLM-5.3-Flash

- [Official model card](https://huggingface.co/zai-org/GLM-5.3-Flash): native multimodality, 320B/18B, 30T training corpus, hybrid sparse/linear attention, mHC and MIT weights.
- [Released configuration](https://huggingface.co/zai-org/GLM-5.3-Flash/raw/main/config.json): repeating three `linear_attention` entries plus one `deepseek_sparse_attention`; `hc_mult=4`, `hc_sinkhorn_iters=20`, indexer fields.
- [mHC component paper](https://arxiv.org/html/2512.24880v1): doubly stochastic residual mixing; inherited DeepSeek mechanism, not a GLM invention. Original two-stream arithmetic demonstrates convex mixing only, without claiming full-network stability.
- [Official family repository](https://github.com/zai-org/GLM-5): distinguishes the Flash base from the larger GLM-5.3 post-training release.
- [Weight history](https://huggingface.co/zai-org/GLM-5.3-Flash/commits/main): August 26 weight uploads.
- [Official blog](https://z.ai/blog/glm-5.3-flash) returned an empty shell in both web extraction and direct HTTP. No unobserved blog detail is claimed.

## Selection boundaries

- Kimi K1.5 is an important reasoning paper but its original flagship weights were not established as a public release in this investigation, so it does not receive a dated open-weight folder.
- Kimi K2-Instruct-0905 and K2.6 were inspected through Moonshot's official chronology/blogs; their improvements extend the K2/K2.5 agentic trajectory. They are not separate architectural lessons here.
- Kimi Linear and Attention Residuals are explicitly sourced as component innovations within K3. A later dedicated Kimi Linear lesson could be useful, but the current page avoids claiming that KDA first appeared in K3.
- GLM-4.6 and 4.7 changes are version-separated in the GLM-4.5 discussion. GLM-5.2 IndexShare and GLM-5.3 post-training appear as distinct later developments in the GLM-5 discussion; the new Flash base gets its own lesson.
- Qwen3.6, 3.7, general 3.8 checkpoints, Coder, VL, Omni, speech and image families are not exhaustively catalogued. The Qwen selection follows a concrete progression in data, reasoning, hybrid memory, native visual fusion, and lookup/depth architecture.
- Hosted-only previews and announced future models are excluded. No inference from a future-facing model name is presented as a released model.

## Editorial and validation checks

- Every lesson has four sections, at least two figures, and an interactive mechanism figure.
- All JSON parses; all `#source-*` links resolve to a source ID in the same lesson.
- Figure parameters are explicitly illustrative unless a KV dimension is grounded in a released configuration.
- No benchmark rankings or kernel-speed ratios are reproduced; this avoids conflating incompatible evaluation settings and separates mechanisms from marketing.
- Derived source detail is kept brief and paraphrased. Most explanatory length comes from original arithmetic, worked examples and evaluation reasoning, rather than extensive reproduction of any one source.
