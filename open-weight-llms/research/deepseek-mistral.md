# DeepSeek and Mistral research memo

Research cutoff: October 3, 2026. Dates in folders follow public weights rather than paper submission. The eight lessons contain original worked examples, labeled toy calculations, and release-specific claims linked to primary sources. Benchmark rankings and API pricing are intentionally outside the teaching scope.

## Release decisions

| Folder | Date evidence | Mechanism selected |
| --- | --- | --- |
| `2023-09-27-mistral-7b` | [Official announcement](https://mistral.ai/news/announcing-mistral-7b/) | GQA plus original v0.1 sliding-window cache |
| `2023-12-08-mixtral-8x7b` | [Original Mistral AI torrent post](https://x.com/MistralAI/status/1733150512395038967), verified via official X syndication JSON | Sparse feed-forward experts, total versus active parameters |
| `2024-05-06-deepseek-v2` | [Repository news](https://github.com/deepseek-ai/DeepSeek-V2) explicitly gives May 6 | MLA plus inherited DeepSeekMoE |
| `2024-12-26-deepseek-v3` | [Official change log](https://api-docs.deepseek.com/updates/) | Routing bias, mixed precision, sequential MTP |
| `2025-01-20-deepseek-r1` | [Official release](https://api-docs.deepseek.com/news/news250120/) | GRPO reasoning post-training and sequence distillation |
| `2025-09-29-deepseek-v3-2` | Official change log; [Exp weights](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp) | DSA first shipped in Exp; full December model discussed separately |
| `2026-04-24-deepseek-v4` | [Official release and public weight links](https://deepseek.com/en/news/v4-preview/) | Sequence compression, CSA/HCA and constrained residual mixing |
| `2026-09-10-deepseek-v4-1-flash` | [Official release and public weight link](https://deepseek.com/en/news/deepseek-v4-1-flash/) | CED, cross-layer global-cache reuse, FP4 and approximate replay |

The Mixtral blog date is December 11. Direct X-page retrieval failed; its official syndication endpoint returned the original author, torrent text and `created_at=2023-12-08T15:44:17.000Z`. Endpoint: `https://cdn.syndication.twimg.com/tweet-result?id=1733150512395038967&lang=en&token=0`. This resolves the otherwise common December 8/9/11 ambiguity.

## Sources read and claim anchors

### Mistral 7B and Mixtral

- [Mistral 7B paper](https://arxiv.org/html/2310.06825v1): Section 2 and Table 1. Read architecture, rolling buffer and prompt chunking. The lesson isolates GQA in its cache widget and explicitly excludes rolling-window eviction there.
- [GQA paper](https://arxiv.org/html/2305.13245v3): mechanism and grouped KV heads; establishes prior art.
- [Sparse Transformer paper](https://arxiv.org/abs/1904.10509): abstract and prior sparse-attention contribution. Used only for historical attribution, not a claim that its architecture equals Mistral.
- [Mixtral paper](https://arxiv.org/html/2401.04088v1): Sections 2 and 5. Read top-two routing, shared components and routing analysis. Do not label experts as fixed academic subjects.
- [Mixtral official announcement](https://mistral.ai/news/mixtral-of-experts/), [model card](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1), and [8×22B release](https://mistral.ai/news/mixtral-8x22b/): public weights, counts, license and scaled companion.
- [Sparsely gated MoE paper](https://arxiv.org/abs/1701.06538): abstract; used only to establish pre-Mixtral conditional computation.

### DeepSeek-V2 and V3

- [V2 report](https://arxiv.org/html/2405.04434v5): Sections 2.1.2–2.1.4 and architecture discussion. Read latent projections, projection absorption and decoupled RoPE. The latent slider deliberately does not reproduce exact MLA storage.
- [DeepSeekMoE report](https://arxiv.org/html/2401.06066v1): fine-grained segmentation and shared-expert isolation. This is prior work adopted in V2.
- [V2 repository](https://github.com/deepseek-ai/DeepSeek-V2): date, downloads, headline dimensions and separate code/model licenses.
- [V3 report](https://arxiv.org/html/2412.19437v2): Section 2.1.2 bias routing; Section 2.2 causal MTP; FP8 discussion and training-accounting scope. “Auxiliary-loss-free” must not conceal the small complementary sequence loss.
- [Load-balancing paper](https://arxiv.org/html/2408.15664v1): traffic bias motivation and selection-score distinction.
- [Earlier MTP paper](https://arxiv.org/html/2404.19737v1): independent future-token heads, contrasted with V3’s sequential construction.
- [V3 repository](https://github.com/deepseek-ai/DeepSeek-V3): parameters, released FP8 weights and original model license.

### R1 and V3.2

- [R1 report](https://arxiv.org/html/2501.12948v1): Sections 2.2–2.4, rewards, R1-Zero limitations, cold start, full pipeline and distillation. Distinguish pretrained R1-Zero plus RL from final R1’s mixed recipe.
- [DeepSeekMath report](https://arxiv.org/html/2402.03300v1): Section 4.1 and equations for GRPO. Establishes algorithm origin before R1; the widget omits the full loss.
- [R1 repository](https://github.com/deepseek-ai/DeepSeek-R1): 800,000-sample distillation description, base checkpoints and license inheritance.
- [V3.2 report](https://arxiv.org/html/2512.02556v1): Sections 2–3; learned indexer, top-k token selection, dense warm-up, sparse adaptation and agent post-training. Exp and full V3.2 share architecture.
- [Exp card](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp) and [full model card](https://huggingface.co/deepseek-ai/DeepSeek-V3.2): public weights, synthesis, Speciale’s lack of tool calling, MIT license.

### V4 and V4.1

- [V4 report](https://arxiv.org/html/2606.19348v1): Sections 2.1–2.3, sequence compression and CSA/HCA. Read dense access over heavily compressed entries versus sparse access over less-compressed entries; both retain local attention.
- [mHC paper](https://arxiv.org/html/2512.24880v1): Sections 3–4, nonnegative doubly stochastic residual mixing. The lesson’s numeric example constrains the residual mapping, not the entire neural network.
- [V4 model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash): architecture, inherited components, downloadable weights and license. The June report is later than the April preview, which the date note makes explicit.
- [V4.1 report](https://arxiv.org/html/2609.19969v1): Sections 2.2–2.4 and 3.2. Read CED, CSA2 modes, FP4 main KV, HBM versus persistent storage and bounded replay. Replay is approximate.
- [V4.1 model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash): encoder/decoder layer split, image support, extra architecture components and MIT license.

## Scope and exclusions

Mixtral 8×22B appears as a scaling companion inside the Mixtral lesson. DeepSeekMoE and DeepSeekMath are credited where their mechanisms enter the narrative; the lessons do not imply V2 or R1 invented them on release day. V2.5, V3-0324, V3.1, R1-0528, V4-0731 and V4-Pro-0813 are checkpoint revisions rather than duplicate tutorials of the same central mechanism. Full V3.2 is covered with the earlier architecture debut and an explicit date distinction.

This assigned subset is not a complete inventory of every DeepSeek/Mistral release. Further distinct topics include optical context compression in DeepSeek-OCR, verifier-driven theorem proving, Mistral’s multimodal models and recurrent Codestral Mamba. The collection’s root scope should describe representative mechanisms, not claim literally exhaustive coverage.

## Figure audit

Eight lessons; five sections each; 714–764 body-prose words before figure captions; two figures per lesson. New kinds requested and accepted by the parent: `sparse` and `cache`. Every example identifies its simplifications. No parameter fraction is presented as a measured wall-clock speedup. No teacher-distribution widget is used to misrepresent R1’s sequence imitation. No locality widget is used to misrepresent learned DSA selection. V4’s flow depicts interleaved layer types, not a serial CSA-then-HCA compression operation on one cache.
