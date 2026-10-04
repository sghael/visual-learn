# Additional family research

Cutoff: 2026-10-03. Notes identify the evidence used; the tutorial prose contains original worked examples rather than copied paper passages.

## gpt-oss

Read [official release](https://openai.com/index/introducing-gpt-oss/), [model card](https://deploymentsafety.openai.com/gpt-oss) sections 2–2.6, [implementation verification](https://developers.openai.com/cookbook/articles/gpt-oss/verifying-implementations), [Harmony](https://developers.openai.com/cookbook/articles/openai-harmony), and [Transformers deployment guide](https://developers.openai.com/cookbook/articles/gpt-oss/run-transformers).

- Public launch: 2025-08-05. 20b and 120b share one mechanism tutorial.
- Release table: approximately 21B/3.6B and 117B/5.1B total/active; 32/128 experts, top four; 24/36 layers.
- Model card gives more precise counts 20.91B and 116.83B; rounded numbers on the page are explicitly approximate.
- Expert weights are MXFP4; the whole runtime is not uniformly four-bit. Ideal GB arithmetic excludes scales and cache.
- Harmony controls roles/channels/handoffs; tool execution belongs to host software. Model-generated text is not proof a tool ran.
- Weight availability is not full-data or full-training reproducibility. No claims are made about later safeguard variants.

## Mamba

Read [paper](https://arxiv.org/html/2312.00752v2) sections 3.1–3.5 and Appendix C, [official implementation](https://github.com/state-spaces/mamba), and [checkpoint history](https://huggingface.co/state-spaces/mamba-130m/commits/main).

- HF public metadata read 2026-10-03: `private=false`, `gated=false`; first model release commit c4dcd5d65dff7d0bff5a4c97e30d82e57190f1b0 dated 2023-12-03T23:07:32Z.
- Selectivity changes input-dependent state dynamics; recurrent decoding and scan-based training have different execution patterns.
- Scalar gated recurrence is an original didactic reduction informed by Appendix C; it is not the full multidimensional block.
- Memory chart counts only an illustrative layer's state. Constant recurrent state does not imply constant training memory.

## Tülu 3

Read [release](https://allenai.org/blog/tulu-3-technical), [technical report](https://arxiv.org/html/2411.15124v3) sections 4–6 and related-work discussion, [project](https://allenai.org/tulu), and [model card](https://huggingface.co/allenai/Llama-3.1-Tulu-3-8B).

- Original public release 2024-11-21; initial 8B/70B Llama 3.1 derivatives. Later 405B release is separated.
- Pipeline: curated SFT, DPO, then PPO with verifiable rewards. RLVR is not itself an optimizer, and this tutorial does not substitute GRPO for PPO.
- Deterministic answer checks can miss incorrect reasoning or penalize harmless formatting. The widget intentionally demonstrates both errors with invented strings.
- The model weights retain the Llama license; recipe artifact licensing is separate.

## Olmo 3

Read [launch and updates](https://allenai.org/blog/olmo3), [report](https://arxiv.org/html/2512.13961v1) model-flow overview, midtraining sections 3.5 and RL-Zero analyses, [Think card](https://huggingface.co/allenai/Olmo-3-32B-Think), and [inference docs](https://docs.allenai.org/quick_start/running_locally).

- Public launch 2025-11-20; December arXiv and Olmo 3.1 updates are distinguished.
- Main contribution taught: documented data/checkpoint/code paths that permit earlier interventions and controlled experiments.
- The toy mixture allocates 100B tokens, but its percentages are not the measured Dolma 3 Dolmino composition and do not predict quality.
- The later Olmo-core 3 announcement (2026-10-01) describes training infrastructure and scale tests, not a released new pretrained Olmo checkpoint; excluded from dated model coverage.

## LLaDA

Read [paper](https://arxiv.org/html/2502.09992v3) sections 2.1–2.4, [project demonstrations](https://ml-gsai.github.io/LLaDA-demo/), [official repository and FAQ](https://github.com/ML-GSAI/LLaDA), [model card](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct), and [checkpoint history](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct/commits/main).

- Date ambiguity is disclosed: retrospective repo news says 2025-02-14; HF repo created February 19; weight-bearing Instruct commit 6059b30c9531767ce47e0ca6a8161a6bb0c67693 is 2025-02-20. A commit does not prove the repository was public on that date. Folder uses the verifiable artifact date, with this caveat on-page.
- Bidirectional transformer mask prediction, random masking levels, masked-only weighted loss, repeated denoising/remasking.
- The original implementation's cache/speed limitations are explicitly stated. Scripted fill order does not reproduce confidence or claim latency gains.
- LLaDA 1.5, MoE, iLLaDA, and the separate LLaDA2 line are not silently conflated with the original checkpoint. Later variants are mentioned as scope limits.

## BitNet b1.58 2B4T

Read [technical report](https://arxiv.org/html/2504.12285v2) architecture, training, and implementation; [earlier b1.58 paper](https://arxiv.org/html/2402.17764v1); [official model card](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T); and [implementation/release notes](https://github.com/microsoft/BitNet).

- Model release dated 2025-04-14 by official repository; April 16 paper date is separate.
- Ternary forward weights, absolute-mean scales, eight-bit activations, native training; higher-precision optimization state is separate from deployed low-bit representation.
- The scale 0.5 in the widget is invented for teaching. Log₂(3) is an information calculation, not an exact whole-model bytes-per-parameter claim.
- Specialized arithmetic needs compatible kernels. No speed, accuracy, or device-fit promise is inferred from the quantizer.
