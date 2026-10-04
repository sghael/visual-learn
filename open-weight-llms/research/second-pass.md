# Second correctness and clarity pass

All 40 chapters were reread against primary sources. The cutoff remains October 3, 2026. This pass distinguishes actual released mechanisms, experimental evidence and invented teaching calculations.

## OpenAI gpt-oss

The [model card](https://deploymentsafety.openai.com/gpt-oss) supports the exact parameter and checkpoint-size table, four selected experts, alternating full/local attention, 128-token local window and 64 query / 8 KV heads. Checkpoint GiB are separated from serving memory. MXFP4 includes block-scale overhead and was applied during post-training.

The [reference attention implementation](https://github.com/openai/gpt-oss/blob/main/gpt_oss/torch/model.py) appends a learned sink score before softmax, then removes that entry before mixing values. A new worked example uses values 2, 6 and 10: equal token weights average to 6; an equally weighted zero-contribution sink reduces output to 4.5. This is a teaching example, not measured model behavior or an extra cached token.

The [Harmony guide](https://developers.openai.com/cookbook/articles/openai-harmony) and [conversation-history guide](https://developers.openai.com/cookbook/articles/gpt-oss/handle-raw-cot) separate message endings, host tool handoffs and completed turns. Current-turn analysis survives a tool round trip; prior completed-turn analysis is excluded from the next turn. The lesson uses semantic labels rather than treating display strings as a parser specification.

## Corrections across the collection

- DeepSeek-R1 transfers curated reasoning data into supervised training restarted from V3-Base; its diagram now makes that restart explicit.
- Mixtral uses full causal attention in its original release; it does not inherit Mistral 7B’s sliding window unchanged.
- MLA’s learned bottleneck is separated from the exact algebraic absorption of learned projections.
- Qwen3.8’s 512 complete four-token blocks can be accompanied by an incomplete tail of up to three tokens.
- GLM-4.5 curriculum ablations use a smaller experimental model; the stated reasoning recipe omits a KL penalty.
- Kimi K3’s full report was recovered and read; the chapter now explains the release-specific KDA, Block AttnRes and latent-expert dimensions.
- Phi mini-flash timing conditions include random weights, an A100 and a custom inference stack. Phi vision resolution ablations used a smaller experimental model.
- Olmo’s November releases are distinguished from later 32B Instruct availability.
- LLaDA’s mask-loss weighting and fixed teaching canvas are distinguished from real EOS termination.

Each chapter carries the relevant primary citations and qualifications. The family memos preserve the earlier research trail; this note records the subsequent corrections.

## Presentation and checks

The prose column is wider throughout the book. Caption placement adapts at 1280px. Shared experts are directly labeled as always active, verifier control groups have distinct accessible names, and ternary rounding uses symmetric ties away from zero. Tests cover every widget instance and the new attention-sink calculation. Browser checks validate layout and controls; they do not reproduce model benchmark results.
