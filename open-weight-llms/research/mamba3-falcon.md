# Mamba-3 and Falcon-H1 research memo

Research cutoff: 2026-10-03. Two independent mechanism tutorials, with five sections, two figures and 704/735 prose words.

## Mamba-3

- Folder: `2026-07-28-mamba-3`.
- Primary paper read: [Mamba-3](https://arxiv.org/html/2603.15569v1), March 16, 2026, especially Sections 3.1–3.4, Figure 1, and inference/retrieval discussion.
- Primary technical post read: [author post at Together AI](https://www.together.ai/blog/mamba-3), March 17. Read architecture, MIMO hardware motivation, retrieval boundaries, and latency methodology.
- Artifact read: [official MIMO 1.5B model card](https://huggingface.co/state-spaces/mamba3-mimo-1.5b). Public checkpoint configuration is distinguished from experiments described in the paper. The tutorial does not transfer paper benchmark values to the downloaded model.
- Independent API verification (unauthenticated): `https://huggingface.co/api/models/state-spaces/mamba3-mimo-1.5b` returned `private: false`, `gated: false`, `createdAt: 2026-07-27T22:06:17.000Z`, `lastModified: 2026-07-28T14:46:30.000Z`.
- Independent commits verification: the `/commits/main` API returned commit `bc6b5d0f7994fe4cb3478242e92da8daf9ee29ec`, titled **Mamba-3 public release**, dated `2026-07-28T14:46:30.000Z`. This explicit release marker supports the folder date. Neither paper publication nor initial repository creation is substituted for a weight-release date.
- README contains an obsolete private-repository authentication sentence. It is identified as stale; no authentication or gated download was needed to inspect public metadata.
- Teaching arithmetic: the recurrence example sets decay 0.8, step 1, endpoint mixture 0.5, state 10, previous projected input 2 and current projected input 6. Hence `8 + 0.4×2 + 0.5×6 = 11.8`. It is an invented scalar example, not an inference trace. Geometric rotations and state sizes are likewise illustrative.

## Falcon-H1

- Folder: `2025-05-21-falcon-h1`.
- Release evidence read: [TII announcement](https://www.tii.ae/news/middle-easts-leading-ai-powerhouse-tii-launches-two-new-ai-models-falcon-arabic-first-arabic), dated May 21, 2025.
- Primary paper read: [Falcon-H1](https://arxiv.org/html/2507.22448v1), architecture section, Figure 1, configuration table, related architecture discussion. July report date is not used as the release date.
- Primary technical post read: [Falcon team release post](https://falcon-lm.github.io/blog/falcon-h1/), family links, architecture, depth tradeoff and training design.
- Artifact read: [official 7B base card](https://huggingface.co/tiiuae/Falcon-H1-7B-Base), checkpoint purpose and license.
- Central diagram explicitly places attention and Mamba-2 in the **same parallel stage**. Concatenation then precedes output projection. It does not depict alternating sequential attention/SSM blocks or promise simultaneous kernel execution.
- State figure displays memory components separately; prose explicitly adds them. The hybrid retains a growing KV term. No flat-total-memory or throughput claim is made.

## Validation

Parsed both JSON files and checked counts, source IDs, caption simplifications and related slugs. Both use existing `state` and `flow` widgets. No new widget implementation is required. Benchmark marketing and copied prose are omitted; numerical teaching examples are independently constructed.
