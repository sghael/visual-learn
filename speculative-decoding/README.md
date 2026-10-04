# Speculative decoding

A self-contained visual tutorial on the original linear draft–target algorithm
described by Leviathan et al. and Chen et al. It explains why a small draft model
can reduce the number of sequential target-model calls, how verification retains
an accepted prefix, and why sampling needs a probability correction.

The page has six sections, five figures, and three interactive widgets. All token
examples are hand-authored. No language model runs or downloads in the browser.

## Open the page

Open `index.html` directly in a browser using `file://`. The page uses classic
scripts and local relative asset paths, so it needs no build step or server.
Google Fonts are optional; system fonts provide fallbacks.

To serve just this tutorial, replace the example directory with its absolute path:

```sh
python3 -m http.server 8000 --bind 0.0.0.0 --directory /absolute/path/to/visual-learn/speculative-decoding
```

Use the port printed by the server and open the serving machine's reachable
address. If the port is occupied, choose a free
one and restart. On configured agent hosts, `free-port` selects an available port
and `agent-preview-url <actual-port>` prints the shareable address. The collection
back-link needs the full checkout or the published site. The same relative assets work under GitHub Pages at
`/visual-learn/speculative-decoding/`.

## Figures and controls

1. **Decoding schedule:** a static comparison of sequential target calls with
   sequential drafting followed by one verification pass. Bar lengths illustrate
   the mechanism; they are not benchmark measurements.
2. **One greedy round:** choose a first-token mismatch, third-token mismatch, or
   complete match. Replay/Pause, Step, and Reset expose four draft calls, one
   target pass, and the committed result. A completed example is visible on load.
   Changing cases or resetting cancels playback. With reduced motion enabled,
   Replay/Play shows the completed round immediately; manual Step still works.
3. **Verification prefixes:** a static table shows the separate prefix used for
   each target distribution and why distributions after a rejection are unused.
4. **Probability correction:** vary the draft probability of “lake.” The remaining
   mass is split between “river” and “sea” in a 3:2 ratio. The table computes
   accepted mass, the correction conditional on rejection, and final output.
   “Draft equals target” sets all three probabilities exactly; moving the slider
   leaves that preset. No random sampling is involved.
5. **Latency model:** vary acceptance, draft length, draft-step cost, and
   verification cost. Three presets demonstrate different cost/acceptance
   combinations. The curve compares draft lengths 1–12 and identifies the best
   length within that range under the selected assumptions.

## Models and accuracy limits

The notation is `p` for target and `q` for draft. Chen et al. use the reverse.
Sampling accepts a proposal `x` with probability `min(1, p(x) / q(x))`; on
rejection, it samples from normalized `max(0, p - q)`. The displayed distributions
represent the actual sampling distributions, after any temperature or filtering.
When the distributions match, rejection is impossible and correction is unused.

Verification only retains a consecutive accepted prefix. After the first
rejection, later predictions have the wrong prefix and are discarded. Full
acceptance permits one bonus token. Greedy equivalence assumes consistent
tie-breaking and equivalent numerical results. Stochastic equivalence concerns
the output distribution, not matching the text produced with a particular seed.
Acceptance says nothing about a statement's factual truth.

The simplified latency model assumes constant, independent acceptance probability
`alpha`, draft length `k`, draft-step cost `c`, and verification cost `v`, with
costs measured in ordinary target-step units:

```text
expected tokens = 1 + alpha + alpha^2 + ... + alpha^k
modeled speedup  = expected tokens / (v + k*c)
```

The original idealized analysis uses `v = 1`; this page also permits more
expensive verification. The curve holds `alpha`, `c`, and `v` fixed as `k`
changes. It omits setup, communication, cache management, and finite-response
effects. Real acceptance depends on context, and verification cost can grow with
block length. These values are illustrative, not hardware measurements.

The tutorial covers a shared token vocabulary and a single linear draft. It does
not implement token trees, alternative tokenizers, cache rollback, or stopping
logic. A real decoder must discard rejected cache state and honor end-of-sequence
and length limits on committed output.

## Tests

Use the pnpm version pinned by `packageManager` in `package.json`:

```sh
pnpm --dir /absolute/path/to/visual-learn/speculative-decoding install --frozen-lockfile
pnpm --dir /absolute/path/to/visual-learn/speculative-decoding test
```

The tests use Node's test runner and Playwright. They try installed Chrome first
and fall back to Playwright Chromium. By default they open the local `file://`
page. Set `TEST_PAGE_URL` to check a served or published page, for example:

```sh
TEST_PAGE_URL=https://sghael.github.io/visual-learn/speculative-decoding/ pnpm --dir /absolute/path/to/visual-learn/speculative-decoding test
```

Coverage includes rendering at 1440 px and 390 px, page overflow, chart text size,
console and page errors, greedy cases, animation cancellation, reduced motion,
probability correction, slider endpoints, preset calculations, keyboard controls,
and section navigation. Web fonts are stubbed for stable offline testing.
Screenshots still need visual inspection before publishing.

## Files

- `index.html`: lesson text, figures, controls, and source links.
- `styles.css`: a local copy of the house style plus page-specific rules.
- `models.js`: pure probability-mass and latency calculations.
- `main.js`: widget state, rendering, playback, and reading progress.
- `test/page.test.mjs`: browser and mathematical regression checks.
- `package.json` and `pnpm-lock.yaml`: pinned test tooling; no production build.

## Sources

- Yaniv Leviathan, Matan Kalman, and Yossi Matias,
  [*Fast Inference from Transformers via Speculative Decoding*](https://proceedings.mlr.press/v202/leviathan23a.html),
  ICML 2023. Algorithm 1, §§2–3, and Appendix A.1 establish the algorithm,
  acceptance rule, simplified speed analysis, and distributional proof.
- Charlie Chen et al.,
  [*Accelerating Large Language Model Decoding with Speculative Sampling*](https://arxiv.org/abs/2302.01318),
  2023. Algorithm 2 and the conditional-scoring discussion explain parallel
  verification and the numerical and serving assumptions.
