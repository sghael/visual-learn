# AGENTS.md

Guidance for agents building explainers in `sghael/visual-learn`. Read it
before adding or changing anything. The workspace rules in
`~/Developer/AGENTS.md` still apply: work in a worktree, commit with the
active-agent author, open a PR, run `ai-review`, land with `pr-land`.

## What this repo is

Interactive, single-page visual explainers on how machine-learning systems
work, in the spirit of the Polo Club explainers. `main` is published by
GitHub Pages at `https://sghael.github.io/visual-learn/`. A folder `foo/` at
the repo root is live at `https://sghael.github.io/visual-learn/foo/` about a
minute after it lands. There is no per-folder deploy step and no site build.

## Adding an explainer

1. **One folder per explainer, at the repo root**, named for the subject in
   kebab-case (`jax-vs-pytorch`, `qat-explainer`), never for the session or
   the author. Do not touch other explainers' folders in the same PR.
2. **Everything the page needs lives in its folder**: `index.html`, its CSS,
   its JS, its data. No build step, no bundler, no framework, no imports from
   other folders. Third-party scripts only from cdnjs, pinned to a version,
   and only when they carry real weight. Google Fonts with system-font
   fallbacks are fine.
3. **Relative URLs only.** The page must work three ways: opened from disk
   (`file://`), served locally (`python3 -m http.server`), and under the
   `/visual-learn/<folder>/` subpath on GitHub Pages. Never link to `/x`.
4. **A `README.md` in the folder**: what the explainer teaches, how to open
   it, what each widget does, and any accuracy caveats or simplifications.
5. **Add a card to the root `index.html` only when the explainer is finished.**
   Unfinished work may live on a branch, but the landing page lists only
   what a reader should see. Keep the card's description to one sentence
   and the meta line to what is true (section count, reading time, folder).

## Quality bar, checked before opening a PR

- **Renders clean at two widths.** Headless Chrome at 1440 px and 390 px:
  zero console errors, zero page errors, zero horizontal page overflow. Wide
  diagrams scroll inside their own container (`overflow-x: auto`); the page
  body never scrolls sideways. Look at your own screenshots before asking
  for review.
- **Controls are real controls.** Buttons are `<button>`, sliders are
  `<input type="range">`, all keyboard reachable with a visible focus state.
  Animations respect `prefers-reduced-motion` and start from a visible
  resting state, never from `opacity: 0`.
- **Color is never the only channel.** Label what you color. Use a
  categorical palette validated for colorblind safety and keep entity colors
  fixed across the page (in `jax-vs-pytorch`, JAX is always violet, PyTorch
  always orange, "active" always green).
- **Claims are sourced.** Framework behavior names the version it describes;
  hardware numbers are vendor peaks marked approximate; simplified models
  say "simplified" on the page itself, not only in the README.
- **Interactive state ships tests.** Any explainer with steppers, toggles,
  presets or animations gets a `package.json` with a `test` script and
  `test/*.test.mjs` on `node --test`, driven by Playwright with
  `channel: 'chrome'` locally (it falls back to bundled Chromium in CI). Pin
  `packageManager` to pnpm and commit `pnpm-lock.yaml`. Test what has bitten
  before: switching examples mid-animation, reset during a run, preset
  labels versus what they compute. `jax-vs-pytorch/test/` is the reference,
  including a fake-clock test for animation races.
- **CI is automatic.** `.github/workflows/test.yml` discovers every root
  folder that has a `package.json` and runs its `pnpm test`. Nothing at the
  root needs editing when you add a folder.

## After landing

- Wait for the `pages build and deployment` run, then curl the live URL and
  run your headless-Chrome pass once against it. Web fonts and relative
  paths behave differently on the subpath than from disk; the live page is
  the one readers get.

## Conventions that keep the collection coherent

These are defaults, not rules. Diverge when the subject calls for it.

- Light, quiet page on a warm off-white ground; a serif display face with a
  sans body and a mono face for code (`jax-vs-pytorch` uses Instrument Serif,
  Inter and JetBrains Mono).
- Prose column about 68 characters wide; visualization "stages" up to 68rem,
  each with a title, a hint, its controls, and a one-line takeaway or legend
  in the footer.
- Narrative order per section: a hook in prose, the concept, the interactive
  widget, a takeaway. A sticky section nav with a reading-progress bar helps
  on long pages.
- Widgets are self-contained modules that register with a tiny shared
  toolkit inside the folder (see `jax-vs-pytorch/js/common.js`); there is no
  cross-folder sharing on purpose, so each explainer stays copyable.
