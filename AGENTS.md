# AGENTS.md

Repo-local guidance for `sghael/visual-learn`. The shared workspace rules in
`~/Developer/AGENTS.md` still apply.

## Layout

- One explainer per top-level folder, each fully standalone: `index.html`,
  its own CSS and JS, no build step, no framework, no shared code between
  folders. Fonts may load from Google Fonts with system fallbacks.
- The root `index.html` is the landing page. Add a card there when an
  explainer is ready to publish; leave unfinished folders off the landing page.
- `main` deploys straight to GitHub Pages at `https://sghael.github.io/visual-learn/`.
  Every folder is served under its own path, so there is no per-folder deploy
  step. Use relative URLs only, so each page also works from disk and from a
  local `python3 -m http.server`.

## Quality bar

- Verify in headless Chrome at a desktop width and a phone width before
  opening a PR: no console errors, no horizontal page overflow, wide diagrams
  scroll inside their own container.
- Explainers with interactive state ship a `test/` folder and a `package.json`
  with a `test` script (Playwright, `channel: 'chrome'` locally, bundled
  Chromium in CI). CI discovers every folder with a `package.json` on its own;
  nothing at the root needs editing.
- Technical claims about frameworks or hardware are stated with the version
  or vendor figure they came from, and simplified models say so on the page.
