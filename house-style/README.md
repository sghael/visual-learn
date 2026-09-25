# House style

Every explainer in Visual Learn is set in one visual and verbal style, after
Edward Tufte: show the data, put words beside the graphic they explain, and
remove everything else. This folder holds the reference.

- `base.css` — the stylesheet. Copy its rules into a new explainer's own CSS
  and add page rules after them. Explainers never link to this file.
- `specimen.html` — one page that uses every pattern in `base.css`: top bar,
  title, subtitle, lede, sidenotes, margin notes, a figure with controls and a
  margin caption, a wide figure of small multiples, a table, a code block and
  the colophon. Open it before building a page.

## Figures and interactions

Every figure earns its place. Before adding a widget, chart, diagram or
animation, answer in one sentence: what does the reader understand after using
it that a well-made static figure or a sentence would not give them? Then pick
one form:

- **Static figure** when interaction adds nothing. Draw the best frame well.
- **Small multiples** when the reader compares states (bit widths, eager
  versus compiled, 0 versus 4 registers, one date versus another). Put two to
  eight panels side by side on the same scale, so the comparison is visible at
  once instead of remembered across a toggle.
- **Interactive figure** when the reader should feel a relationship: a slider
  for a continuous variable, a segmented control for a few named cases, a
  stepper for a process whose order is the lesson.

An interactive figure:

- rests on a meaningful, labeled state before any input; the resting state
  already shows the main point;
- moves only after a button press, with Pause or Step and Reset beside it, and
  respects `prefers-reduced-motion`;
- keeps its controls in one `.controls` row above the graphic, as real
  `<button>`, `<input type="range">` and `<select>` elements that show their
  current value;
- reaches every detail by tap and keyboard as well as hover.

## Charts and diagrams

- Size SVG in CSS pixels: measure the container and redraw on resize, or wrap
  the figure in `.scroll-x` with a `--min` width. Chart text is at least 11 px
  at a 390 px viewport.
- Label data directly, at the line end or beside the bar. A legend, when
  unavoidable, is a phrase in the caption ("blue: the frontier").
- Context in grey (`--c-grey`), the subject in one color. Map each entity to
  one `--c-*` color at the top of the page's CSS and keep it for the whole
  page. Label everything that is colored.
- Hairline gridlines only where the reader reads values off them; 1 px
  strokes; no fills behind charts, no gradients, shadows or 3D. Log axes say
  "log scale" in the axis title.
- Annotate the point that matters in italic serif (`.annot`) beside it. Give
  labels that cross lines the `.halo` class.
- Diagrams draw boxes only for real components, with thin arrows and text on
  or beside the thing.
- Tables are booktabs: rules above and below the header and at the end, hairlines
  between rows, numbers right-aligned.
- A figure built on a simplified or simulated model says "simplified" or
  "simulated" in its caption.

## Page structure

- Sticky `.topbar`: a "Visual Learn" link to `../`, the short title, section
  links, a 1 px reading-progress line, and `aria-current` on the current
  section.
- `h1`, an italic `.subtitle` saying what the page teaches, a `.lede` of two
  to four sentences stating the problem, and a `.meta` line with the section
  count and reading time.
- Each section: prose that sets up the question, the concept, the figure, a
  one-line takeaway in the caption (`.takeaway`), then the consequences.
  Headings describe content ("How `jit` traces a function").
- Caveats, sources, definitions and version notes go in numbered sidenotes
  beside the sentence they qualify.
- A `.colophon` closes the page: sources, data dates, and links to Visual
  Learn and the repository.
- One light theme on `--paper`. Source Serif 4 for text, Source Sans 3 for
  labels, controls and tables, IBM Plex Mono for code.

## Writing

Write for an intelligent reader new to the subject. Introduce concepts in the
order they are needed, define each term at first use, and explain mechanisms
with concrete examples and real numbers with units. Say which kind of claim a
sentence makes: a fact, a reported finding ("the paper reports"), a vendor
figure ("vendor peak, approximate"), or a simulation ("in this simulation").
Name versions and dates for framework and product behavior. Use American
spelling.

Prefer short declarative sentences in which the subject does the verb, one
idea per paragraph, and a colon or a new sentence where a dash chain would go.
Name the quantity that changes, by how much, and why. Bold a term once, at its
definition. Start a section with its subject and end it on its last point.

Plain words replace hype and filler. Rewrite sentences built on *powerful,
robust, seamless, crucial, key insight, unlock, leverage, delve, landscape,
journey, intricate, elegant, simply, just, essentially, notably, under the
hood*; openers such as "In this section we will…", "Let's…" and "Here's the
thing"; closers that restate the paragraph; reflexive triplets; and "it's not
X, it's Y" framing, unless the contrast itself teaches something.
