/* K2 Horizon explainer. Plain SVG, no libraries. Every figure is drawn at the
   measured width of its container and redrawn when that width changes, so
   chart text stays at its CSS size on a phone. */
'use strict';
(function () {
  const NS = 'http://www.w3.org/2000/svg';
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  function el(tag, attrs, parent, text) {
    const e = document.createElementNS(NS, tag);
    if (attrs) for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  }
  function svgIn(host, w, h, label) {
    host.textContent = '';
    const s = el('svg', { width: w, height: h, viewBox: `0 0 ${w} ${h}` }, host);
    if (label) { s.setAttribute('role', 'img'); s.setAttribute('aria-label', label); }
    return s;
  }
  // Draw at the host's width now and whenever it changes. Returns a redraw function.
  function responsive(host, draw) {
    let last = -1;
    const run = (force) => {
      const w = Math.floor(host.clientWidth);
      if (!w || (w === last && !force)) return;
      last = w; draw(w);
    };
    if ('ResizeObserver' in window) new ResizeObserver(() => run(false)).observe(host);
    window.addEventListener('resize', () => run(false));
    run(true);
    return () => run(true);
  }
  const f1 = (v) => v.toLocaleString('en-US', { minimumFractionDigits: 1, maximumFractionDigits: 1 });
  const f2 = (v) => v.toFixed(2);

  /* ---------------- top bar: progress line and current section ---------------- */
  (function topbar() {
    const nav = document.querySelector('.topbar nav');
    const links = [...nav.querySelectorAll('a')];
    const secs = links.map((a) => document.querySelector(a.getAttribute('href')));
    const bar = document.getElementById('progress');
    let current = -2;
    function update() {
      const h = document.documentElement;
      const max = h.scrollHeight - h.clientHeight;
      bar.style.width = (max > 0 ? Math.min(100, (h.scrollTop / max) * 100) : 0) + '%';
      let cur = -1;
      const line = window.innerHeight * 0.3;
      secs.forEach((s, i) => { if (s && s.getBoundingClientRect().top <= line) cur = i; });
      if (cur === current) return;
      current = cur;
      links.forEach((a, i) => { if (i === cur) a.setAttribute('aria-current', 'true'); else a.removeAttribute('aria-current'); });
      const a = links[cur];
      if (a) {
        const l = a.offsetLeft, r = l + a.offsetWidth;
        if (l < nav.scrollLeft) nav.scrollLeft = l - 8;
        else if (r > nav.scrollLeft + nav.clientWidth) nav.scrollLeft = r - nav.clientWidth + 24;
      }
    }
    window.addEventListener('scroll', update, { passive: true });
    window.addEventListener('resize', update);
    update();
  })();

  /* ---------------- Figure 1: stored versus active parameters ---------------- */
  const MODELS = [
    { id: '0.9B', total: 0.9, active: 0.9 },
    { id: '3.7B', total: 3.7, active: 3.7 },
    { id: '7B', total: 7, active: 7 },
    { id: '32B', total: 32, active: 32 },
    { id: '36B-A4B', total: 36, active: 4 },
    { id: '375B-A23B', total: 375, active: 23 },
  ];
  (function fleet() {
    const host = document.getElementById('fleetChart');
    responsive(host, (w) => {
      const narrow = w < 520;
      const L = narrow ? 76 : 92, R = 8, top = 6, rowH = 36;
      const H = top + MODELS.length * rowH + 40;
      const s = svgIn(host, w, H);
      const lo = Math.log10(0.5), hi = Math.log10(700);
      const x = (v) => L + ((Math.log10(v) - lo) / (hi - lo)) * (w - L - R - 78);
      const yEnd = top + MODELS.length * rowH;
      const grid = el('g', { class: 'grid' }, s);
      [1, 10, 100].forEach((t) => {
        el('line', { x1: x(t), x2: x(t), y1: top, y2: yEnd }, grid);
        el('text', { x: x(t), y: yEnd + 16, 'text-anchor': 'middle', class: 'tick' }, s, t + 'B').style.cssText = 'font:11.5px var(--sans);fill:var(--ink-3)';
      });
      el('text', { x: L, y: yEnd + 34, class: 'axis-title' }, s, 'parameters, log scale');
      MODELS.forEach((m, i) => {
        const y = top + i * rowH + rowH / 2;
        el('text', { x: 0, y: y + 4, class: 'label' }, s, m.id);
        const xs = x(m.total), xa = x(m.active);
        if (m.total !== m.active) {
          el('line', { x1: xa, x2: xs, y1: y, y2: y, stroke: 'var(--rule-2)', 'stroke-width': 1 }, s);
          el('circle', { cx: xs, cy: y, r: 5.5, fill: 'var(--paper)', stroke: 'var(--ink-3)', 'stroke-width': 1.25 }, s);
          el('circle', { cx: xa, cy: y, r: 4.5, fill: 'var(--active)' }, s);
          el('text', { x: xa - 10, y: y + 4, 'text-anchor': 'end', class: 'label muted' }, s, m.active + 'B active');
          el('text', { x: xs + 10, y: y + 4, class: 'label muted' }, s, m.total + 'B stored');
        } else {
          el('circle', { cx: xs, cy: y, r: 6.5, fill: 'var(--paper)', stroke: 'var(--ink-3)', 'stroke-width': 1.25 }, s);
          el('circle', { cx: xs, cy: y, r: 4.5, fill: 'var(--active)' }, s);
          el('text', { x: xs + 11, y: y + 4, class: 'label muted' }, s, m.total + 'B, all active');
        }
      });
    });
  })();

  /* ---------------- Figure 3: three layer types as small multiples ---------------- */
  const LAYERS = {
    dense: { ffn: 0, lit: [], v: 0, vlit: [], foot: '28 to 64 layers, all like this' },
    moe: { ffn: 192, lit: [3, 29, 44, 70, 101, 126, 150, 181], v: 0, vlit: [], foot: '61 layers; the first 3 are dense' },
    mova: { ffn: 100, lit: [5, 22, 35, 47, 62, 71, 86, 97], v: 64, vlit: [6, 25, 38, 57], foot: '48 layers; the first 3 are dense' },
  };
  (function layers() {
    const CELL = 9, GAP = 2, COLS = 16, GW = COLS * (CELL + GAP) - GAP;
    function grid(s, x0, y0, n, lit, cls) {
      const on = new Set(lit);
      for (let i = 0; i < n; i++) {
        const c = i % COLS, r = Math.floor(i / COLS);
        const a = { class: cls + (on.has(i) ? ' on' : ''), x: x0 + c * (CELL + GAP) + 0.5, y: y0 + r * (CELL + GAP) + 0.5, width: CELL - 1, height: CELL - 1 };
        if (on.has(i)) { a.fill = 'var(--active)'; a.stroke = 'var(--active)'; } else { a.fill = 'none'; a.stroke = 'var(--rule-2)'; }
        el('rect', a, s);
      }
      return y0 + Math.ceil(n / COLS) * (CELL + GAP) - GAP;
    }
    function box(s, x, y, w, h, label, active) {
      el('rect', { x: x + 0.5, y: y + 0.5, width: w - 1, height: h - 1, fill: active ? 'var(--active)' : 'none', stroke: active ? 'var(--active)' : 'var(--ink-3)' }, s);
      if (label) el('text', { x: x + w / 2, y: y + h / 2 + 4, 'text-anchor': 'middle', class: 'label' }, s, label).style.fill = active ? '#fff' : 'var(--ink)';
    }
    function arrow(s, x, y1, y2) {
      el('line', { x1: x, x2: x, y1, y2: y2 - 1, stroke: 'var(--ink-3)' }, s);
      el('path', { d: `M${x - 3.5},${y2 - 6} L${x},${y2} L${x + 3.5},${y2 - 6}`, fill: 'none', stroke: 'var(--ink-3)' }, s);
    }
    document.querySelectorAll('#layerMultiples .panel').forEach((panel) => {
      const v = LAYERS[panel.dataset.variant];
      const host = panel.querySelector('.graphic');
      responsive(host, (w) => {
        // Side by side, all three panels share one height so their blocks line
        // up; stacked on a phone, each panel is only as tall as its content.
        const single = panel.parentElement.clientWidth < 2 * GW + 120;
        const x0 = 0, cx = x0 + GW / 2;
        const s = svgIn(host, w, 424);
        el('text', { x: x0, y: 12, class: 'label muted' }, s, 'one token’s hidden vector');
        arrow(s, cx, 20, 38);
        // attention: Q and K are always single projections; V is one box or a routed pool
        el('text', { x: x0, y: 54, class: 'label' }, s, 'Attention');
        box(s, x0, 62, 36, 22, 'Q', true);
        box(s, x0 + 44, 62, 36, 22, 'K', true);
        let top;
        if (v.v) {
          el('text', { x: x0, y: 104, class: 'label' }, s, `V: 4 of ${v.v} value experts`);
          top = grid(s, x0, 112, v.v, v.vlit, 'vx') + 10;
        } else {
          box(s, x0 + 88, 62, 36, 22, 'V', true);
          el('text', { x: x0, y: 104, class: 'label muted' }, s, 'one value projection');
          top = single ? 112 : 166;
        }
        arrow(s, cx, top, top + 18);
        const f = top + 34;
        el('text', { x: x0, y: f, class: 'label' }, s, 'Feed-forward');
        let yb;
        if (!v.ffn) {
          box(s, x0, f + 8, GW, 36, 'one network', true);
          el('text', { x: x0, y: f + 62, class: 'label muted' }, s, 'all weights used for every token');
          yb = f + 62;
        } else {
          el('text', { x: x0, y: f + 18, class: 'label muted' }, s, `router picks 8 of ${v.ffn} experts`);
          const end = grid(s, x0, f + 26, v.ffn, v.lit, 'ex');
          box(s, x0, end + 8, GW, 12, '', true);
          el('text', { x: x0, y: end + 36, class: 'label muted' }, s, '+ 1 shared expert, always used');
          yb = end + 36;
        }
        const H = single ? yb + 44 : 424;
        s.setAttribute('height', H); s.setAttribute('viewBox', `0 0 ${w} ${H}`);
        el('line', { x1: x0, x2: GW, y1: H - 22, y2: H - 22, stroke: 'var(--rule)' }, s);
        el('text', { x: x0, y: H - 6, class: 'annot' }, s, v.foot);
      });
    });
  })();

  /* ---------------- Figure 4: where the 36B-A4B's parameters sit ---------------- */
  // Billions of non-embedding parameters, computed from config.json and
  // modeling_k2_horizon.py (see README for the arithmetic).
  const LEDGER = [
    { name: 'Routed feed-forward experts', stored: 26.542, active: 2.123, note: '8 of 100 per layer' },
    { name: 'Value experts (MoVA)', stored: 7.5497, active: 0.4719, note: '4 of 64 per layer', focus: true },
    { name: 'Attention: Q, K, O and gate', stored: 1.6437, active: 1.6437 },
    { name: 'Shared experts, dense FFNs', stored: 0.4070, active: 0.4070 },
  ];
  (function ledger() {
    const host = document.getElementById('movaLedger');
    responsive(host, (w) => {
      const narrow = w < 560;
      const L = narrow ? 0 : 186, R = narrow ? 4 : 160;
      const rowH = narrow ? 62 : 40, top = 4;
      const H = top + LEDGER.length * rowH + 4;
      const s = svgIn(host, w, H);
      const x = (v) => L + (v / 26.542) * (w - L - R);
      LEDGER.forEach((d, i) => {
        const y0 = top + i * rowH;
        const by = narrow ? y0 + 20 : y0 + 12;
        const cls = 'label' + (d.focus ? '' : ' muted');
        el('text', { x: 0, y: narrow ? y0 + 13 : by + 11, class: cls }, s, d.name).style.fontWeight = d.focus ? 600 : 400;
        el('rect', { x: L, y: by, width: Math.max(1, x(d.stored) - L), height: 14, fill: 'var(--c-grey)', 'fill-opacity': 0.55 }, s);
        el('rect', { x: L, y: by, width: Math.max(1, x(d.active) - L), height: 14, fill: 'var(--active)' }, s);
        const b1 = (v) => (v >= 1 ? f1(v) : f2(v)) + "B";
        const txt = d.stored === d.active ? `${b1(d.stored)}, all used` : `${b1(d.stored)} stored · ${b1(d.active)} used`;
        const t = narrow ? el('text', { x: 0, y: by + 30, class: 'label muted' }, s, txt + (d.note ? ` (${d.note})` : ''))
          : el('text', { x: x(d.stored) + 8, y: by + 11, class: 'label muted' }, s, txt);
        t.style.fontSize = '12px';
      });
    });
  })();

  /* ---------------- Figure 5: Uno decoding, simulated ---------------- */
  const WORDS = 'Uno drafts a block of tokens in one pass and the base model checks them in the next pass so every round adds at least two tokens and at most one more than the block size'.split(' ');
  const ALTS = ['the', 'a', 'it', 'then', 'one', 'model', 'token', 'each', 'and', 'more', 'first', 'next', 'with'];
  const alt = (i) => { let k = (i * 7 + 3) % ALTS.length; while (ALTS[k] === WORDS[i]) k = (k + 1) % ALTS.length; return ALTS[k]; };
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  // One simulation. tick() advances one forward pass for both decoders.
  function makeRun(B, p, seed) {
    const N = WORDS.length, rand = mulberry32(seed);
    const run = {
      B, p, seed, N,
      ar: { pos: 0, passes: 0, hist: [[0, 0]] },
      uno: { pos: 0, passes: 0, phase: 'draft', draft: null, out: [], hist: [[0, 0]] },
      done() { return this.ar.pos >= N && this.uno.pos >= N; },
      tick() {
        let moved = false;
        const a = this.ar, u = this.uno;
        if (a.pos < N) { a.pos++; a.passes++; a.hist.push([a.passes, a.pos]); moved = true; }
        if (u.pos < N) {
          moved = true; u.passes++;
          if (u.phase === 'draft') {
            // The draft pass proposes up to B tokens. The first comes from the
            // base weights and is always accepted; each later one passes with
            // the invented probability p until the first rejection.
            const len = Math.min(B, N - u.pos);
            let acc = 1;
            while (acc < len && rand() < p) acc++;
            u.draft = { start: u.pos, len, acc };
            u.phase = 'verify';
          } else {
            const d = u.draft;
            for (let i = 0; i < d.acc; i++) u.out.push({ w: WORDS[d.start + i], kind: 'acc' });
            let pos = d.start + d.acc;
            if (d.acc < d.len) u.out.push({ w: alt(pos), kind: 'rej' });
            if (pos < N) { u.out.push({ w: WORDS[pos], kind: 'ver' }); pos++; } // replacement or bonus token
            u.pos = pos; u.draft = null; u.phase = 'draft';
          }
          u.hist.push([u.passes, u.pos]);
        }
        return moved;
      },
    };
    return run;
  }
  function simulate(B, p, seed) { const r = makeRun(B, p, seed); while (r.tick()); return { tokens: r.uno.pos, passes: r.uno.passes, rejected: r.uno.out.filter((t) => t.kind === 'rej').length }; }

  (function uno() {
    const play = document.getElementById('unoPlay'), step = document.getElementById('unoStep'), reset = document.getElementById('unoReset');
    const acc = document.getElementById('unoAcc'), accOut = document.getElementById('unoAccOut');
    const blockBtns = [...document.querySelectorAll('#unoBlock button')];
    const arText = document.getElementById('arText'), unoText = document.getElementById('unoText');
    const arCount = document.getElementById('arCount'), unoCount = document.getElementById('unoCount');
    const chartHost = document.getElementById('unoChart');
    let B = 4, p = 0.7, seed = 1, run, timer = null;

    const span = (w, cls) => `<span class="tok ${cls}">${w}</span>`;
    function renderText() {
      const a = run.ar, u = run.uno;
      arText.innerHTML = WORDS.map((w, i) => span(w, i < a.pos ? 'ar' : 'todo')).join(' ');
      let html = u.out.map((t) => span(t.w, t.kind));
      if (u.draft) {
        const d = u.draft;
        for (let i = 0; i < d.len; i++) html.push(span(i === d.acc ? alt(d.start + i) : WORDS[d.start + i], 'pend'));
      }
      const from = u.draft ? u.draft.start + u.draft.len : u.pos;
      for (let i = from; i < WORDS.length; i++) html.push(span(WORDS[i], 'todo'));
      unoText.innerHTML = html.join(' ');
      arCount.textContent = `${a.pos} tokens · ${a.passes} passes`;
      unoCount.textContent = `${u.pos} tokens · ${u.passes} passes · ${u.passes ? f2(u.pos / u.passes) : '–'} per pass`;
    }
    const redrawChart = responsive(chartHost, (w) => drawChart(w));
    function drawChart(w) {
      if (!run) return;
      const N = run.N, L = 34, R = 10, T = 10, Bm = 40, H = 200;
      const s = svgIn(chartHost, w, H);
      const x = (v) => L + (v / N) * (w - L - R), y = (v) => T + (1 - v / N) * (H - T - Bm);
      const axis = el('g', { class: 'axis' }, s);
      el('line', { x1: L, x2: x(N), y1: y(0) + 6, y2: y(0) + 6 }, axis);
      el('line', { x1: L - 6, x2: L - 6, y1: y(0), y2: y(N) }, axis);
      [[0, 'start'], [N, 'end']].forEach(([v, anchor]) => {
        el('text', { x: x(v), y: y(0) + 20, 'text-anchor': anchor, class: 'tick' }, s, String(v)).style.cssText = 'font:11.5px var(--sans);fill:var(--ink-3)';
        el('text', { x: L - 10, y: y(v) + 4, 'text-anchor': 'end' }, s, String(v)).style.cssText = 'font:11.5px var(--sans);fill:var(--ink-3)';
      });
      el('text', { x: x(N / 2), y: H - 4, 'text-anchor': 'middle', class: 'axis-title' }, s, 'forward passes');
      el('text', { x: L + 2, y: T + 2, class: 'axis-title' }, s, 'tokens written');
      const path = (hist) => hist.map(([px, t], i) => (i ? 'L' : 'M') + x(px).toFixed(1) + ',' + y(t).toFixed(1)).join('');
      el('path', { d: path(run.ar.hist), fill: 'none', stroke: 'var(--c-grey)', 'stroke-width': 1.5 }, s);
      el('path', { d: path(run.uno.hist), fill: 'none', stroke: 'var(--accept)', 'stroke-width': 2, 'stroke-linejoin': 'round' }, s);
      const a = run.ar, u = run.uno;
      // Label the grey line where it is: below and right of its current end
      // while it is short, in the empty lower-right corner once it is long.
      const arLabel = w < 560 ? 'autoregressive' : 'autoregressive: 1 per pass';
      if (a.passes && a.passes < N * 0.6) el('text', { x: x(a.passes) + 6, y: y(a.pos) + 18, class: 'label muted' }, s, arLabel);
      else if (a.passes) el('text', { x: x(N), y: y(N * 0.3), 'text-anchor': 'end', class: 'label muted' }, s, arLabel);
      if (u.passes) {
        const ux = x(u.passes), uy = y(u.pos);
        el('circle', { cx: ux, cy: uy, r: 3, fill: 'var(--accept)' }, s);
        const lbl = el('text', { x: ux + 8, y: Math.max(T + 22, uy + 4), class: 'label' }, s, w < 560 ? 'Uno' : `Uno: ${f2(u.pos / u.passes)} per pass`);
        lbl.style.fill = 'var(--accept)';
        lbl.style.fontWeight = 600;
      }
      if (a.passes) el('circle', { cx: x(a.passes), cy: y(a.pos), r: 2.5, fill: 'var(--c-grey)' }, s);
    }
    function render() { renderText(); redrawChart(); }
    function stop() { if (timer) { clearInterval(timer); timer = null; } play.textContent = 'Play'; play.setAttribute('aria-pressed', 'false'); }
    function fresh() { run = makeRun(B, p, seed); }
    function complete() { fresh(); while (run.tick()); }
    function advance() { const moved = run.tick(); render(); if (!moved || run.done()) stop(); }
    play.addEventListener('click', () => {
      if (timer) { stop(); return; }
      if (run.done()) { seed++; fresh(); render(); }
      timer = setInterval(advance, reduced ? 900 : 600);
      play.textContent = 'Pause'; play.setAttribute('aria-pressed', 'true');
    });
    step.addEventListener('click', () => { stop(); if (run.done()) { seed++; fresh(); } advance(); });
    reset.addEventListener('click', () => { stop(); fresh(); render(); });
    blockBtns.forEach((b) => b.addEventListener('click', () => {
      blockBtns.forEach((o) => o.setAttribute('aria-pressed', String(o === b)));
      B = +b.dataset.block; stop(); complete(); render();
    }));
    acc.addEventListener('input', () => { p = +acc.value / 100; accOut.textContent = acc.value + '%'; stop(); complete(); render(); });
    complete(); render(); // resting state: a finished run at the default settings
  })();

  /* ---------------- Figure 8: reported benchmark scores ---------------- */
  // Transcribed from the "Full results" tables on ifm.ai/blog/k2 (3 Sep 2026).
  // null = not reported. elo rows are GDPVal-AA ratings, not percentages.
  const _ = null;
  const BENCH = {
    '0.9B': { k2: 'K2-Horizon-0.9B', comps: ['Qwen3.5-0.8B', 'OpenBMB-1B', 'Qwen3.5-2B'], rows: [
      ['HumanEval+', 79.9, 16.5, 65.2, 75.6], ['MBPP+', 68.0, 35.4, 60.6, 67.7], ['BFCL v4', 28.0, 25.3, 25.2, 43.6],
      ['AIME 2025', 41.7, 1.0, 40.4, 34.2], ['AIME 2026', 48.5, 0.2, 40.4, 38.8], ['HMMT Feb 2026', 25.8, 0.6, 23.3, 22.7],
      ['LiveCodeBench v6', 37.4, 6.6, 33.5, 29.8], ['GPQA Diamond', 27.3, 11.9, 26.3, 54.9]] },
    '3.7B': { k2: 'K2-Horizon-3.7B', comps: ['Qwen3.5-4B', 'G9v3-3B', 'Granite 4.2-3B', 'Granite 4.2-8B', 'Nemotron 3 Nano-4B', 'Nemotron 3.5 30B-A3B', 'Gemma 4-E4B'], rows: [
      ['SWE-bench Verified', 68.6, 41.2, 16.4, 32.2, _, _, 51.6, _], ['Terminal-Bench 2.1', 25.1, 25.8, 6.0, 13.9, _, 3.7, _, 2.0],
      ['BFCL v4', 50.9, 55.7, 47.9, 50.8, 48.4, 36.8, _, _], ['tau3-Banking', 17.7, 6.8, _, 5.6, _, _, 9.3, 5.0],
      ['HMMT Feb 2026', 70.5, 61.6, 34.1, 57.2, _, 34.7, _, _], ['SciCode', 25.9, 16.1, 17.7, 24.9, _, 16.4, _, _],
      ['GPQA Diamond', 65.4, 77.1, 43.8, 55.9, _, 51.3, _, 52.0], ["Humanity's Last Exam", 12.9, 9.9, 4.5, 6.6, _, 4.9, _, _]] },
    '7B': { k2: 'K2-Horizon-7B', comps: ['Qwen3.5-9B', 'Gemma 4-12B', 'Granite 4.2-8B'], rows: [
      ['SWE-bench Verified', 70.6, 50.8, 30.6, 47.7], ['Terminal-Bench 2.1', 39.1, 29.2, 27.3, 18.4], ['tau3-Banking', 25.8, 7.0, _, 7.6],
      ['BrowseComp', 59.0, _, _, _], ['AA-LCR', 68.0, 65.3, 61.7, 43.3], ['HMMT Feb 2026', 73.3, 65.7, 63.1, 66.5],
      ['SciCode', 31.6, 27.5, 38.2, 30.4], ["Humanity's Last Exam", 18.6, 14.9, 15.7, 9.7]] },
    '32B': { k2: 'K2-Horizon-32B (Stage 1)', comps: ['Qwen3.8-27B', 'Muse Glimmer-30B', 'IBM Granite 4.2 30B'], rows: [
      ['tau3-Banking', 22.5, 48.0, 23.5, 14.4], ['Terminal-Bench 2.1', 36.6, 79.8, 51.7, 26.6], ['SciCode', 30.2, 44.7, 43.6, 36.6],
      ["Humanity's Last Exam", 22.8, 33.9, 22.0, 11.2], ['GPQA Diamond', 82.3, 90.5, 83.5, 64.4], ['CritPt', 1.4, 5.4, 2.6, 0.3],
      ['AA-LCR', 65.3, 77.3, 80.0, 46.7], ['AA-Omniscience accuracy', 16.8, 15.6, 27.0, 10.1], ['AA-Omniscience non-hallucination', 58.3, 69.7, 18.1, 74.4]] },
    '36B-A4B': { k2: 'K2-Horizon-MoVA-36B-A4B', comps: ['Nemotron 3 Ultra', 'Nemotron 3 Super', 'G9v3-39A5B', 'Qwen3.6-35B-A3B', 'Muse Glimmer-30B', 'Gemma 4 31B-it'], rows: [
      ['tau3-Banking', 26.8, 14.2, 10.3, 22.1, 9.3, 23.5, 14.8], ['Terminal-Bench 2.1', 58.6, 53.9, 38.6, 32.6, 44.9, 51.7, 43.4],
      ['SciCode', 38.9, 39.9, 36.0, 34.0, 35.8, 43.6, 43.4], ["Humanity's Last Exam", 25.2, 28.4, 20.8, 17.5, 22.2, 22.0, 23.6],
      ['GPQA Diamond', 80.8, 86.7, 80.0, 80.5, 84.1, 83.5, 85.7], ['CritPt', 2.1, 3.1, 3.1, 0.3, 0.3, 2.6, 1.4],
      ['AA-LCR', 66.3, 71.0, 60.3, 62.0, 66.7, 80.0, 68.3], ['AA-Omniscience accuracy', 18.8, 22.6, 24.3, 14.9, 18.8, 27.0, 20.0],
      ['AA-Omniscience non-hallucination', 69.2, 70.3, 13.0, 87.0, 49.5, 18.1, 15.0]] },
    '375B-A23B': { k2: 'K2-Horizon-375B-A23B', comps: ['Nemotron 3 Ultra', 'Inkling (xhigh)', 'MiniMax-M3 (max)', 'GLM 5.2 (max)', 'GPT 5.6 Luna (max)', 'GPT 5.6 Terra (high)', 'Claude Sonnet 5 (max)'], rows: [
      ['GDPVal-AA', 1441, 1162, 1234, 1380, 1498, 1569, 1503, 1584],
      ['tau3-Banking', 34.0, 14.2, 29.1, 15.3, 34.6, 31.1, 28.7, 37.3], ['Toolathlon Verified', 65.3, 34.3, 45.5, 53.7, 59.9, 67.5, 64.8, 71.6],
      ['Automation Bench Public', 25.3, 8.0, 12.8, 20.5, 26.2, 33.5, 28.0, 34.7], ['Apex-Agents (pass@1)', 24.8, 9.0, 19.0, 23.8, 26.9, 28.6, 25.4, 31.7],
      ['MCPMark', 67.7, 45.7, 51.2, 48.8, 72.4, 66.9, 74.0, 65.3], ['BrowseComp', 72.8, 44.4, 77.1, 83.5, _, 83.3, _, 84.7],
      ['WildClawBench', 50.9, 34.2, 52.3, 56.4, 55.0, 50.4, 60.0, _], ['Terminal-Bench 2.1', 70.2, 53.9, 55.1, 65.2, 77.9, 80.9, 75.7, 80.5],
      ['SciCode', 42.7, 39.9, 46.1, 45.4, 50.5, 52.5, 50.1, 53.6], ['SWE-Atlas-QnA (strict)', 48.4, _, 25.5, 42.3, 46.4, _, _, _],
      ['SWE Bench Pro (strict)', 42.6, 38.7, 43.1, 43.8, 46.7, 48.8, _, _], ["Humanity's Last Exam", 32.0, 28.4, 31.9, 39.0, 41.1, 39.5, 38.5, 41.3],
      ['GPQA Diamond', 87.3, 86.7, 87.2, 92.9, 89.5, 91.1, 89.6, 91.1], ['CritPt', 8.6, 3.1, 5.4, 3.7, 20.9, 21.0, 22.9, 16.9],
      ['AA-LCR', 76.0, 71.0, 73.3, 80.3, 76.7, 78.3, 73.3, 77.0], ['AA-Omniscience accuracy', 23.0, 23.0, 42.0, 17.0, 24.0, 43.0, 45.0, 40.0],
      ['AA-Omniscience non-hallucination', 74.7, 70.0, 32.0, 82.0, 74.0, 7.0, 10.0, 61.0]],
      elo: ['GDPVal-AA'], audited: { 'Terminal-Bench 2.1': 66.9 } },
  };
  function rowStats(b, r) {
    const k2 = r[1];
    let best = null, bestName = null;
    b.comps.forEach((c, i) => { const v = r[i + 2]; if (v != null && (best == null || v > best)) { best = v; bestName = c; } });
    return { k2, best, bestName, leads: best == null ? null : k2 >= best };
  }
  function summary(id) {
    const b = BENCH[id];
    let lead = 0, n = 0;
    b.rows.forEach((r) => { const st = rowStats(b, r); if (st.leads != null) { n++; if (st.leads) lead++; } });
    return { lead, n };
  }

  (function bench() {
    const seg = document.getElementById('benchModel');
    const host = document.getElementById('benchChart');
    const table = document.getElementById('benchTable');
    const takeaway = document.getElementById('benchTakeaway');
    let current = '375B-A23B';
    Object.keys(BENCH).forEach((id) => {
      const b = document.createElement('button');
      b.type = 'button'; b.dataset.model = id; b.textContent = id;
      b.setAttribute('aria-pressed', String(id === current));
      b.addEventListener('click', () => { current = id; select(); });
      seg.appendChild(b);
    });
    const redraw = responsive(host, (w) => draw(w));
    function select() {
      seg.querySelectorAll('button').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.model === current)));
      redraw(); fillTable();
      const { lead, n } = summary(current);
      takeaway.textContent = `${BENCH[current].k2} has the top score in ${lead} of the ${n} rows that list a comparison.`;
    }
    function fillTable() {
      const b = BENCH[current];
      const head = `<thead><tr><th scope="col">Benchmark</th><th class="n k2" scope="col">${b.k2}</th>${b.comps.map((c) => `<th class="n" scope="col">${c}</th>`).join('')}</tr></thead>`;
      const body = b.rows.map((r) => {
        const elo = b.elo && b.elo.includes(r[0]);
        const cells = r.slice(1).map((v, i) => `<td class="n${i === 0 ? ' k2' : ''}${v == null ? ' dim' : ''}">${v == null ? '–' : (elo ? v.toLocaleString('en-US') : f1(v))}</td>`).join('');
        return `<tr><th scope="row">${r[0]}${elo ? ' (Elo)' : ''}</th>${cells}</tr>`;
      }).join('');
      table.innerHTML = head + '<tbody>' + body + '</tbody>';
    }
    function draw(w) {
      const b = BENCH[current];
      const narrow = w < 640;
      const isElo = (r) => !!(b.elo && b.elo.includes(r[0]));
      const pctRows = b.rows.filter((r) => !isElo(r));
      const eloRows = b.rows.filter(isElo);
      const L = narrow ? 4 : 206, R = narrow ? 8 : 318;
      const pitch = narrow ? 64 : 28;
      const head = narrow ? 34 : 22;
      const eloH = eloRows.length ? head + pitch + (narrow ? 22 : 26) : 0;
      const H = eloH + head + pctRows.length * pitch + 8;
      const s = svgIn(host, w, H, `Dot plot of ${b.rows.length} reported benchmark scores for ${b.k2}; the table below lists every value.`);
      const x0 = L, x1 = w - R;
      const tick = (x, y, t, anchor) => { el('text', { x, y, 'text-anchor': anchor || 'middle' }, s, t).style.cssText = 'font:11.5px var(--sans);fill:var(--ink-3)'; };
      // One benchmark: its label, a hairline for the scale, grey dots for the
      // comparison models, a blue dot for K2, and K2 versus the best other.
      function row(r, yTop, sc, fmt) {
        const st = rowStats(b, r);
        const sy = narrow ? yTop + 30 : yTop + pitch / 2;
        const ty = narrow ? yTop + 15 : sy + 4;
        el('text', { x: 0, y: ty, class: 'label' }, s, r[0]);
        el('line', { x1: x0, x2: x1, y1: sy, y2: sy, stroke: 'var(--rule)' }, s);
        b.comps.forEach((c, i) => { const v = r[i + 2]; if (v != null) el('circle', { class: 'other', cx: sc(v), cy: sy, r: 4, fill: 'var(--c-grey)', 'fill-opacity': 0.75 }, s); });
        const aud = b.audited ? b.audited[r[0]] : undefined;
        if (aud != null) el('circle', { class: 'audited', cx: sc(aud), cy: sy, r: 4.5, fill: 'var(--paper)', stroke: 'var(--k2)', 'stroke-width': 1.25 }, s);
        el('circle', { class: 'k2', cx: sc(st.k2), cy: sy, r: 5, fill: 'var(--k2)', stroke: 'var(--paper)', 'stroke-width': 1.5 }, s);
        const t = el('text', { class: 'label row-note', x: narrow ? 0 : x1 + 18, y: narrow ? yTop + 52 : ty }, s);
        el('tspan', { style: 'fill:var(--k2);font-weight:600' }, t, 'K2 ' + fmt(st.k2));
        if (aud != null) el('tspan', { style: 'fill:var(--ink-2)' }, t, ` (${f1(aud)} audited)`);
        el('tspan', { style: 'fill:var(--ink-2)' }, t, st.best == null ? ' · no comparison reported' : ` · best other ${fmt(st.best)}, ${st.bestName}`);
        if (narrow) t.style.fontSize = '12px';
      }
      let y = 0;
      if (eloRows.length) {
        const lo = 1100, hi = 1650, sc = (v) => x0 + ((v - lo) / (hi - lo)) * (x1 - x0);
        el('text', { x: 0, y: 13, class: 'axis-title' }, s, 'Elo rating, own scale');
        y += head;
        eloRows.forEach((r) => row(r, y, sc, (v) => v.toLocaleString('en-US')));
        const sy = narrow ? y + 30 : y + pitch / 2;
        [1200, 1400, 1600].forEach((t) => tick(sc(t), sy + (narrow ? 36 : 20), t.toLocaleString('en-US')));
        y = eloH;
      }
      const sc = (v) => x0 + (v / 100) * (x1 - x0);
      el('text', { x: 0, y: y + 13, class: 'axis-title' }, s, 'Score, %');
      [0, 50, 100].forEach((t) => tick(sc(t), y + (narrow ? 29 : 13), String(t), t === 0 ? 'start' : t === 100 ? 'end' : 'middle'));
      y += head;
      const grid = el('g', { class: 'grid' }, s);
      [0, 50, 100].forEach((t) => el('line', { x1: sc(t), x2: sc(t), y1: y, y2: y + pctRows.length * pitch }, grid));
      pctRows.forEach((r, i) => row(r, y + i * pitch, sc, f1));
    }
    select();
  })();

  // Exposed for the browser tests: pure functions only.
  window.K2 = { simulate, summary, WORDS_N: WORDS.length };
})();
