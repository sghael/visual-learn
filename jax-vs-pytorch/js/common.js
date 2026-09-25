/* ------------------------------------------------------------------
   JT: the tiny shared toolkit for this page's figures. No dependencies.
   Every figure file registers itself with JT.widget('name', (el) => {...})
   and main.js mounts it on the element with [data-widget="name"].
   Nothing here draws chrome: figures are plain content inside <figure>.
   ------------------------------------------------------------------ */
(function () {
  'use strict';

  const JT = {};
  const registry = new Map();

  JT.reducedMotion = !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);

  JT.$ = (sel, root) => (root || document).querySelector(sel);
  JT.$$ = (sel, root) => Array.from((root || document).querySelectorAll(sel));
  JT.clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  JT.escape = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  function applyAttrs(node, attrs, isSvg) {
    if (!attrs) return;
    for (const [k, v] of Object.entries(attrs)) {
      if (v === null || v === undefined || v === false) continue;
      if (k === 'class') node.setAttribute('class', v);
      else if (k === 'text') node.textContent = v;
      else if (k === 'html') node.innerHTML = v;
      else if (k === 'style' && typeof v === 'object') Object.assign(node.style, v);
      else if (k === 'dataset') Object.assign(node.dataset, v);
      else if (k.startsWith('on') && typeof v === 'function') node.addEventListener(k.slice(2).toLowerCase(), v);
      else if (!isSvg && k in node && typeof v !== 'object' && k !== 'list') { try { node[k] = v; } catch (e) { node.setAttribute(k, v); } }
      else node.setAttribute(k, v === true ? '' : v);
    }
  }
  function appendChildren(node, children) {
    if (children === undefined || children === null) return;
    const list = Array.isArray(children) ? children : [children];
    for (const c of list) {
      if (c === null || c === undefined || c === false || c === '') continue;
      node.appendChild(typeof c === 'string' || typeof c === 'number' ? document.createTextNode(String(c)) : c);
    }
  }
  /** JT.el('div', {class:'x', text:'hi', onClick}, [children]) */
  JT.el = (tag, attrs, children) => { const n = document.createElement(tag); applyAttrs(n, attrs, false); appendChildren(n, children); return n; };
  /** JT.svg('rect', {x:0, y:0, width:10, height:10}) */
  JT.svg = (tag, attrs, children) => { const n = document.createElementNS('http://www.w3.org/2000/svg', tag); applyAttrs(n, attrs, true); appendChildren(n, children); return n; };

  /* ---------------------------------------------------------------- */
  /* Syntax highlighting (python + jaxpr). Lines are .ln[data-line].    */
  /* ---------------------------------------------------------------- */
  const PY_KW = /^(?:def|return|if|else|elif|for|while|in|not|and|or|import|from|as|with|class|lambda|None|True|False|pass|yield|is|del|global|nonlocal|try|except|raise|assert|break|continue)$/;
  const PY_RE = /(#.*$)|("""[\s\S]*?"""|'''[\s\S]*?'''|f?"(?:[^"\\]|\\.)*"|f?'(?:[^'\\]|\\.)*')|(@[A-Za-z_][\w.]*)|(\b\d+(?:\.\d*)?(?:e[+-]?\d+)?\b)|(\b[A-Za-z_]\w*\b)(?=\s*\()|(\b[A-Za-z_]\w*\b)/gm;
  function highlightPython(src) {
    return src.split('\n').map((line, i) => {
      const out = line.replace(PY_RE, (m, cm, str, dec, num, fn, word) => {
        if (cm) return `<span class="tok-cm">${JT.escape(cm)}</span>`;
        if (str) return `<span class="tok-str">${JT.escape(str)}</span>`;
        if (dec) return `<span class="tok-dec">${JT.escape(dec)}</span>`;
        if (num) return `<span class="tok-num">${JT.escape(num)}</span>`;
        if (fn) return PY_KW.test(fn) ? `<span class="tok-kw">${fn}</span>` : fn;
        if (word) {
          if (PY_KW.test(word)) return `<span class="tok-kw">${word}</span>`;
          if (/^(jax|jnp|lax|optax)$/.test(word)) return `<span class="tok-j">${word}</span>`;
          if (/^(torch|nn|F)$/.test(word)) return `<span class="tok-t">${word}</span>`;
          return word;
        }
        return JT.escape(m);
      });
      return `<span class="ln" data-line="${i + 1}">${out || ' '}</span>`;
    }).join('');
  }
  const JAXPR_RE = /(\{ lambda|\blambda\b|\blet\b|\bin\b)|(\b(?:f32|i32|bool|bf16)\[[\d, ]*\])|(\b\d+(?:\.\d+)?\b)/g;
  function highlightJaxpr(src) {
    return src.split('\n').map((line, i) => {
      const out = JT.escape(line).replace(JAXPR_RE, (m, kw, ty, num) => {
        if (kw) return `<span class="tok-kw">${kw}</span>`;
        if (ty) return `<span class="tok-num">${ty}</span>`;
        if (num) return `<span class="tok-num">${num}</span>`;
        return m;
      });
      return `<span class="ln" data-line="${i + 1}">${out || ' '}</span>`;
    }).join('');
  }
  JT.highlight = (src, lang) => (lang === 'jaxpr' ? highlightJaxpr(src) : highlightPython(src));

  /** Build a <pre class="code">. Lines are .ln[data-line]. opts: {lang, class} */
  JT.code = (src, opts) => {
    const o = Object.assign({ lang: 'python', class: '' }, opts || {});
    return JT.el('pre', { class: ('code ' + o.class).trim(), html: JT.highlight(src.replace(/^\n+|\n+$/g, ''), o.lang) });
  };
  /** A code block that shows a one-line placeholder instead of code. */
  JT.placeholder = (text) => JT.el('pre', { class: 'code', html: `<span class="ln placeholder">${JT.escape(text)}</span>` });
  /** Toggle a class on some line numbers ('*' for all) of a code block. */
  JT.markLines = (pre, lines, cls, on) => {
    JT.$$('.ln', pre).forEach((ln) => {
      const n = Number(ln.dataset.line);
      if (lines === '*' || lines.includes(n)) ln.classList.toggle(cls, on !== false);
    });
  };
  JT.clearLines = (pre, cls) => JT.$$('.ln.' + cls, pre).forEach((ln) => ln.classList.remove(cls));

  /* ---------------------------------------------------------------- */
  /* Controls                                                           */
  /* ---------------------------------------------------------------- */
  JT.button = (label, onClick, attrs) => JT.el('button', Object.assign({ class: 'btn', type: 'button', text: label, onClick }, attrs || {}));

  /**
   * Step controller. Nothing moves until the reader presses Play or Step.
   * opts: { total, onStep(i), onReset(), interval, noun }
   * api: { el, next, prev, play, pause, reset, goto, i, playing }
   */
  JT.stepper = (opts) => {
    const o = Object.assign({ total: 1, interval: 1000, noun: 'step' }, opts);
    let i = -1, timer = null, playing = false;
    const readout = JT.el('output', { class: 'readout', 'aria-live': 'polite' });
    const btnPlay = JT.button('Play', () => api.toggle(), { class: 'btn primary', dataset: { role: 'play' } });
    const btnPrev = JT.button('Back', () => { api.pause(); api.prev(); }, { dataset: { role: 'back' } });
    const btnNext = JT.button('Step', () => { api.pause(); api.next(); }, { dataset: { role: 'step' } });
    const btnReset = JT.button('Reset', () => api.reset(), { dataset: { role: 'reset' } });
    const el = JT.el('span', { class: 'control stepper' }, [btnPlay, btnPrev, btnNext, btnReset, readout]);

    function render() {
      readout.textContent = `${o.noun} ${i + 1} of ${o.total}`;
      btnPrev.disabled = i < 0;
      btnNext.disabled = i >= o.total - 1;
      btnPlay.textContent = playing ? 'Pause' : (i >= o.total - 1 ? 'Replay' : 'Play');
    }
    const api = {
      el,
      get i() { return i; },
      get playing() { return playing; },
      get total() { return o.total; },
      goto(n) { i = JT.clamp(n, -1, o.total - 1); if (i >= 0) o.onStep(i); else if (o.onReset) o.onReset(); render(); return api; },
      next() { if (i < o.total - 1) api.goto(i + 1); return api; },
      prev() { if (i >= 0) api.goto(i - 1); return api; },
      reset() { api.pause(); api.goto(-1); return api; },
      play() {
        if (i >= o.total - 1) api.goto(-1);
        playing = true;
        const tick = () => {
          timer = null;
          if (!playing) return;
          api.next();
          if (i >= o.total - 1) { api.pause(); return; }
          timer = setTimeout(tick, JT.reducedMotion ? 60 : o.interval);
        };
        tick();
        return api;
      },
      pause() { playing = false; if (timer) clearTimeout(timer); timer = null; render(); return api; },
      toggle() { return playing ? api.pause() : api.play(); },
      destroy() { api.pause(); },
    };
    render();
    return api;
  };

  /** Segmented control for a handful of named cases. options: [{value, label}] */
  JT.seg = (options, onChange, initial, label, cls) => {
    const el = JT.el('span', { class: ('seg ' + (cls || '')).trim(), role: 'group', 'aria-label': label || null });
    let value = initial === undefined ? options[0].value : initial;
    const buttons = options.map((opt) => {
      const b = JT.el('button', { type: 'button', text: opt.label, 'aria-pressed': String(opt.value === value), dataset: { value: opt.value } });
      b.addEventListener('click', () => { if (value === opt.value) return; el.setValue(opt.value); onChange(value); });
      return b;
    });
    buttons.forEach((b) => el.appendChild(b));
    el.setValue = (v) => { value = v; buttons.forEach((x) => x.setAttribute('aria-pressed', String(x.dataset.value === String(v)))); };
    Object.defineProperty(el, 'value', { get: () => value });
    return el;
  };

  /** A labeled control group: <span class="control"><span class="label">…</span>…</span> */
  JT.control = (label, children) => JT.el('span', { class: 'control' }, [label ? JT.el('span', { class: 'label', text: label }) : null].concat(children));

  /** Call cb(width) now and whenever the element's width changes. Figures draw at real CSS pixels. */
  JT.onWidth = (el, cb) => {
    let last = -1;
    const run = () => { const w = Math.round(el.clientWidth); if (w > 0 && w !== last) { last = w; cb(w); } };
    if ('ResizeObserver' in window) new ResizeObserver(run).observe(el);
    else window.addEventListener('resize', run);
    run();
  };

  JT.f3 = (v) => { const r = (Math.round(v * 1000) / 1000).toFixed(3); return r === '-0.000' ? '0.000' : r; };

  /* ---------------------------------------------------------------- */
  /* Registry                                                           */
  /* ---------------------------------------------------------------- */
  JT.widget = (name, init) => { registry.set(name, init); };
  JT.mountAll = () => {
    JT.$$('[data-widget]').forEach((el) => {
      const init = registry.get(el.dataset.widget);
      if (!init) { console.error('No widget registered for', el.dataset.widget); return; }
      try { init(el); } catch (e) { console.error('Widget failed:', el.dataset.widget, e); }
    });
  };

  window.JT = JT;
})();
