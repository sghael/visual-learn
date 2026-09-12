/* ------------------------------------------------------------------
   JT — tiny shared toolkit for the tutorial widgets.
   No dependencies. Every widget file registers itself with
   JT.widget('name', (container) => {...}) and main.js mounts it on
   the element with [data-widget="name"].
   ------------------------------------------------------------------ */
(function () {
  'use strict';

  const JT = {};
  const registry = new Map();

  JT.colors = {
    jax: '#4a3aa7', jaxDeep: '#34277f', jaxSoft: '#ece9fb', jaxMid: '#b3a9ea',
    torch: '#eb6834', torchDeep: '#b8461a', torchSoft: '#fdeae1', torchMid: '#f6b79c',
    live: '#1baf7a', liveDeep: '#0e7a55', liveSoft: '#dcf5ea',
    err: '#d03b3b', errSoft: '#fbe5e5',
    ink: '#161613', ink2: '#4b4a45', ink3: '#7a7872', line: '#e7e5de', line2: '#d2cfc5',
    surface: '#ffffff', surface2: '#f4f2ec', bg: '#fbfaf7',
  };

  JT.reducedMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  JT.$ = (sel, root) => (root || document).querySelector(sel);
  JT.$$ = (sel, root) => Array.from((root || document).querySelectorAll(sel));
  JT.wait = (ms) => new Promise((r) => setTimeout(r, JT.reducedMotion ? Math.min(ms, 30) : ms));
  JT.clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  JT.lerp = (a, b, t) => a + (b - a) * t;

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
      if (c === null || c === undefined || c === false) continue;
      node.appendChild(typeof c === 'string' || typeof c === 'number' ? document.createTextNode(String(c)) : c);
    }
  }
  /** JT.el('div', {class:'x', text:'hi', onClick}, [children]) */
  JT.el = (tag, attrs, children) => { const n = document.createElement(tag); applyAttrs(n, attrs, false); appendChildren(n, children); return n; };
  /** JT.svg('rect', {x:0, y:0, width:10, height:10, fill:'#000'}) */
  JT.svg = (tag, attrs, children) => { const n = document.createElementNS('http://www.w3.org/2000/svg', tag); applyAttrs(n, attrs, true); appendChildren(n, children); return n; };

  /** Inject a scoped stylesheet once. */
  JT.style = (id, css) => {
    if (document.getElementById('style-' + id)) return;
    document.head.appendChild(JT.el('style', { id: 'style-' + id, text: css }));
  };

  /* ---------------------------------------------------------------- */
  /* Syntax highlighting (python + jaxpr), line-wrapped                 */
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
        if (fn) return PY_KW.test(fn) ? `<span class="tok-kw">${fn}</span>` : `<span class="tok-fn">${fn}</span>`;
        if (word) {
          if (PY_KW.test(word)) return `<span class="tok-kw">${word}</span>`;
          if (/^(jax|jnp|lax|optax|flax)$/.test(word)) return `<span class="tok-j">${word}</span>`;
          if (/^(torch|nn|F)$/.test(word)) return `<span class="tok-t">${word}</span>`;
          return word;
        }
        return JT.escape(m);
      });
      return `<span class="ln" data-line="${i + 1}">${out || ' '}</span>`;
    }).join('');
  }
  const JAXPR_RE = /(\{ lambda|\blambda\b|\blet\b|\bin\b)|(\b[a-z]\d*:(?:f32|i32|bool|bf16)\[[\d, ]*\])|(\b(?:f32|i32|bool|bf16)\[[\d, ]*\])|(\b(?:mul|add|sub|div|sin|cos|tanh|exp|log|dot_general|dot|neg|integer_pow|reduce_sum|reduce_max|cond|scan|select_n|gt|lt|convert_element_type|broadcast_in_dim|transpose|pjit|max|logistic|square|sqrt|psum|all_gather|reduce_scatter|all_to_all|custom_jvp_call|jvp|transpose)\b)|(\b\d+(?:\.\d+)?\b)/g;
  function highlightJaxpr(src) {
    return src.split('\n').map((line, i) => {
      const out = JT.escape(line).replace(JAXPR_RE, (m, kw, typed, ty, prim, num) => {
        if (kw) return `<span class="tok-kw">${kw}</span>`;
        if (typed) { const [v, t] = typed.split(':'); return `${v}:<span class="tok-num">${t}</span>`; }
        if (ty) return `<span class="tok-num">${ty}</span>`;
        if (prim) return `<span class="tok-fn">${prim}</span>`;
        if (num) return `<span class="tok-num">${num}</span>`;
        return m;
      });
      return `<span class="ln" data-line="${i + 1}">${out || ' '}</span>`;
    }).join('');
  }
  JT.highlight = (src, lang) => (lang === 'jaxpr' ? highlightJaxpr(src) : lang === 'plain' ? src.split('\n').map((l, i) => `<span class="ln" data-line="${i + 1}">${JT.escape(l) || ' '}</span>`).join('') : highlightPython(src));

  /** Build a <pre class="code"> element. Returns the element; lines are .ln[data-line]. */
  JT.code = (src, opts) => {
    const o = Object.assign({ lang: 'python', class: '' }, opts || {});
    return JT.el('pre', { class: 'code ' + o.class, html: JT.highlight(src.replace(/^\n+|\n+$/g, ''), o.lang) });
  };
  /** Toggle a class on a set of line numbers in a code block. */
  JT.markLines = (pre, lines, cls, on) => {
    JT.$$('.ln', pre).forEach((ln) => {
      const n = Number(ln.dataset.line);
      if (lines === '*' || lines.includes(n)) ln.classList.toggle(cls, on !== false);
    });
  };
  JT.clearLines = (pre, cls) => JT.$$('.ln.' + cls, pre).forEach((ln) => ln.classList.remove(cls));

  /* ---------------------------------------------------------------- */
  /* Tooltip                                                            */
  /* ---------------------------------------------------------------- */
  let tipEl = null;
  function ensureTip() { if (!tipEl) { tipEl = JT.el('div', { class: 'tip', role: 'tooltip' }); document.body.appendChild(tipEl); } return tipEl; }
  JT.tip = {
    show(html, x, y) {
      const t = ensureTip(); t.innerHTML = html; t.classList.add('show');
      const r = t.getBoundingClientRect();
      let left = x + 14, top = y + 14;
      if (left + r.width > window.innerWidth - 8) left = x - r.width - 14;
      if (top + r.height > window.innerHeight - 8) top = y - r.height - 14;
      t.style.left = Math.max(8, left) + 'px'; t.style.top = Math.max(8, top) + 'px';
    },
    hide() { if (tipEl) tipEl.classList.remove('show'); },
  };
  /** Any descendant with [data-tip] gets a hover/focus tooltip. */
  JT.bindTips = (root) => {
    root.addEventListener('pointermove', (e) => {
      const t = e.target.closest && e.target.closest('[data-tip]');
      if (t && root.contains(t)) JT.tip.show(t.dataset.tip, e.clientX, e.clientY); else JT.tip.hide();
    });
    root.addEventListener('pointerleave', () => JT.tip.hide());
    root.addEventListener('focusin', (e) => {
      const t = e.target.closest && e.target.closest('[data-tip]');
      if (t) { const r = t.getBoundingClientRect(); JT.tip.show(t.dataset.tip, r.left + r.width / 2, r.bottom); }
    });
    root.addEventListener('focusout', () => JT.tip.hide());
  };

  /* ---------------------------------------------------------------- */
  /* Visibility + controls                                              */
  /* ---------------------------------------------------------------- */
  JT.onVisible = (el, cb, opts) => {
    if (!('IntersectionObserver' in window)) { cb(); return; }
    const io = new IntersectionObserver((entries) => { if (entries.some((e) => e.isIntersecting)) { io.disconnect(); cb(); } }, Object.assign({ threshold: 0.25 }, opts || {}));
    io.observe(el);
  };

  const ICONS = {
    play: '<svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><path d="M4 2.5v11l9-5.5z"/></svg>',
    pause: '<svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><path d="M4 2.5h3v11H4zM9 2.5h3v11H9z"/></svg>',
    next: '<svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><path d="M3 2.5v11l7-5.5z"/><path d="M11.5 2.5h1.5v11h-1.5z"/></svg>',
    prev: '<svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><path d="M13 2.5v11l-7-5.5z"/><path d="M3 2.5h1.5v11H3z"/></svg>',
    reset: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" aria-hidden="true"><path d="M3.5 8a4.5 4.5 0 1 0 1.3-3.2"/><path d="M3.2 2.6v2.9h2.9"/></svg>',
  };
  JT.icon = (name) => ICONS[name] || '';

  /**
   * Step controller. opts: { total, onStep(i, dir), onReset(), interval }
   * Returns { el, next, prev, play, pause, reset, goto, get i, get playing }
   */
  JT.stepper = (opts) => {
    const o = Object.assign({ total: 1, interval: 900, loop: false }, opts);
    let i = -1, timer = null, playing = false;
    const readout = JT.el('span', { class: 'readout' });
    const btnPlay = JT.el('button', { class: 'btn primary', type: 'button', html: ICONS.play + '<span>Play</span>', 'aria-label': 'Play' });
    const btnPrev = JT.el('button', { class: 'btn', type: 'button', html: ICONS.prev, 'aria-label': 'Previous step', title: 'Previous step' });
    const btnNext = JT.el('button', { class: 'btn', type: 'button', html: ICONS.next, 'aria-label': 'Next step', title: 'Next step' });
    const btnReset = JT.el('button', { class: 'btn', type: 'button', html: ICONS.reset + '<span>Reset</span>', 'aria-label': 'Reset' });
    const el = JT.el('div', { class: 'controls stepper' }, [btnReset, btnPrev, btnPlay, btnNext, readout]);

    function render() {
      readout.innerHTML = i < 0 ? `step <b>0</b> / ${o.total}` : `step <b>${i + 1}</b> / ${o.total}`;
      btnPrev.disabled = i < 0; btnNext.disabled = i >= o.total - 1;
      btnPlay.innerHTML = (playing ? ICONS.pause : ICONS.play) + `<span>${playing ? 'Pause' : (i >= o.total - 1 ? 'Replay' : 'Play')}</span>`;
    }
    const api = {
      el,
      get i() { return i; },
      get playing() { return playing; },
      get total() { return o.total; },
      set total(n) { o.total = n; render(); },
      goto(n, dir) { i = JT.clamp(n, -1, o.total - 1); if (i >= 0) o.onStep(i, dir || 1); else if (o.onReset) o.onReset(); render(); return api; },
      next() { if (i < o.total - 1) api.goto(i + 1, 1); return api; },
      prev() { if (i >= 0) api.goto(i - 1, -1); return api; },
      reset() { api.pause(); i = -1; if (o.onReset) o.onReset(); render(); return api; },
      play() {
        if (i >= o.total - 1) { i = -1; if (o.onReset) o.onReset(); }
        playing = true; render();
        const tick = () => {
          if (!playing) return;
          if (i >= o.total - 1) { if (o.loop) { i = -1; if (o.onReset) o.onReset(); } else { api.pause(); return; } }
          api.next();
          timer = setTimeout(tick, JT.reducedMotion ? 60 : o.interval);
        };
        tick();
        return api;
      },
      pause() { playing = false; if (timer) clearTimeout(timer); timer = null; render(); return api; },
      toggle() { return playing ? api.pause() : api.play(); },
    };
    btnPlay.addEventListener('click', () => api.toggle());
    btnPrev.addEventListener('click', () => { api.pause(); api.prev(); });
    btnNext.addEventListener('click', () => { api.pause(); api.next(); });
    btnReset.addEventListener('click', () => api.reset());
    render();
    return api;
  };

  /** Segmented control. options: [{value, label, cls}] */
  JT.seg = (options, onChange, initial, cls) => {
    const el = JT.el('div', { class: 'seg ' + (cls || ''), role: 'group' });
    let value = initial === undefined ? options[0].value : initial;
    const buttons = options.map((opt) => {
      const b = JT.el('button', { type: 'button', text: opt.label, 'aria-pressed': String(opt.value === value), dataset: { value: opt.value } });
      b.addEventListener('click', () => { if (value === opt.value) return; value = opt.value; buttons.forEach((x) => x.setAttribute('aria-pressed', String(x.dataset.value === String(value)))); onChange(value); });
      return b;
    });
    buttons.forEach((b) => el.appendChild(b));
    el.setValue = (v) => { value = v; buttons.forEach((x) => x.setAttribute('aria-pressed', String(x.dataset.value === String(v)))); };
    Object.defineProperty(el, 'value', { get: () => value });
    return el;
  };

  /** Stage scaffold: returns {root, head, body, foot, controls} */
  JT.stage = (container, opts) => {
    const o = Object.assign({ title: '', hint: '', foot: true }, opts || {});
    container.classList.add('stage');
    const controls = JT.el('div', { class: 'controls' });
    const head = JT.el('div', { class: 'stage-head' }, [
      JT.el('h4', { text: o.title }), o.hint ? JT.el('span', { class: 'hint', text: o.hint }) : null, controls,
    ]);
    const body = JT.el('div', { class: 'stage-body' + (o.tight ? ' tight' : '') });
    const foot = o.foot ? JT.el('div', { class: 'stage-foot' }) : null;
    container.replaceChildren(head, body, foot || '');
    return { root: container, head, body, foot, controls };
  };

  JT.fmt = (n, d) => (typeof n === 'number' ? n.toLocaleString(undefined, { maximumFractionDigits: d === undefined ? 0 : d }) : n);

  /* ---------------------------------------------------------------- */
  /* Registry                                                           */
  /* ---------------------------------------------------------------- */
  JT.widget = (name, init) => { registry.set(name, init); };
  JT.mountAll = () => {
    JT.$$('[data-widget]').forEach((el) => {
      const init = registry.get(el.dataset.widget);
      if (!init) { console.warn('No widget registered for', el.dataset.widget); return; }
      try { init(el); } catch (e) { console.error('Widget failed:', el.dataset.widget, e); }
    });
  };

  window.JT = JT;
})();
