/* Page glue: highlight static code, mount figures, mark the current section, reading progress. */
(function () {
  'use strict';
  const JT = window.JT;

  function highlightStatic() {
    JT.$$('pre[data-lang]').forEach((pre) => {
      const src = pre.textContent;
      pre.classList.add('code');
      pre.innerHTML = JT.highlight(src.replace(/^\n+|\n+$/g, ''), pre.dataset.lang);
    });
  }

  function scrollSpy() {
    const links = JT.$$('.topbar nav a');
    const sections = links.map((a) => document.getElementById(a.getAttribute('href').slice(1))).filter(Boolean);
    const bar = JT.$('.topbar .progress');
    const nav = JT.$('.topbar nav');
    let ticking = false, current = null;
    function update() {
      ticking = false;
      const y = window.scrollY + window.innerHeight * 0.3;
      let active = null;
      for (const s of sections) if (s.offsetTop <= y) active = s;
      const max = document.documentElement.scrollHeight - window.innerHeight;
      if (bar) bar.style.width = (max > 0 ? Math.min(100, (100 * window.scrollY) / max) : 0) + '%';
      if (active === current) return;
      current = active;
      links.forEach((a) => {
        if (active && a.getAttribute('href') === '#' + active.id) a.setAttribute('aria-current', 'true');
        else a.removeAttribute('aria-current');
      });
      const link = links.find((a) => a.hasAttribute('aria-current'));
      if (link && nav) {
        const r = link.getBoundingClientRect(), nr = nav.getBoundingClientRect();
        if (r.left < nr.left || r.right > nr.right) nav.scrollLeft = link.offsetLeft - nav.offsetLeft - 16;
      }
    }
    window.addEventListener('scroll', () => { if (!ticking) { ticking = true; requestAnimationFrame(update); } }, { passive: true });
    window.addEventListener('resize', update);
    update();
  }

  document.addEventListener('DOMContentLoaded', () => {
    highlightStatic();
    JT.mountAll();
    scrollSpy();
  });
})();
