/* Page glue: mount widgets, scroll spy, reading progress. */
(function () {
  'use strict';
  const JT = window.JT;

  function scrollSpy() {
    const links = JT.$$('.topbar nav a');
    const sections = links.map((a) => document.getElementById(a.getAttribute('href').slice(1))).filter(Boolean);
    const bar = JT.$('.topbar .progress');
    let ticking = false;
    function update() {
      ticking = false;
      const y = window.scrollY + 120;
      let active = null;
      for (const s of sections) if (s.offsetTop <= y) active = s;
      links.forEach((a) => a.classList.toggle('active', active && a.getAttribute('href') === '#' + active.id));
      const max = document.documentElement.scrollHeight - window.innerHeight;
      if (bar) bar.style.transform = 'scaleX(' + (max > 0 ? window.scrollY / max : 0) + ')';
      if (!active && links.length) links[0].parentElement.scrollTo({ left: 0, behavior: 'smooth' });
      if (active) {
        const link = links.find((a) => a.classList.contains('active'));
        if (link && link.scrollIntoView) {
          const nav = link.parentElement; const r = link.getBoundingClientRect(); const nr = nav.getBoundingClientRect();
          if (r.left < nr.left || r.right > nr.right) nav.scrollTo({ left: link.offsetLeft - 24, behavior: 'smooth' });
        }
      }
    }
    window.addEventListener('scroll', () => { if (!ticking) { ticking = true; requestAnimationFrame(update); } }, { passive: true });
    update();
  }

  document.addEventListener('DOMContentLoaded', () => {
    JT.mountAll();
    scrollSpy();
  });
})();
