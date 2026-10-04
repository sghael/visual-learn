(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const { residual, expectedTokens, speedup } = window.SpeculativeMath;
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');

  // One owner for the timer. Every state-changing control cancels it first.
  const cases = {
    third: { draft: ['along', 'the', 'lake', '.'], target: ['along', 'the', 'river', '.', 'Then'], keep: 2 },
    all: { draft: ['along', 'the', 'river', '.'], target: ['along', 'the', 'river', '.', 'Then'], keep: 4 },
    first: { draft: ['beside', 'a', 'lake', '.'], target: ['along', 'the', 'river', '.', 'Then'], keep: 0 },
  };
  let roundStep = 6, timer = null;
  function stopRound() { if (timer !== null) clearTimeout(timer); timer = null; }
  function renderRound() {
    const example = cases[$('round-case').value];
    const verified = roundStep >= 5, committed = roundStep === 6;
    const cell = (text, className) => `<div class="${className}">${text}</div>`;
    let grid = cell('', 'position') + [1, 2, 3, 4, 'Bonus'].map(n => cell(typeof n === 'number' ? `Token ${n}` : n, 'position')).join('');
    grid += cell('Draft proposes', 'row-label draft-text');
    grid += example.draft.map((token, i) => cell(i < roundStep ? token : '—', `token ${i < roundStep ? 'draft' : 'empty'}${committed && i >= example.keep ? ' discarded' : ''}`)).join('') + cell('—', 'verdict');
    grid += cell('Target predicts', 'row-label target-text');
    grid += example.target.map((token, i) => cell(verified ? token : '—', `token ${verified ? 'target' : 'empty'}${committed && i > example.keep ? ' discarded' : ''}`)).join('');
    grid += cell('Decision', 'row-label');
    grid += example.target.map((_, i) => {
      if (!verified) return cell('Waiting', 'verdict');
      if (i < example.keep) return cell('Accept', 'verdict accept');
      if (i === example.keep) return cell(i === 4 ? 'Bonus token' : 'Reject → replace', `verdict ${i === 4 ? 'bonus' : 'reject'}`);
      return cell('Discard suffix', 'verdict');
    }).join('');
    $('round-grid').innerHTML = grid;
    $('round-step').textContent = `Step ${roundStep} of 6`;
    $('round-play').textContent = timer !== null ? 'Pause' : roundStep === 6 ? 'Replay' : 'Play';
    $('round-next').disabled = roundStep === 6;
    const descriptions = [
      'Ready. The committed prefix is “We walked.” No draft tokens have been proposed yet.',
      'Draft step 1: propose a token after “We walked.”',
      'Draft step 2: use the first proposal as context for the next one.',
      'Draft step 3: extend the same candidate continuation.',
      'Draft step 4: the block is ready. The draft has made four sequential calls.',
      'One target pass scores all five positions. Read the decisions from left to right.',
      example.keep === 4 ? 'All four proposals match. Keep them and append “Then,” the target’s bonus token.' : example.keep === 0 ? 'The first proposal fails. Replace it with “along” and discard all later proposals.' : 'Keep “along the,” replace “lake” with “river,” and discard the suffix. The later period matched, but its prefix was wrong.',
    ];
    $('round-note').textContent = descriptions[roundStep];
    $('round-output').innerHTML = 'We walked' + (committed ? ` <span class="output-accepted">${example.draft.slice(0, example.keep).join(' ')}</span> <span class="output-target">${example.target[example.keep]}</span>` : '');
    $('round-committed').textContent = committed ? example.keep + 1 : 0;
  }
  function tickRound() {
    roundStep = Math.min(6, roundStep + 1);
    timer = roundStep < 6 ? setTimeout(tickRound, 900) : null;
    renderRound();
  }
  $('round-play').addEventListener('click', () => {
    if (timer !== null) { stopRound(); renderRound(); return; }
    if (reducedMotion.matches) { roundStep = 6; renderRound(); return; }
    if (roundStep === 6) roundStep = 0;
    tickRound();
  });
  $('round-next').addEventListener('click', () => { stopRound(); roundStep = Math.min(6, roundStep + 1); renderRound(); });
  $('round-reset').addEventListener('click', () => { stopRound(); roundStep = 0; renderRound(); });
  $('round-case').addEventListener('change', () => { stopRound(); roundStep = 6; renderRound(); });
  reducedMotion.addEventListener('change', () => { stopRound(); renderRound(); });
  renderRound();

  // Show exact probability mass, avoiding Monte Carlo noise in the lesson.
  const p = [0.2, 0.5, 0.3], words = ['lake', 'river', 'sea'];
  let exactDraft = false;
  const percent = value => `${(value * 100).toFixed(1).replace(/\.0$/, '')}%`;
  function renderProbability() {
    const lake = Number($('q-lake').value) / 100;
    const q = exactDraft ? [...p] : [lake, (1 - lake) * 0.6, (1 - lake) * 0.4];
    const model = residual(p, q);
    $('q-value').textContent = percent(q[0]);
    $('prob-overlap').textContent = percent(1 - model.rejection);
    $('prob-message').textContent = model.correction ? `${percent(model.rejection)} rejection; the correction restores the missing mass.` : 'No rejection. The correction is unused because the distributions match.';
    $('prob-exact').setAttribute('aria-pressed', String(exactDraft));
    $('prob-default').setAttribute('aria-pressed', String(!exactDraft && lake === 0.5));
    $('prob-bars').innerHTML = [
      { title: 'Draft q', description: 'What gets proposed', values: q, kind: 'draft' },
      { title: 'Target p', description: 'What we want to sample', values: p, kind: 'target' },
      { title: 'Final output', description: 'Accepted mass + correction mass', values: model.output, kind: 'output' },
    ].map(panel => `<div class="distribution-panel"><h3>${panel.title}</h3><p>${panel.description}</p><div class="prob-rows">${panel.values.map((value, i) => `<div class="prob-bar-row"><div class="bar-heading"><span>${words[i]}</span><span>${percent(value)}</span></div><div class="mass-track" aria-hidden="true">${panel.kind === 'output' ? `<span class="mass accepted" style="width:${model.accepted[i] * 100}%"></span><span class="mass target" style="width:${Math.max(0, value - model.accepted[i]) * 100}%"></span>` : `<span class="mass ${panel.kind}" style="width:${value * 100}%"></span>`}</div></div>`).join('')}</div></div>`).join('');
    $('prob-table').querySelector('tbody').innerHTML = words.map((word, i) => `<tr data-token="${word}"><th scope="row">${word}</th><td class="n" data-col="target">${percent(p[i])}</td><td class="n" data-col="draft">${percent(q[i])}</td><td class="n" data-col="accepted">${percent(model.accepted[i])}</td><td class="n" data-col="correction">${model.correction ? percent(model.correction[i]) : '—'}</td><td class="n" data-col="output">${percent(model.output[i])}</td></tr>`).join('');
  }
  $('q-lake').addEventListener('input', () => { exactDraft = false; renderProbability(); });
  $('prob-exact').addEventListener('click', () => { exactDraft = true; $('q-lake').value = 20; renderProbability(); });
  $('prob-default').addEventListener('click', () => { exactDraft = false; $('q-lake').value = 50; renderProbability(); });
  renderProbability();

  const speedInputs = ['acceptance', 'draft-length', 'draft-cost', 'verify-cost'];
  const presets = { balanced: [80, 4, 10, 100], poor: [20, 4, 10, 100], expensive: [80, 4, 50, 150] };
  function renderSpeed() {
    const values = speedInputs.map(id => Number($(id).value));
    const [a, k, c, v] = [values[0] / 100, values[1], values[2] / 100, values[3] / 100];
    $('acceptance-value').textContent = percent(a);
    $('length-value').textContent = k;
    $('draft-cost-value').textContent = c.toFixed(2);
    $('verify-cost-value').textContent = v.toFixed(2);
    $('tokens-value').textContent = expectedTokens(a, k).toFixed(2);
    $('cost-value').textContent = (v + k * c).toFixed(2);
    const speed = speedup(a, k, c, v);
    $('speed-value').textContent = `${speed.toFixed(2)}×`;
    document.querySelectorAll('[data-speed-preset]').forEach(button => button.setAttribute('aria-pressed', String(presets[button.dataset.speedPreset].every((n, i) => n === values[i]))));
    const curve = Array.from({ length: 12 }, (_, i) => speedup(a, i + 1, c, v));
    const best = curve.indexOf(Math.max(...curve)) + 1;
    const comparison = Math.abs(speed - 1) < 1e-10 ? 'Same modeled latency as ordinary decoding.' : `${speed > 1 ? 'Faster' : 'Slower'} than ordinary decoding in this model.`;
    $('speed-note').textContent = `${comparison} Best draft length among 1–12: ${best} (${curve[best - 1].toFixed(2)}×).`;
    const maxY = Math.max(2, Math.ceil(Math.max(...curve))), left = 60, right = 825, top = 32, bottom = 254;
    const x = length => left + (length - 1) * (right - left) / 11;
    const y = value => bottom - value / maxY * (bottom - top);
    let svg = '<title>Simplified speedup versus draft length</title><desc>The dashed line marks ordinary decoding at 1×. The blue curve shows the model; the outlined point is the selected draft length.</desc>';
    for (let i = 0; i <= 4; i++) {
      const value = maxY * i / 4;
      svg += `<line x1="${left}" x2="${right}" y1="${y(value)}" y2="${y(value)}" stroke="var(--rule)"/><text x="${left - 12}" y="${y(value) + 5}" text-anchor="end">${value.toFixed(1)}×</text>`;
    }
    svg += `<line x1="${left}" x2="${right}" y1="${y(1)}" y2="${y(1)}" stroke="var(--ink-2)" stroke-dasharray="5 5"/><text class="halo" x="${right}" y="${y(1) - 7}" text-anchor="end">ordinary decoding · 1×</text>`;
    svg += `<path class="curve" d="${curve.map((value, i) => `${i ? 'L' : 'M'}${x(i + 1)},${y(value)}`).join(' ')}"/>`;
    curve.forEach((value, i) => {
      svg += `<circle cx="${x(i + 1)}" cy="${y(value)}" r="${i + 1 === k ? 6 : 3}" fill="${i + 1 === k ? 'var(--paper)' : 'var(--target)'}" stroke="var(--target)" stroke-width="${i + 1 === k ? 2 : 1}"/><text x="${x(i + 1)}" y="276" text-anchor="middle">${i + 1}</text>`;
    });
    svg += `<text x="${left}" y="18">Modeled speedup</text><text x="${right}" y="302" text-anchor="end">Draft length k (tokens)</text>`;
    $('speed-chart').innerHTML = svg;
    $('speed-chart').setAttribute('aria-label', `Simplified speedup versus draft length. Selected ${k} tokens: ${speed.toFixed(2)} times. Best among 1 through 12: ${best} tokens, ${curve[best - 1].toFixed(2)} times.`);
  }
  speedInputs.forEach(id => $(id).addEventListener('input', renderSpeed));
  document.querySelectorAll('[data-speed-preset]').forEach(button => button.addEventListener('click', () => {
    presets[button.dataset.speedPreset].forEach((value, i) => { $(speedInputs[i]).value = value; }); renderSpeed();
  }));
  renderSpeed();

  const sections = Array.from(document.querySelectorAll('main section'));
  const navLinks = Array.from(document.querySelectorAll('.topbar nav a'));
  function updateReading() {
    let active = sections[0].id;
    for (const section of sections) if (section.getBoundingClientRect().top <= 140) active = section.id;
    navLinks.forEach(link => link.setAttribute('aria-current', String(link.hash === `#${active}`)));
    const range = document.documentElement.scrollHeight - window.innerHeight;
    document.querySelector('.progress').style.width = `${range > 0 ? Math.min(100, window.scrollY / range * 100) : 0}%`;
  }
  window.addEventListener('scroll', updateReading, { passive: true });
  window.addEventListener('resize', updateReading);
  updateReading();
})();
