/* ------------------------------------------------------------------
   Two hardware figures.
   'systolic': a 4x4 weight-stationary systolic array (a simplified TPU MXU)
               computing C = A·B one clock cycle per step. Rests on cycle 4,
               where the diagonal wavefront is visible.
   'gpu':      a static drawing of the same 4x4 output as 16 thread blocks
               assigned to 8 streaming multiprocessors in two waves.
   ------------------------------------------------------------------ */
(function () {
  'use strict';
  const JT = window.JT;
  if (!JT) return;

  /* ---------------------------------------------------------------- */
  /* The matmul and the systolic model                                  */
  /* ---------------------------------------------------------------- */
  const A = [[1, 2, 1, 3], [2, 1, 3, 1], [3, 1, 2, 2], [1, 3, 2, 1]];
  const B = [[2, 1, 3, 1], [1, 3, 1, 2], [3, 2, 1, 1], [1, 1, 2, 3]];
  const N = 4;
  const CYCLES = 3 * N - 2;            // M + K + N − 2 for a skewed weight-stationary array
  const TOTAL_MACS = N * N * N;        // 64
  const C = A.map((row) => B[0].map((_, n) => row.reduce((s, a, k) => s + a * B[k][n], 0)));

  // Weight-stationary dataflow. Cell (k, n) holds B[k][n]. Activation A[m][k] enters row k
  // from the left on cycle m + k and shifts one cell right per cycle. The partial sum for
  // C[m][n] starts at row 0 of column n and shifts one cell down per cycle, so at cycle t
  // cell (k, n) is working on output row m = t − k − n.
  const cellRow = (t, k, n) => t - k - n;
  const cellActive = (t, k, n) => { const m = cellRow(t, k, n); return m >= 0 && m < N; };
  const macsAt = (t) => { let c = 0; for (let k = 0; k < N; k++) for (let n = 0; n < N; n++) if (cellActive(t, k, n)) c++; return c; };
  const partial = (m, n, k) => { let s = 0; for (let j = 0; j <= k; j++) s += A[m][j] * B[j][n]; return s; };

  // Self-check: the cycle-by-cycle model must reproduce A·B in exactly CYCLES cycles with N³ MACs.
  (function selfCheck() {
    const sim = A.map(() => B[0].map(() => 0));
    let macs = 0;
    for (let t = 0; t < CYCLES; t++) for (let k = 0; k < N; k++) for (let n = 0; n < N; n++) {
      if (!cellActive(t, k, n)) continue;
      const m = cellRow(t, k, n);
      sim[m][n] += A[m][k] * B[k][n]; macs++;
    }
    const same = sim.every((r, m) => r.every((v, n) => v === C[m][n]));
    const lastExit = (N - 1) + (N - 1) + (N - 1);
    if (!same || macs !== TOTAL_MACS || lastExit !== CYCLES - 1) {
      throw new Error(`silicon: systolic model failed self-check (match=${same}, macs=${macs}, lastExit=${lastExit}, cycles=${CYCLES})`);
    }
  })();

  const S = JT.svg;
  const text = (x, y, str, attrs) => S('text', Object.assign({ x, y, text: String(str) }, attrs || {}));
  const centered = (x, y, str, attrs) => text(x, y, str, Object.assign({ 'text-anchor': 'middle', 'dominant-baseline': 'central' }, attrs || {}));
  const place = (g, x, y, visible) => { g.style.transform = `translate(${x}px, ${y}px)`; g.style.opacity = visible ? '1' : '0'; };
  const cls = (el, name, on) => el.classList.toggle(name, !!on);

  /* ---------------------------------------------------------------- */
  /* Systolic array                                                     */
  /* ---------------------------------------------------------------- */
  JT.widget('systolic', (container) => {
    container.classList.add('silicon', 'systolic');
    if (JT.reducedMotion) container.classList.add('no-motion');
    const CELL = 44, GX = 150, GY = 40, SLOT = 19, CH = 24, PW = 18, PH = 16, QW = 26;
    const CY = GY + N * CELL + 22;
    const W = GX + N * CELL + 10, H = CY + N * CH + 4;
    const colX = (n) => GX + n * CELL + CELL / 2;
    const svg = S('svg', { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'A 4 by 4 weight-stationary systolic array computing C = A times B, one clock cycle per step.' });

    svg.appendChild(text(0, 24, 'A, skewed', { class: 'lbl' }));
    svg.appendChild(text(GX, 24, 'B, held in the cells', { class: 'lbl' }));
    svg.appendChild(text(GX - 10, CY + CH / 2, 'C = A·B', { class: 'lbl strong', 'text-anchor': 'end', 'dominant-baseline': 'central' }));
    svg.appendChild(text(GX - 10, CY + CH / 2 + 17, 'sums leave', { class: 'lbl', 'text-anchor': 'end', 'dominant-baseline': 'central' }));
    svg.appendChild(text(GX - 10, CY + CH / 2 + 32, 'at the bottom', { class: 'lbl', 'text-anchor': 'end', 'dominant-baseline': 'central' }));

    for (let k = 0; k < N; k++) {
      const y = GY + k * CELL + CELL / 2;
      svg.appendChild(S('line', { class: 'wire', x1: 0, y1: y, x2: GX - 1, y2: y }));
      svg.appendChild(S('polygon', { class: 'wirehead', points: `${GX},${y} ${GX - 5},${y - 2.8} ${GX - 5},${y + 2.8}` }));
    }
    for (let n = 0; n < N; n++) {
      const x = colX(n);
      svg.appendChild(S('line', { class: 'wire', x1: x, y1: GY + N * CELL, x2: x, y2: CY - 1 }));
      svg.appendChild(S('polygon', { class: 'wirehead', points: `${x},${CY} ${x - 2.8},${CY - 5} ${x + 2.8},${CY - 5}` }));
    }
    const cells = [];
    for (let k = 0; k < N; k++) for (let n = 0; n < N; n++) {
      const x = GX + n * CELL, y = GY + k * CELL;
      const g = S('g', { class: 'cell' }, [S('rect', { x, y, width: CELL, height: CELL }), text(x + CELL - 4, y + 13, B[k][n], { class: 'w', 'text-anchor': 'end' })]);
      cells.push(g); svg.appendChild(g);
    }
    const ccells = [];
    for (let m = 0; m < N; m++) for (let n = 0; n < N; n++) {
      const x = GX + n * CELL, y = CY + m * CH;
      const g = S('g', { class: 'ccell' }, [S('rect', { x, y, width: CELL, height: CH }), centered(x + CELL / 2, y + CH / 2, C[m][n])]);
      ccells.push(g); svg.appendChild(g);
    }
    const psums = [];
    for (let m = 0; m < N; m++) for (let n = 0; n < N; n++) {
      const g = S('g', { class: 'ps' }, [S('rect', { x: 0, y: 0, width: QW, height: PH, rx: 2 }), centered(QW / 2, PH / 2 + 0.5, '')]);
      psums.push(g); svg.appendChild(g);
    }
    const toks = [];
    for (let m = 0; m < N; m++) for (let k = 0; k < N; k++) {
      const g = S('g', { class: 'tok' }, [S('rect', { x: 0, y: 0, width: PW, height: PH, rx: 2 }), centered(PW / 2, PH / 2 + 0.5, A[m][k])]);
      toks.push(g); svg.appendChild(g);
    }

    const note = JT.el('p', { class: 'note', 'aria-live': 'polite' });
    const stepper = JT.stepper({ total: CYCLES, interval: 800, onStep: render, onReset: () => render(-1), noun: 'cycle' });
    container.append(JT.el('div', { class: 'controls' }, [stepper.el]), svg, note);

    function render(t) {
      let macs = 0;
      cells.forEach((g, i) => { const k = (i / N) | 0, n = i % N; const on = t >= 0 && cellActive(t, k, n); cls(g, 'on', on); if (on) macs++; });
      ccells.forEach((g, i) => { const m = (i / N) | 0, n = i % N; cls(g, 'done', t >= m + n + N - 1); });
      toks.forEach((g, i) => {
        const m = (i / N) | 0, k = i % N, p = t - m - k;
        const y = GY + k * CELL + CELL / 2 - PH / 2;
        if (p < 0) { place(g, GX - 8 - (-p - 0.5) * SLOT - PW / 2, y, true); cls(g, 'in', false); }
        else if (p < N) { place(g, GX + p * CELL + 3, y, true); cls(g, 'in', true); }
        else { place(g, GX + N * CELL + 8, y, false); cls(g, 'in', true); }
      });
      psums.forEach((g, i) => {
        const m = (i / N) | 0, n = i % N, k = t - m - n;
        const x = colX(n) - QW / 2 - 6;
        if (k < 0) place(g, x, GY - PH / 2, false);
        else if (k < N) { g.lastChild.textContent = partial(m, n, k); place(g, x, GY + (k + 1) * CELL - PH / 2, true); }
        else place(g, x, CY + m * CH + CH / 2 - PH / 2, false);
      });
      let sofar = 0; for (let u = 0; u <= t; u++) sofar += macsAt(u);
      if (t < 0) { note.innerHTML = 'Before cycle 1: the 16 weights of B sit in the cells, and the rows of A wait on the left, each row delayed one cycle more than the one above.'; return; }
      const enter = [], out = [];
      for (let k = 0; k < N; k++) { const m = t - k; if (m >= 0 && m < N) enter.push(`a[${m}][${k}]`); }
      for (let n = 0; n < N; n++) { const m = t - n - (N - 1); if (m >= 0 && m < N) out.push(`c[${m}][${n}]`); }
      let s = `Cycle ${t + 1}: ` + (enter.length ? `${enter.join(', ')} enter` : 'no new inputs') + `; <b>${macs}</b> cell${macs === 1 ? '' : 's'} multiply and accumulate` + (out.length ? `; ${out.join(', ')} leave${out.length === 1 ? 's' : ''}` : '');
      s += t === CYCLES - 1 ? `. Done: ${TOTAL_MACS} multiply-accumulates in ${CYCLES} cycles.` : `. ${sofar} of ${TOTAL_MACS} multiply-accumulates done.`;
      note.innerHTML = s;
    }
    stepper.goto(3);
  });

  /* ---------------------------------------------------------------- */
  /* GPU: thread blocks on streaming multiprocessors (static)           */
  /* ---------------------------------------------------------------- */
  JT.widget('gpu', (container) => {
    container.classList.add('silicon', 'gpu');
    const BLOCKS = 16, SMS = 8, T = 30, TG = 3;
    const W = 340, H = 212;
    const svg = S('svg', { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'Sixteen thread blocks, one per output tile, run on eight streaming multiprocessors in two waves: blocks 0 to 7, then 8 to 15.' });
    svg.appendChild(text(0, 12, 'C: 16 output tiles', { class: 't b' }));
    svg.appendChild(text(0, 27, 'one thread block each', { class: 't muted' }));
    for (let i = 0; i < BLOCKS; i++) {
      const r = (i / 4) | 0, c = i % 4, x = c * (T + TG), y = 38 + r * (T + TG);
      svg.appendChild(S('rect', { class: 'tile ' + (i < SMS ? 'w1' : 'w2'), x, y, width: T, height: T }));
      svg.appendChild(centered(x + T / 2, y + T / 2 + 0.5, i, { class: 't' }));
    }
    svg.appendChild(text(0, 38 + 4 * (T + TG) + 16, 'dark: wave 1 · light: wave 2', { class: 't muted' }));
    const X0 = 158;
    svg.appendChild(text(X0, 12, '8 streaming multiprocessors', { class: 't b' }));
    svg.appendChild(text(X0, 27, 'wave 1 → wave 2', { class: 't muted' }));
    for (let i = 0; i < SMS; i++) {
      const c = i % 2, r = (i / 2) | 0, x = X0 + c * 92, y = 38 + r * 36;
      svg.appendChild(S('rect', { class: 'sm', x, y, width: 86, height: 30, rx: 2 }));
      svg.appendChild(text(x + 7, y + 19.5, 'SM' + i, { class: 't b' }));
      svg.appendChild(text(x + 80, y + 19.5, `${i} → ${i + SMS}`, { class: 't', 'text-anchor': 'end' }));
    }
    svg.appendChild(text(X0, 38 + 4 * 36 + 16, 'assignment chosen at run time', { class: 'annot' }));
    container.append(svg);
  });
})();
