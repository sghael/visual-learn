/* ------------------------------------------------------------------
   silicon.js — "One matmul, two machines"
   Left pane: a 4×4 weight-stationary systolic array (the TPU MXU)
   stepping through C = A·B one clock cycle at a time.
   Right pane: the same matmul as a GPU grid of thread blocks that a
   hardware scheduler hands to streaming multiprocessors in waves.
   ------------------------------------------------------------------ */
(function () {
  'use strict';
  const JT = window.JT;
  if (!JT) return;

  JT.style('silicon', `
#w-silicon .pane-body { padding: 0.7rem 0.9rem 0.9rem; }
#w-silicon .sil-ctl { margin-bottom: 0.65rem; }
#w-silicon .sil-ctl .stepper { margin-left: 0; }
#w-silicon .sil-ctl .stepper .readout { display: none; }
#w-silicon svg.sil { width: 100%; max-width: 480px; margin: 0 auto; overflow: visible; }
#w-silicon .sil-read { display: block; margin-top: 0.65rem; }
#w-silicon .sil-cap { font-size: 0.78rem; line-height: 1.45; color: var(--ink-3); margin-top: 0.3rem; min-height: 3.6em; text-wrap: pretty; }
#w-silicon .sil-cap b { color: var(--ink-2); font-weight: 600; }
#w-silicon .sil-take { flex: 1 1 18rem; }
#w-silicon .sil .lbl { font-size: 10.5px; fill: var(--ink-3); }
#w-silicon .sil .lbl.strong { fill: var(--ink-2); font-weight: 600; }
#w-silicon .sil .wire { stroke: var(--line-2); stroke-width: 1; }
#w-silicon .sil .wirehead { fill: var(--line-2); }
#w-silicon .sil .icon { fill: var(--surface); stroke: var(--line-2); }
#w-silicon .sil .icon-line { stroke: var(--ink-3); stroke-width: 1; }
/* TPU side */
#w-silicon .cell rect { fill: var(--surface); stroke: var(--line-2); transition: fill 180ms ease-out, stroke 180ms ease-out; }
#w-silicon .cell.on rect { fill: var(--live-soft); stroke: var(--live); }
#w-silicon .cell .w { font-size: 10px; fill: var(--ink-3); transition: fill 180ms; }
#w-silicon .cell.on .w { fill: var(--live-deep); }
#w-silicon .tok, #w-silicon .ps { transition: transform 220ms ease-out, opacity 220ms ease-out; }
#w-silicon .tok rect { fill: var(--jax-soft); stroke: var(--jax-mid); transition: fill 180ms, stroke 180ms; }
#w-silicon .tok text { font-size: 11px; font-weight: 600; fill: var(--jax-deep); transition: fill 180ms; }
#w-silicon .tok.in rect { fill: var(--jax); stroke: var(--jax-deep); }
#w-silicon .tok.in text { fill: #fff; }
#w-silicon .ps rect { fill: var(--surface); stroke: var(--live-deep); stroke-width: 1.2; }
#w-silicon .ps text { font-size: 11px; font-weight: 600; fill: var(--live-deep); }
#w-silicon .ccell rect { fill: var(--surface); stroke: var(--line-2); transition: fill 200ms, stroke 200ms; }
#w-silicon .ccell text { font-size: 11px; font-weight: 600; fill: var(--jax-deep); opacity: 0; transition: opacity 200ms; }
#w-silicon .ccell.done rect { fill: var(--jax-soft); stroke: var(--jax-mid); }
#w-silicon .ccell.done text { opacity: 1; }
/* GPU side */
#w-silicon .qb { transition: opacity 250ms; }
#w-silicon .qb rect { fill: var(--torch-soft); stroke: var(--torch-mid); }
#w-silicon .qb text { font-size: 9.5px; font-weight: 600; fill: var(--torch-deep); }
#w-silicon .qb.gone { opacity: 0.15; }
#w-silicon .qb.dim { opacity: 0.45; }
#w-silicon .hbm rect.bar { fill: var(--surface-2); stroke: var(--line-2); transition: stroke 200ms; }
#w-silicon .hbm.busy rect.bar { stroke: var(--torch); }
#w-silicon .rail { stroke: var(--line-2); stroke-width: 1; fill: none; }
#w-silicon .sm .box { fill: var(--surface); stroke: var(--line-2); transition: fill 200ms, stroke 200ms; }
#w-silicon .sm.res .box { stroke: var(--torch); }
#w-silicon .sm.on .box { fill: var(--live-soft); stroke: var(--live); }
#w-silicon .sm circle { fill: var(--line-2); transition: fill 200ms; }
#w-silicon .sm.res circle { fill: var(--torch-mid); }
#w-silicon .sm.on circle { fill: var(--live); }
#w-silicon .sm .smem { fill: var(--line); transition: fill 200ms; }
#w-silicon .sm.load .smem { fill: var(--torch); }
#w-silicon .sm .name { font-size: 10px; font-weight: 600; fill: var(--ink-2); }
#w-silicon .sm .blk { font-size: 10px; font-weight: 600; fill: var(--torch-deep); }
#w-silicon .sm .blk.idle { fill: var(--ink-3); font-weight: 400; }
#w-silicon .arr { opacity: 0; transition: opacity 200ms; }
#w-silicon .arr.show { opacity: 1; }
#w-silicon .arr line { stroke: var(--torch); stroke-width: 1.5; }
#w-silicon .arr polygon { fill: var(--torch); }
#w-silicon .tile rect { fill: var(--surface); stroke: var(--line-2); transition: fill 220ms, stroke 220ms; }
#w-silicon .tile text { font-size: 9.5px; fill: var(--ink-3); transition: fill 220ms; }
#w-silicon .tile.done rect { fill: var(--torch); stroke: var(--torch-deep); }
#w-silicon .tile.done text { fill: #fff; font-weight: 600; }
#w-silicon .wave { font-size: 10.5px; fill: var(--ink-3); transition: fill 200ms; }
#w-silicon .wave.now { fill: var(--torch-deep); font-weight: 600; }
#w-silicon .wave.done { fill: var(--ink-2); }
#w-silicon.no-motion * { transition: none !important; }
  `);

  /* ---------------------------------------------------------------- */
  /* The matmul                                                         */
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
    const lastExit = (N - 1) + (N - 1) + (N - 1); // cycle index on which C[N−1][N−1] leaves the array
    if (!same || macs !== TOTAL_MACS || lastExit !== CYCLES - 1) {
      throw new Error(`silicon: systolic model failed self-check (match=${same}, macs=${macs}, lastExit=${lastExit}, cycles=${CYCLES})`);
    }
  })();

  /* ---------------------------------------------------------------- */
  /* SVG helpers                                                        */
  /* ---------------------------------------------------------------- */
  const S = JT.svg;
  const text = (x, y, str, attrs) => S('text', Object.assign({ x, y, text: String(str) }, attrs || {}));
  const centered = (x, y, str, attrs) => text(x, y, str, Object.assign({ 'text-anchor': 'middle', 'dominant-baseline': 'central' }, attrs || {}));
  function arrow(x1, y1, x2, y2, attrs) {
    const dx = x2 - x1, dy = y2 - y1, len = Math.hypot(dx, dy) || 1, ux = dx / len, uy = dy / len;
    const hl = 5, hw = 2.8, bx = x2 - ux * hl, by = y2 - uy * hl;
    return S('g', attrs, [
      S('line', { x1, y1, x2: bx, y2: by }),
      S('polygon', { points: `${x2},${y2} ${bx - uy * hw},${by + ux * hw} ${bx + uy * hw},${by - ux * hw}` }),
    ]);
  }
  const place = (g, x, y, visible) => { g.style.transform = `translate(${x}px, ${y}px)`; g.style.opacity = visible ? '1' : '0'; };
  const cls = (el, name, on) => el.classList.toggle(name, !!on);

  /* ---------------------------------------------------------------- */
  /* Left pane: TPU systolic array                                      */
  /* ---------------------------------------------------------------- */
  function buildTPU() {
    const CELL = 44, GX = 140, GY = 36, SLOT = 18, CH = 24, PW = 16, PH = 14, QW = 22;
    const CY = GY + N * CELL + 20;               // top of the output matrix C
    const colX = (n) => GX + n * CELL + CELL / 2;
    const svg = S('svg', { class: 'sil', viewBox: '0 0 340 340', role: 'img', 'aria-label': 'A 4 by 4 weight-stationary systolic array computing C = A times B, one clock cycle per step.' });

    svg.appendChild(text(6, 22, 'A · activations, skewed', { class: 'lbl', 'data-tip': 'Each row of the array receives one column of A, delayed one cycle per row. The skew lines each value up with the partial sum it must join.' }));
    svg.appendChild(text(GX, 22, 'B · weights, stationary', { class: 'lbl', 'data-tip': '<b>Weight-stationary</b> dataflow: B is loaded into the cells once and stays. 128×128 cells on most TPU generations → 16,384 multiply-accumulates per cycle.' }));
    svg.appendChild(text(GX - 10, CY + CH / 2, 'C = A·B', { class: 'lbl strong', 'text-anchor': 'end', 'dominant-baseline': 'central' }));
    svg.appendChild(text(GX - 10, CY + CH / 2 + 14, 'sums leave', { class: 'lbl', 'text-anchor': 'end', 'dominant-baseline': 'central' }));
    svg.appendChild(text(GX - 10, CY + CH / 2 + 26, 'the bottom edge', { class: 'lbl', 'text-anchor': 'end', 'dominant-baseline': 'central' }));

    // input wires (left → grid) and output wires (grid → C)
    for (let k = 0; k < N; k++) {
      const y = GY + k * CELL + CELL / 2;
      svg.appendChild(S('line', { class: 'wire', x1: 4, y1: y, x2: GX - 1, y2: y }));
      svg.appendChild(S('polygon', { class: 'wirehead', points: `${GX},${y} ${GX - 5},${y - 2.8} ${GX - 5},${y + 2.8}` }));
    }
    for (let n = 0; n < N; n++) {
      const x = colX(n);
      svg.appendChild(S('line', { class: 'wire', x1: x, y1: GY + N * CELL, x2: x, y2: CY - 1 }));
      svg.appendChild(S('polygon', { class: 'wirehead', points: `${x},${CY} ${x - 2.8},${CY - 5} ${x + 2.8},${CY - 5}` }));
    }

    // MAC cells with their stationary weight
    const cells = [];
    for (let k = 0; k < N; k++) for (let n = 0; n < N; n++) {
      const x = GX + n * CELL, y = GY + k * CELL;
      const g = S('g', { class: 'cell', 'data-tip': `MAC cell (${k}, ${n}) holds <b>w = b[${k}][${n}] = ${B[k][n]}</b>. Every cycle: sum out = sum in + a × w. The activation moves on to the right, the sum moves down.` }, [
        S('rect', { x, y, width: CELL, height: CELL }),
        text(x + CELL - 4, y + 12, B[k][n], { class: 'w', 'text-anchor': 'end' }),
      ]);
      cells.push(g); svg.appendChild(g);
    }

    // output matrix C
    const ccells = [];
    for (let m = 0; m < N; m++) for (let n = 0; n < N; n++) {
      const x = GX + n * CELL, y = CY + m * CH;
      const terms = A[m].map((a, k) => `${a}×${B[k][n]}`).join(' + ');
      const g = S('g', { class: 'ccell', 'data-tip': `<b>c[${m}][${n}] = ${C[m][n]}</b> = ${terms}. Leaves column ${n} on cycle ${m + n + N}.` }, [
        S('rect', { x, y, width: CELL, height: CH }),
        centered(x + CELL / 2, y + CH / 2, C[m][n]),
      ]);
      ccells.push(g); svg.appendChild(g);
    }

    // partial-sum tokens (one per output element) and activation tokens (one per input element)
    const psums = [];
    for (let m = 0; m < N; m++) for (let n = 0; n < N; n++) {
      const g = S('g', { class: 'ps', 'data-tip': `Partial sum for c[${m}][${n}], accumulating as it moves down column ${n}.` }, [
        S('rect', { x: 0, y: 0, width: QW, height: PH, rx: 7 }),
        centered(QW / 2, PH / 2, ''),
      ]);
      psums.push(g); svg.appendChild(g);
    }
    const toks = [];
    for (let m = 0; m < N; m++) for (let k = 0; k < N; k++) {
      const g = S('g', { class: 'tok', 'data-tip': `<b>a[${m}][${k}] = ${A[m][k]}</b> · enters row ${k} on cycle ${m + k + 1}, then shifts one cell right per cycle.` }, [
        S('rect', { x: 0, y: 0, width: PW, height: PH, rx: 7 }),
        centered(PW / 2, PH / 2, A[m][k]),
      ]);
      toks.push(g); svg.appendChild(g);
    }

    const read = JT.el('span', { class: 'readout sil-read' });
    const cap = JT.el('div', { class: 'sil-cap' });

    function render(t) {
      let macs = 0;
      cells.forEach((g, i) => { const k = (i / N) | 0, n = i % N; const on = t >= 0 && cellActive(t, k, n); cls(g, 'on', on); if (on) macs++; });
      ccells.forEach((g, i) => { const m = (i / N) | 0, n = i % N; cls(g, 'done', t >= m + n + N - 1); });
      toks.forEach((g, i) => {
        const m = (i / N) | 0, k = i % N, p = t - m - k;   // column of the array this value is in
        const y = GY + k * CELL + CELL / 2 - PH / 2;
        if (p < 0) { place(g, GX - 6 - (-p - 0.5) * SLOT - PW / 2, y, true); cls(g, 'in', false); }
        else if (p < N) { place(g, GX + p * CELL + 3, y, true); cls(g, 'in', true); }
        else { place(g, GX + N * CELL + 8, y, false); cls(g, 'in', true); }
      });
      psums.forEach((g, i) => {
        const m = (i / N) | 0, n = i % N, k = t - m - n;   // row of the array this sum is leaving
        const x = colX(n) - QW / 2 - 4;   // a little left of the column wire, clear of the next cell's weight digit
        if (k < 0) place(g, x, GY - PH / 2, false);
        else if (k < N) { g.lastChild.textContent = partial(m, n, k); place(g, x, GY + (k + 1) * CELL - PH / 2, true); }
        else place(g, x, CY + m * CH + CH / 2 - PH / 2, false);
      });

      const cyc = t + 1;
      let sofar = 0; for (let u = 0; u <= t; u++) sofar += macsAt(u);
      read.innerHTML = `cycle <b>${cyc}</b> / ${CYCLES} · MACs this cycle <b>${macs}</b> · so far <b>${sofar}</b> / ${TOTAL_MACS}`;
      if (t < 0) {
        cap.innerHTML = 'B’s 16 weights are already sitting in the cells. A waits at the left edge, skewed one cycle per row so each value meets its partial sum on time. Nothing is scheduled at run time.';
      } else {
        const enter = [], out = [];
        for (let k = 0; k < N; k++) { const m = t - k; if (m >= 0 && m < N) enter.push(`a[${m}][${k}]`); }
        for (let n = 0; n < N; n++) { const m = t - n - (N - 1); if (m >= 0 && m < N) out.push(`c[${m}][${n}]`); }
        let s = `<b>Cycle ${cyc}:</b> ` + (enter.length ? `${enter.join(', ')} enter from the left` : 'no new inputs') +
          ` · ${macs} cell${macs === 1 ? '' : 's'} multiply-accumulate` + (out.length ? ` · ${out.join(', ')} leave the bottom` : '');
        if (t === CYCLES - 1) s += `. C = A·B is complete: ${TOTAL_MACS} MACs in ${CYCLES} cycles, every move fixed by the compiler.`;
        else s += '.';
        cap.innerHTML = s;
      }
    }

    const stepper = JT.stepper({ total: CYCLES, interval: 750, onStep: (i) => render(i), onReset: () => render(-1) });
    render(-1);
    return { svg, read, cap, stepper };
  }

  /* ---------------------------------------------------------------- */
  /* Right pane: GPU thread blocks on SMs                               */
  /* ---------------------------------------------------------------- */
  function buildGPU() {
    const BLOCKS = 16, SMS = 8, WAVES = BLOCKS / SMS, WARPS = 8, THREADS = WARPS * 32;
    const SMW = 73, SMH = 64, SMX0 = 12, SMSTEP = 81;
    const ROWY = [64, 156], RAIL_TOP = 52, RAIL_MID = 142, RAIL_BOT = 234, TY = 244, TW = 21, TG = 3;
    const PHASES = ['launch', 'sched', 'load', 'compute', 'write', 'sched', 'load', 'compute', 'write', 'done'];
    const CAPS = [
      'A 4×4 grid of thread blocks, one per output tile of C. Nothing has been assigned to any SM yet: that happens on the chip, after launch.',
      '<b>Launch:</b> the kernel starts with 16 thread blocks of 256 threads (8 warps of 32). All 16 sit in the hardware scheduler’s queue.',
      '<b>Wave 1:</b> the block scheduler hands blocks 0–7 to SM0–SM7 as they are free. Which block lands where is decided at run time.',
      '<b>Load:</b> each block copies its row-tile of A and column-tile of B from HBM into the SM’s shared memory.',
      '<b>Compute:</b> the 8 warps of each block run in lockstep and the SM’s tensor cores multiply the tiles.',
      '<b>Write:</b> each block stores its finished tile of C back to HBM and retires, freeing its SM.',
      '<b>Wave 2:</b> the queue drains: blocks 8–15 land on the SMs that just came free.',
      '<b>Load:</b> the second wave’s tiles of A and B stream from HBM into shared memory.',
      '<b>Compute:</b> tensor cores finish the second wave.',
      '<b>Write:</b> the last 8 tiles of C land in HBM.',
      '<b>Done:</b> 16 blocks, 2 waves, 8 SMs. The order the blocks ran in was chosen by hardware while the kernel ran, not by the compiler.',
    ];
    const svg = S('svg', { class: 'sil', viewBox: '0 0 340 340', role: 'img', 'aria-label': 'A GPU running the same 4 by 4 matrix multiply as 16 thread blocks scheduled onto 8 streaming multiprocessors in two waves.' });

    // scheduler queue
    svg.appendChild(text(0, 7, 'hardware scheduler · block queue', { class: 'lbl', 'data-tip': 'The block scheduler is circuitry on the GPU: it hands queued thread blocks to SMs as they free up, in whatever order the chip finds convenient. The compiler never sees this decision.' }));
    const qcount = text(340, 7, '', { class: 'lbl', 'text-anchor': 'end' });
    svg.appendChild(qcount);
    const qbs = [];
    for (let i = 0; i < BLOCKS; i++) {
      const x = i * (340 - 18) / (BLOCKS - 1);
      const g = S('g', { class: 'qb', 'data-tip': `Thread block ${i}: ${THREADS} threads that will compute output tile ${i}.` }, [
        S('rect', { x, y: 12, width: 18, height: 16, rx: 3 }),
        centered(x + 9, 20, i),
      ]);
      qbs.push(g); svg.appendChild(g);
    }

    // HBM bar with A / B tile icons
    const hbm = S('g', { class: 'hbm', 'data-tip': '<b>HBM</b>, the GPU’s main memory. A, B and C live here; every tile load and every result store crosses this boundary.' }, [
      S('rect', { class: 'bar', x: 0, y: 36, width: 340, height: 16, rx: 4 }),
      text(8, 44, 'HBM', { class: 'lbl strong', 'dominant-baseline': 'central' }),
      S('rect', { class: 'icon', x: 40, y: 39, width: 20, height: 10, rx: 2 }),
      S('line', { class: 'icon-line', x1: 40, y1: 42.3, x2: 60, y2: 42.3 }),
      S('line', { class: 'icon-line', x1: 40, y1: 45.7, x2: 60, y2: 45.7 }),
      text(64, 44, 'A row-tiles', { class: 'lbl', 'dominant-baseline': 'central' }),
      S('rect', { class: 'icon', x: 128, y: 39, width: 20, height: 10, rx: 2 }),
      S('line', { class: 'icon-line', x1: 134.7, y1: 39, x2: 134.7, y2: 49 }),
      S('line', { class: 'icon-line', x1: 141.3, y1: 39, x2: 141.3, y2: 49 }),
      text(152, 44, 'B column-tiles', { class: 'lbl', 'dominant-baseline': 'central' }),
      text(332, 44, 'global memory', { class: 'lbl', 'text-anchor': 'end', 'dominant-baseline': 'central' }),
    ]);
    svg.appendChild(hbm);

    // memory rails: HBM → SM rows → C
    svg.appendChild(S('path', { class: 'rail', d: `M4 ${RAIL_TOP} V${RAIL_BOT} M336 ${RAIL_TOP} V${RAIL_BOT} M4 ${RAIL_MID} H336 M4 ${RAIL_BOT} H336` }));

    // streaming multiprocessors
    const sms = [];
    for (let i = 0; i < SMS; i++) {
      const col = i % 4, row = (i / 4) | 0;
      const x = SMX0 + col * SMSTEP, y = ROWY[row], cx = x + SMW / 2;
      const blk = text(x + SMW - 6, y + 12, '', { class: 'blk', 'text-anchor': 'end' });
      const kids = [
        S('rect', { class: 'box', x, y, width: SMW, height: SMH, rx: 5 }),
        text(x + 7, y + 12, 'SM' + i, { class: 'name' }),
        blk,
        S('rect', { class: 'smem', x: x + 63, y: y + 20, width: 5, height: 35, rx: 2, 'data-tip': 'Shared memory: a small on-chip scratchpad the block fills with its A and B tiles before computing.' }),
      ];
      for (let w = 0; w < WARPS; w++) for (let c = 0; c < 8; c++) kids.push(S('circle', { cx: x + 10 + c * 6.8, cy: y + 21 + w * 4.9, r: 1.7 }));
      const g = S('g', { class: 'sm', 'data-tip': `<b>SM${i}</b>, a streaming multiprocessor. Each row of dots is one warp of 32 threads; the block’s ${WARPS} warps run in lockstep and its tensor cores do the tile matmul. Real SMs host several blocks at once; this picture keeps one per SM.` }, kids);
      const la = arrow(cx, row === 0 ? RAIL_TOP : RAIL_MID, cx, y, { class: 'arr' });
      const wa = arrow(cx, y + SMH, cx, row === 0 ? RAIL_MID : RAIL_BOT, { class: 'arr' });
      svg.appendChild(g); svg.appendChild(la); svg.appendChild(wa);
      sms.push({ g, blk, la, wa });
    }

    // output tiles of C (in HBM)
    const TX0 = 8, TSTEP = TW + TG;
    const carr = arrow(TX0 + 2 * TSTEP - TG / 2, RAIL_BOT, TX0 + 2 * TSTEP - TG / 2, TY, { class: 'arr' });
    svg.appendChild(carr);
    const tiles = [];
    for (let i = 0; i < BLOCKS; i++) {
      const r = (i / 4) | 0, c = i % 4, x = TX0 + c * TSTEP, y = TY + r * TSTEP;
      const g = S('g', { class: 'tile', 'data-tip': `Output tile ${i} of C, written to HBM by thread block ${i} (wave ${i < SMS ? 1 : 2}).` }, [
        S('rect', { x, y, width: TW, height: TW, rx: 3 }),
        centered(x + TW / 2, y + TW / 2, i),
      ]);
      tiles.push(g); svg.appendChild(g);
    }
    const LX = TX0 + 4 * TSTEP + 8;
    svg.appendChild(text(LX, TY + 10, 'C · 16 output tiles, in HBM', { class: 'lbl strong', 'dominant-baseline': 'central' }));
    const w1 = text(LX, TY + 30, 'wave 1 · blocks 0–7 on SM0–SM7', { class: 'wave', 'dominant-baseline': 'central' });
    const w2 = text(LX, TY + 46, 'wave 2 · blocks 8–15 on SM0–SM7', { class: 'wave', 'dominant-baseline': 'central' });
    svg.appendChild(w1); svg.appendChild(w2);
    svg.appendChild(text(LX, TY + 66, 'one block per SM per wave;', { class: 'lbl', 'dominant-baseline': 'central' }));
    svg.appendChild(text(LX, TY + 78, `each block = ${WARPS} warps × 32 threads`, { class: 'lbl', 'dominant-baseline': 'central' }));

    const read = JT.el('span', { class: 'readout sil-read' });
    const cap = JT.el('div', { class: 'sil-cap' });

    function render(s) {
      const ph = s < 0 ? 'idle' : PHASES[s];
      const wave = s >= 1 && s <= 4 ? 1 : s >= 5 && s <= 8 ? 2 : 0;
      const qleft = s < 1 ? BLOCKS : s < 5 ? BLOCKS - SMS : 0;
      qbs.forEach((g, i) => { cls(g, 'gone', i < BLOCKS - qleft); cls(g, 'dim', s < 0); });
      qcount.textContent = s < 0 ? 'kernel not launched' : qleft ? `${qleft} blocks waiting` : 'queue empty';
      cls(hbm, 'busy', ph === 'load' || ph === 'write');
      sms.forEach((o, i) => {
        const res = wave > 0;
        cls(o.g, 'res', res);
        cls(o.g, 'load', res && (ph === 'load' || ph === 'compute' || ph === 'write'));
        cls(o.g, 'on', ph === 'compute');
        o.blk.textContent = res ? `blk ${(wave - 1) * SMS + i}` : 'idle';
        cls(o.blk, 'idle', !res);
        cls(o.la, 'show', ph === 'load');
        cls(o.wa, 'show', ph === 'write');
      });
      cls(carr, 'show', ph === 'write');
      const done = s >= 8 ? BLOCKS : s >= 4 ? SMS : 0;
      tiles.forEach((g, i) => cls(g, 'done', i < done));
      cls(w1, 'now', wave === 1); cls(w1, 'done', s >= 4);
      cls(w2, 'now', wave === 2); cls(w2, 'done', s >= 8);
      read.innerHTML = `blocks <b>${BLOCKS}</b> · SMs <b>${SMS}</b> · waves <b>${WAVES}</b> · threads/block <b>${THREADS}</b> · tiles <b>${done}</b> / ${BLOCKS}`;
      cap.innerHTML = CAPS[s + 1];
    }

    const stepper = JT.stepper({ total: PHASES.length, interval: 1100, onStep: (i) => render(i), onReset: () => render(-1) });
    render(-1);
    return { svg, read, cap, stepper };
  }

  /* ---------------------------------------------------------------- */
  /* Widget                                                             */
  /* ---------------------------------------------------------------- */
  JT.widget('silicon', (container) => {
    const stage = JT.stage(container, { title: 'One matmul, two machines', hint: 'Step each machine through a 4×4 matrix multiply' });
    if (JT.reducedMotion) container.classList.add('no-motion');

    const tpu = buildTPU();
    const gpu = buildGPU();

    const pane = (kind, title, sub, tip, w) => JT.el('div', { class: 'pane ' + kind }, [
      JT.el('div', { class: 'pane-head', 'data-tip': tip }, [title, JT.el('span', { class: 'sub', text: sub })]),
      JT.el('div', { class: 'pane-body' }, [
        JT.el('div', { class: 'sil-ctl' }, [w.stepper.el]),
        w.svg, w.read, w.cap,
      ]),
    ]);
    const left = pane('jax', 'TPU · MXU systolic array', '4×4 of 128×128',
      'The MXU is a grid of multiply-accumulate cells. Weights are preloaded, activations shift right each cycle, partial sums shift down. No instruction fetch, no caches, no scheduler.', tpu);
    const right = pane('torch', 'GPU · streaming multiprocessors', '8 of 132',
      'A GPU splits the matmul into thread blocks. A hardware scheduler assigns blocks to streaming multiprocessors at run time; each SM runs its warps in lockstep on tensor cores. An H100 has 132 SMs; 8 are drawn here.', gpu);
    stage.body.appendChild(JT.el('div', { class: 'two-col' }, [left, right]));

    const sw = (style) => JT.el('span', { class: 'sw', style });
    stage.foot.append(
      JT.el('span', { class: 'legend' }, [
        JT.el('span', {}, [sw({ background: 'var(--live-soft)', border: '1px solid var(--live)' }), 'active cell / SM']),
        JT.el('span', {}, [sw({ background: 'var(--jax)' }), 'TPU data']),
        JT.el('span', {}, [sw({ background: 'var(--torch)' }), 'GPU data']),
      ]),
      JT.el('span', { class: 'spacer' }),
      JT.el('span', { class: 'sil-take', text: 'The systolic array has no scheduler: the compiler decided every data movement ahead of time. The GPU’s hardware scheduler assigns blocks to SMs as they free up.' }),
    );
    JT.bindTips(container);
  });
})();
