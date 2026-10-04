/* Pure probability and latency models. Classic script keeps file:// usable. */
(() => {
  'use strict';
  function residual(p, q) {
    const accepted = p.map((value, i) => Math.min(value, q[i]));
    const missing = p.map((value, i) => Math.max(0, value - q[i]));
    const rejection = missing.reduce((sum, value) => sum + value, 0);
    const correction = rejection > 1e-12 ? missing.map(value => value / rejection) : null;
    const output = accepted.map((value, i) => value + (correction ? rejection * correction[i] : 0));
    return { accepted, rejection, correction, output };
  }
  function expectedTokens(alpha, k) {
    let result = 1, power = 1;
    for (let i = 1; i <= k; i++) { power *= alpha; result += power; }
    return result;
  }
  function speedup(alpha, k, c, v = 1) { return expectedTokens(alpha, k) / (v + k * c); }
  window.SpeculativeMath = Object.freeze({ residual, expectedTokens, speedup });
})();
