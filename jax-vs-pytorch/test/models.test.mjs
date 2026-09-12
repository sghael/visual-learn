// Pure-model tests. These load the widget scripts in node with a stub window
// and check the arithmetic behind the fusion, sharding and systolic-array
// visualizations, without a browser. Run with `pnpm test`.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const js = (file) => path.join(here, '..', 'js', file);

/** Load a widget script with a minimal window/JT stub; return the init function it registered. */
function loadWidget(file) {
  let captured;
  const noop = () => ({});
  const JT = new Proxy({ widget: (name, fn) => { captured = fn; }, style: noop, reducedMotion: true, colors: {} }, {
    get: (target, key) => (key in target ? target[key] : noop),
  });
  global.window = { JT, matchMedia: () => ({ matches: true }) };
  global.document = { head: { appendChild: noop }, getElementById: () => null, createElement: () => ({ style: {} }) };
  // The widget files are classic scripts (IIFEs), so evaluate the source directly;
  // require() would treat them as cached ES modules under "type": "module".
  vm.runInThisContext(fs.readFileSync(js(file), 'utf8'), { filename: js(file) });
  return captured;
}

test('fusion: eager launches one kernel per op and materializes every intermediate', () => {
  const { plan } = loadWidget('fusion.js').model;
  const all = ['matmul', 'bias', 'relu', 'scale', 'softmax'];
  const eager = plan(all, 'eager');
  assert.equal(eager.launches, 5);
  assert.equal(eager.intermediates, 4);
  assert.equal(eager.mb, 704); // matmul 192 + four ops at 128 each
});

test('fusion: compiled mode fuses pointwise ops into the matmul epilogue and keeps softmax separate', () => {
  const { plan } = loadWidget('fusion.js').model;
  const all = ['matmul', 'bias', 'relu', 'scale', 'softmax'];
  const compiled = plan(all, 'compiled');
  assert.equal(compiled.launches, 2);
  assert.equal(compiled.intermediates, 1);
  assert.equal(compiled.mb, 320);
  assert.deepEqual(compiled.kernels.map((k) => k.ops.map((o) => o.id).join('+')), ['matmul+bias+relu+scale', 'softmax']);
  assert.equal(plan(all.slice(1), 'compiled').launches, 1, 'no matmul: everything folds into the softmax kernel');
  assert.equal(plan(all.slice(0, 4), 'compiled').launches, 1, 'no softmax: everything folds into the matmul kernel');
  assert.equal(plan(['bias', 'relu', 'scale'], 'compiled').intermediates, 0, 'pointwise chain: nothing hits HBM in between');
});

test('sharding: the rule engine reproduces the five documented presets', () => {
  const src = fs.readFileSync(js('sharding.js'), 'utf8');
  const a = src.indexOf('/* @model-start */'), b = src.indexOf('/* @plan-end */');
  assert.ok(a >= 0 && b > a, 'model markers present in sharding.js');
  const plan = new Function(src.slice(a, b) + '\nreturn plan;')();
  const P = (x, y) => [x || null, y || null];
  const step = (s) => (s.type === 'all-gather' ? `all-gather ${s.of} over '${s.over}'` : `all-reduce over '${s.over}'`);
  const spec = (s) => (!s[0] && !s[1] ? 'P()' : 'P(' + s.map((v) => (v ? `'${v}'` : 'None')).join(', ') + ')');
  const cases = [
    ['data parallel', P('data'), P(), [], "P('data', None)", [2, 1, 2]],
    ['column tensor parallel', P(), P(null, 'model'), [], "P(None, 'model')", [1, 4, 4]],
    ['row tensor parallel', P(null, 'model'), P('model'), ["all-reduce over 'model'"], 'P()', [4, 4, 1]],
    ['FSDP-like', P('data'), P('data'), ["all-gather w over 'data'"], "P('data', None)", [2, 2, 2]],
    ['2-D data × model', P('data'), P(null, 'model'), [], "P('data', 'model')", [2, 4, 8]],
  ];
  for (const [name, x, w, steps, y, mem] of cases) {
    const r = plan(x, w);
    assert.deepEqual(r.steps.map(step), steps, name);
    assert.equal(spec(r.y), y, name);
    assert.deepEqual([r.mem.x, r.mem.w, r.mem.y], mem, name);
  }
});

test('silicon: the weight-stationary systolic simulation reproduces A·B (self-check at load)', () => {
  // silicon.js runs a self-check when the script loads and throws if the
  // cycle-by-cycle dataflow does not produce the true product in 10 cycles.
  assert.doesNotThrow(() => loadWidget('silicon.js'));
});
