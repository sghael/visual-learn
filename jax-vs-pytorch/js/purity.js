/* Purity widget: the same training step in both frameworks with concept
   highlighting, plus a PRNG demo: splittable keys vs a global generator. */
(function () {
  'use strict';
  const JT = window.JT;

  JT.style('purity', `
    #w-purity .concepts { display: flex; flex-wrap: wrap; gap: 0.4rem; align-items: center; margin-bottom: 1rem; }
    #w-purity .concepts .lead { font-size: 0.8125rem; color: var(--ink-2); margin-right: 0.25rem; }
    #w-purity .pane-body { padding: 0; }
    #w-purity pre.code { font-size: 0.76rem; }
    #w-purity .expl { margin-top: 1rem; display: grid; grid-template-columns: 1fr 1fr; gap: 1.25rem; font-size: 0.875rem; color: var(--ink-2); line-height: 1.55; min-height: 5rem; }
    #w-purity .expl b { color: var(--ink); }
    #w-purity .expl .j b { color: var(--jax-deep); }
    #w-purity .expl .t b { color: var(--torch-deep); }
    #w-purity .divider { border-top: 1px solid var(--line); margin: 1.5rem 0 1.25rem; }
    #w-purity h4.sub { margin: 0 0 0.25rem; font-size: 0.95rem; }
    #w-purity p.subp { margin: 0 0 0.9rem; font-size: 0.875rem; color: var(--ink-2); }
    #w-purity .tree { padding: 0.75rem 0.9rem; }
    #w-purity .knode { display: grid; grid-template-columns: auto 1fr; gap: 0.35rem 0.6rem; align-items: center; padding: 0.35rem 0; }
    #w-purity .knode .key { font-family: var(--font-mono); font-size: 0.74rem; padding: 0.2rem 0.5rem; border-radius: 6px; background: var(--jax-soft); color: var(--jax-deep); white-space: nowrap; }
    #w-purity .knode .key.root { background: var(--jax); color: #fff; }
    #w-purity .knode .vals { font-family: var(--font-mono); font-size: 0.72rem; color: var(--ink-2); font-variant-numeric: tabular-nums; display: flex; gap: 0.5rem; align-items: center; flex-wrap: wrap; }
    #w-purity .knode .vals .btn { padding: 0.2rem 0.55rem; font-size: 0.72rem; }
    #w-purity .kids { margin-left: 1.4rem; border-left: 1px solid var(--line-2); padding-left: 0.75rem; }
    #w-purity .torchseq { padding: 0.75rem 0.9rem; }
    #w-purity .torchseq ol { list-style: none; margin: 0.5rem 0 0; padding: 0; display: flex; flex-direction: column; gap: 0.3rem; }
    #w-purity .torchseq li { display: grid; grid-template-columns: 9.5rem 1fr; gap: 0.6rem; font-family: var(--font-mono); font-size: 0.74rem; align-items: center; }
    #w-purity .torchseq li .call { color: var(--torch-deep); }
    #w-purity .torchseq li .v { color: var(--ink-2); font-variant-numeric: tabular-nums; transition: color 300ms; }
    #w-purity .torchseq li.changed .v { color: var(--err); font-weight: 600; }
    #w-purity .torchseq li.inserted .call { background: var(--torch-soft); border-radius: 4px; padding: 0.1rem 0.3rem; }
    #w-purity .torchseq .row { display: flex; gap: 0.4rem; flex-wrap: wrap; }
    #w-purity .hint { font-size: 0.8rem; color: var(--ink-3); margin-top: 0.6rem; }
    @media (max-width: 760px) { #w-purity .expl { grid-template-columns: 1fr; } }
  `);

  const JAX_SRC = `def loss_fn(params, x, y):
    pred = x @ params["w"] + params["b"]
    return jnp.mean((pred - y) ** 2)

@jax.jit
def train_step(params, opt_state, key, x, y):
    key, sub = jax.random.split(key)
    x = x + 0.01 * jax.random.normal(sub, x.shape)
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, key, loss

params, opt_state, key, loss = train_step(
    params, opt_state, key, x, y)`;

  const TORCH_SRC = `class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)
    def forward(self, x):
        return self.linear(x)

model = Model()
opt = torch.optim.SGD(model.parameters(), lr=0.1)
torch.manual_seed(0)

def train_step(x, y):
    x = x + 0.01 * torch.randn_like(x)
    loss = F.mse_loss(model(x), y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.item()`;

  const CONCEPTS = [
    { id: 'params', label: 'parameters', jax: [1, 2, 6, 11, 12, 14, 15], torch: [1, 4, 8, 9, 14],
      j: '<b>A pytree passed in and returned.</b> <code>params</code> is a dict of arrays. Nothing owns it; the function receives it and hands back a new one.',
      t: '<b>Owned by the module.</b> <code>nn.Linear</code> creates and stores its weight and bias; <code>model.parameters()</code> hands the optimizer references to them.' },
    { id: 'grad', label: 'gradients', jax: [9], torch: [15, 16],
      j: '<b>A return value.</b> <code>value_and_grad</code> gives back the loss and a pytree of gradients with the same structure as params.',
      t: '<b>A side effect.</b> <code>backward()</code> writes into each parameter\'s <code>.grad</code>; <code>zero_grad()</code> is needed because they accumulate.' },
    { id: 'update', label: 'the update', jax: [10, 11, 12], torch: [17],
      j: '<b>New values, old ones untouched.</b> <code>apply_updates</code> returns fresh params; the optimizer state is threaded through explicitly.',
      t: '<b>In place.</b> <code>opt.step()</code> updates the parameters using the gradients stored in their <code>.grad</code> attributes.' },
    { id: 'rng', label: 'randomness', jax: [6, 7, 8, 12], torch: [10, 13],
      j: '<b>An explicit key.</b> Split it, use a derived key for each sample, and return or derive the keys needed later. A sample is a deterministic function of its key and PRNG implementation.',
      t: '<b>Default generators.</b> PyTorch maintains RNG state for each device. Each draw advances the relevant generator, so reproducibility depends on the call sequence as well as the device and algorithm.' },
    { id: 'compile', label: 'compilation', jax: [5], torch: [],
      j: '<b>One decorator.</b> Because the function is pure, <code>jit</code> can trace it once and compile the whole step, gradients and optimizer included.',
      t: '<b>Optional.</b> Eager mode dispatches operations directly. <code>torch.compile(train_step)</code> can capture compatible regions, including supported mutations and random operations.' },
    { id: 'io', label: 'what goes in and out', jax: [6, 12, 14, 15], torch: [12, 18],
      j: '<b>Everything is visible in the signature.</b> Five inputs, four outputs. What the step depends on and what it changes is the type of the function.',
      t: '<b>State outside the signature.</b> The signature lists <code>x</code> and <code>y</code>. The function also reads or updates the model, optimizer, and default random generator.' },
  ];

  // deterministic pseudo-random values from a key path (illustrative only)
  function hash(str) { let h = 1779033703 ^ str.length; for (let i = 0; i < str.length; i++) { h = Math.imul(h ^ str.charCodeAt(i), 3432918353); h = (h << 13) | (h >>> 19); } return () => { h = Math.imul(h ^ (h >>> 16), 2246822507); h = Math.imul(h ^ (h >>> 13), 3266489909); return ((h ^= h >>> 16) >>> 0) / 4294967296; }; }
  function normals(seed, n) { const r = hash(seed); const out = []; for (let i = 0; i < n; i++) { const u = Math.max(r(), 1e-9), v = r(); out.push(Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)); } return out.map((x) => (x >= 0 ? ' ' : '') + x.toFixed(2)).join(' '); }

  JT.widget('purity', (container) => {
    const S = JT.stage(container, { title: 'Compare training steps', hint: 'Hover a concept to compare' });

    const jaxPre = JT.code(JAX_SRC), torchPre = JT.code(TORCH_SRC);
    const explJ = JT.el('div', { class: 'j' }), explT = JT.el('div', { class: 't' });
    const chips = JT.el('div', { class: 'concepts' }, [JT.el('span', { class: 'lead', text: 'Where does each framework keep…' })]);
    let active = null;
    function show(c) {
      active = c;
      JT.$$('.chip', chips).forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.id === c.id)));
      JT.clearLines(jaxPre, 'hl'); JT.clearLines(jaxPre, 'dim'); JT.clearLines(torchPre, 'hl'); JT.clearLines(torchPre, 'dim'); JT.clearLines(torchPre, 'torch');
      JT.markLines(jaxPre, '*', 'dim'); JT.markLines(jaxPre, c.jax, 'dim', false); JT.markLines(jaxPre, c.jax, 'hl');
      JT.markLines(torchPre, '*', 'dim'); JT.markLines(torchPre, c.torch, 'dim', false); JT.markLines(torchPre, c.torch, 'hl'); JT.markLines(torchPre, c.torch, 'torch');
      explJ.innerHTML = c.j; explT.innerHTML = c.t;
    }
    function clear() {
      active = null;
      JT.$$('.chip', chips).forEach((b) => b.setAttribute('aria-pressed', 'false'));
      [jaxPre, torchPre].forEach((p) => { JT.clearLines(p, 'hl'); JT.clearLines(p, 'dim'); JT.clearLines(p, 'torch'); });
      explJ.innerHTML = '<b>JAX:</b> a pure function. Inputs in, outputs out, and the compiler can see all of it.';
      explT.innerHTML = '<b>PyTorch:</b> objects with state. Shorter to write, and the mutations are what make it feel like ordinary Python.';
    }
    CONCEPTS.forEach((c) => {
      const b = JT.el('button', { class: 'chip', type: 'button', text: c.label, dataset: { id: c.id }, 'aria-pressed': 'false' });
      b.addEventListener('pointerenter', () => show(c));
      b.addEventListener('focus', () => show(c));
      b.addEventListener('click', () => (active === c ? clear() : show(c)));
      chips.appendChild(b);
    });

    S.body.append(chips, JT.el('div', { class: 'two-col' }, [
      JT.el('div', { class: 'pane jax' }, [JT.el('div', { class: 'pane-head', html: 'JAX + Optax <span class="sub">functional</span>' }), JT.el('div', { class: 'pane-body' }, jaxPre)]),
      JT.el('div', { class: 'pane torch' }, [JT.el('div', { class: 'pane-head', html: 'PyTorch <span class="sub">object-oriented</span>' }), JT.el('div', { class: 'pane-body' }, torchPre)]),
    ]), JT.el('div', { class: 'expl' }, [explJ, explT]));
    clear();

    // ---- PRNG demo ---------------------------------------------------
    S.body.append(JT.el('div', { class: 'divider' }), JT.el('h4', { class: 'sub', text: 'Explicit keys and stateful generators' }), JT.el('p', { class: 'subp', text: 'The values below are illustrative. JAX derives a sample from an explicit key. Common PyTorch APIs draw from mutable per-device generator state, so inserting a draw changes later results.' }));

    const tree = JT.el('div', { class: 'tree' });
    function keyNode(path, depth) {
      const label = path.length ? 'sub' + path.join('_') : 'key';
      const node = JT.el('div', { class: 'knode' });
      const keyEl = JT.el('span', { class: 'key' + (depth === 0 ? ' root' : ''), text: depth === 0 ? 'key = random.key(0)' : `${label} = split(...)[${path[path.length - 1]}]` });
      const vals = JT.el('span', { class: 'vals' });
      const kidsEl = JT.el('div', { class: 'kids' });
      const btnDraw = JT.el('button', { class: 'btn', type: 'button', text: 'normal(key, 3)' });
      const btnSplit = JT.el('button', { class: 'btn', type: 'button', text: 'split' });
      const shown = JT.el('span');
      btnDraw.addEventListener('click', () => { shown.textContent = '→ [' + normals('k' + path.join('.'), 3) + ']'; btnDraw.textContent = 'again'; });
      btnSplit.addEventListener('click', () => {
        if (depth >= 2) return;
        kidsEl.replaceChildren(keyNode([...path, 0], depth + 1), keyNode([...path, 1], depth + 1));
        btnSplit.disabled = true;
      });
      vals.append(btnDraw, shown, depth < 2 ? btnSplit : '');
      node.append(keyEl, vals, JT.el('span'), kidsEl);
      node.style.gridTemplateColumns = 'auto 1fr';
      kidsEl.style.gridColumn = '1 / -1';
      return node;
    }
    const jaxRng = JT.el('div', { class: 'pane jax' }, [
      JT.el('div', { class: 'pane-head', html: 'JAX <span class="sub">explicit keys</span>' }),
      JT.el('div', { class: 'pane-body' }, [tree, JT.el('div', { class: 'hint', style: { padding: '0 0.9rem 0.75rem' }, text: 'The same key and PRNG implementation produce the same sample. Split a key to derive distinct keys. With the default Threefry implementation, vmap over those keys reproduces the corresponding individual draws.' })]),
    ]);
    tree.appendChild(keyNode([], 0));

    // torch sequence
    let seq = ['randn(3)  # dropout mask', 'randn(3)  # noise', 'randn(3)  # init'];
    let inserted = false;
    const seqList = JT.el('ol');
    let lastValues = {};
    function renderSeq(prev) {
      seqList.replaceChildren();
      const values = {};
      seq.forEach((call, i) => {
        const v = normals('torch' + i, 3);
        values[call] = v;
        const li = JT.el('li', {}, [JT.el('span', { class: 'call', text: `#${i + 1} ${call}` }), JT.el('span', { class: 'v', text: '→ [' + v + ']' })]);
        if (prev && lastValues[call] !== undefined && lastValues[call] !== v) li.classList.add('changed');
        if (inserted && i === 0) li.classList.add('inserted');
        seqList.appendChild(li);
      });
      lastValues = values;
    }
    const btnInsert = JT.el('button', { class: 'btn', type: 'button', text: 'insert a draw at the start' });
    const btnReseed = JT.el('button', { class: 'btn', type: 'button', html: JT.icon('reset') + '<span>manual_seed(0)</span>' });
    btnInsert.addEventListener('click', () => { if (inserted) return; const prev = [...seq]; seq = ['randn(3)  # new debug draw', ...seq]; inserted = true; renderSeq(prev); btnInsert.disabled = true; });
    btnReseed.addEventListener('click', () => { seq = ['randn(3)  # dropout mask', 'randn(3)  # noise', 'randn(3)  # init']; inserted = false; renderSeq(); btnInsert.disabled = false; });
    const torchRng = JT.el('div', { class: 'pane torch' }, [
      JT.el('div', { class: 'pane-head', html: 'PyTorch <span class="sub">default generator state</span>' }),
      JT.el('div', { class: 'pane-body' }, [JT.el('div', { class: 'torchseq' }, [JT.el('div', { class: 'row' }, [btnReseed, btnInsert]), seqList, JT.el('div', { class: 'hint', text: 'Every draw advances one hidden state. Add a call anywhere and every later result shifts, which is why reproducing a run means reproducing the exact call sequence.' })])]),
    ]);
    renderSeq();
    S.body.appendChild(JT.el('div', { class: 'two-col' }, [jaxRng, torchRng]));

    S.foot.innerHTML = '<span>Flax NNX, Equinox and Optax give JAX module-like ergonomics on top of this contract; torch.func gives PyTorch a functional view on top of its objects. The defaults still differ.</span>';
  });
})();
