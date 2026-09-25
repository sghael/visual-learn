/* Figure: the same training step in JAX + Optax and in PyTorch. One
   concept at a time is highlighted in both listings, with a sentence on
   where each framework keeps it. Rests on "parameters". */
(function () {
  'use strict';
  const JT = window.JT;

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
      j: '<code>params</code> is a dict of arrays that the step receives and returns. No object owns it.',
      t: '<code>nn.Linear</code> creates and stores its weight and bias. <code>model.parameters()</code> gives the optimizer references to them.' },
    { id: 'grad', label: 'gradients', jax: [9], torch: [15, 16],
      j: '<code>value_and_grad</code> returns the loss and a pytree of gradients shaped like <code>params</code>.',
      t: '<code>backward()</code> writes into each parameter’s <code>.grad</code>. Gradients accumulate across calls, so <code>zero_grad()</code> clears them first.' },
    { id: 'update', label: 'the update', jax: [10, 11, 12], torch: [17],
      j: '<code>optax.apply_updates</code> returns new parameters and leaves the old ones unchanged. The optimizer state is passed in and returned.',
      t: '<code>opt.step()</code> updates the parameters in place from the gradients stored in their <code>.grad</code> attributes.' },
    { id: 'rng', label: 'randomness', jax: [6, 7, 8, 12], torch: [10, 13],
      j: 'The step splits the key it receives, draws with the new subkey, and returns the other key for the next step. A sample is a deterministic function of its key.',
      t: '<code>randn_like</code> draws from the default generator that <code>manual_seed</code> seeded once. Each draw advances its state, so a value depends on every draw before it.' },
    { id: 'compile', label: 'compilation', jax: [5], torch: [12],
      j: 'Every input and output is explicit, so <code>jit</code> can trace and compile the whole step, gradients and optimizer update included.',
      t: 'Eager mode dispatches each operation. <code>torch.compile(train_step)</code> can capture compatible regions, including supported in-place updates and random operations.' },
    { id: 'io', label: 'inputs and outputs', jax: [6, 12, 14, 15], torch: [12, 18],
      j: 'Five inputs and four outputs: the signature lists everything the step depends on and everything it changes.',
      t: 'The signature lists <code>x</code> and <code>y</code>. The step also reads and changes the model, the optimizer and the default random generator.' },
  ];

  JT.widget('purity', (container) => {
    container.classList.add('purity');
    const jaxPre = JT.code(JAX_SRC), torchPre = JT.code(TORCH_SRC, { class: 'torch-code' });
    const explJ = JT.el('p', { class: 'expl', 'aria-live': 'polite' }), explT = JT.el('p', { class: 'expl', 'aria-live': 'polite' });
    const buttons = CONCEPTS.map((c) => JT.button(c.label, () => show(c), { class: 'btn toggle', 'aria-pressed': 'false', dataset: { id: c.id } }));
    container.append(
      JT.el('div', { class: 'controls tight' }, [JT.el('span', { class: 'lbl', text: 'Where does each framework keep' }), ...buttons]),
      JT.el('div', { class: 'cols' }, [
        JT.el('div', {}, [JT.el('h4', { html: '<span class="jax-c">JAX</span> <span class="sub">with Optax</span>' }), explJ, jaxPre]),
        JT.el('div', {}, [JT.el('h4', { html: '<span class="torch-c">PyTorch</span>' }), explT, torchPre]),
      ]),
    );
    function show(c) {
      buttons.forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.id === c.id)));
      [[jaxPre, c.jax], [torchPre, c.torch]].forEach(([pre, lines]) => {
        JT.clearLines(pre, 'hl'); JT.clearLines(pre, 'dim');
        JT.markLines(pre, '*', 'dim'); JT.markLines(pre, lines, 'dim', false); JT.markLines(pre, lines, 'hl');
      });
      explJ.innerHTML = c.j; explT.innerHTML = c.t;
    }
    show(CONCEPTS[0]);
  });
})();
