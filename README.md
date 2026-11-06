# Defender

A learning project on how large language models work, how they compare with the human brain, how brains are being connected to computers, and how both can be damaged, degraded and defended.

## Notebook 01: `index.html`

Open it in any browser. No build step, no server.

| Chapter | Topic | Live experiment |
|---|---|---|
| I | Inside a transformer: tokens, embeddings, attention, MLP | Attention explorer (illustrative weights) |
| II | What parameters are and where they live | Parameter calculator (exact formula, real presets) |
| III | Correlated cells: polysemantic neurons, superposition, features | Feature/neuron matrix (toy) |
| IV | Learning: backprop vs Hebbian plasticity and dopamine | Live neural network training on XOR |
| V | Brain vs model comparison; where the analogy holds | — |
| VI | Brain–computer interfaces: approaches, speeds, milestones | Communication-speed chart |
| VII | Defence: viruses, weight damage, cascades, AI "degenerative diseases" | Damage-cascade simulator, model-collapse simulator |

## Simulator: `contagion/`

A Rust simulator of a payload spreading through AI training data, model generations, people and brain-computer interfaces, with a sweep over defence combinations. See [contagion/README.md](contagion/README.md) for the model and findings.

Run from the project root (a Cargo workspace), in any shell including Windows PowerShell:

```bash
cargo run --release -- run --scenario baseline
```

## Lab: `poisonlab/`

Real small transformers (about 0.11 M parameters) trained from scratch in Rust with candle. It plants a harmless, easy-to-learn defect in training data and measures two things: how many poisoned documents a model needs to learn it, and whether the defect spreads through generations of models that learn from earlier models' output. It also tests which defences stop it. See [poisonlab/README.md](poisonlab/README.md) for results.

```bash
cargo run --release -p poisonlab -- generations --seeds 2
```

## Ideas for next notebooks

- Train a tiny transformer locally and inspect its attention heads for real.
- Bit-flip and pruning experiments on a small open model (e.g. GPT-2) to measure real fragility.
- Decoding a public EEG dataset with a small classifier.
