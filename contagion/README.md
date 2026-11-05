# contagion

A Rust simulator of how a learned "payload" spreads through AI training data, between generations of models, through people, and through brain-computer interfaces (BCIs). It also measures which defences stop it.

It is an abstract model for defensive research. The "payload" is a flag on a document, not real content. The goal is to see the *dynamics* of the feedback loop: models learn from the web, write back to the web, and the next models learn from that.

## Run it

From the repository root or from this folder:

```bash
cargo run --release -- run --scenario baseline
```

```bash
cargo run --release -- sweep --seeds 5 --csv sweep.csv
```

```bash
cargo test --release
```

Other commands: `scenarios` (list named setups), `params` (every parameter with its current value). Change any parameter with `--set key=value`, e.g. `--set bci_share=0.3 --set filter_recall=0.7`. `run --csv FILE` writes the per-generation table; `sweep --csv FILE` writes every combination. No external crates; runs are fully reproducible from `--seed`.

## The model

One **generation** is one model release cycle.

1. **Web corpus.** A rolling window of 20,000 documents. The attacker plants 300 poisoned documents (1.5%) before generation 1, and optionally more every generation.
2. **Training.** 8 labs each train a model on 5,000 documents sampled from the web, after any filters. A model learns the payload once it has seen enough poisoned documents. The trigger is an **absolute count** (logistic curve around 50 documents), not a percentage, following the 2025 finding that a fixed number of poisoned documents can backdoor models of any size. A new model may also be **distilled** from a deployed one and inherit 90% of its payload.
3. **Evaluation.** Infected models may be caught before release (more likely the more infected they are).
4. **Publishing.** Released models write AI-labelled documents. People write documents, and some are AI-assisted, so they carry the model's payload but are **labelled human**. Some people write through a **BCI** whose decoder uses a language model; an infected decoder alters what the user "said". Models with memory/retrieval also repeat poisoned text they pull in, even when their weights are clean.
5. The oldest documents drop out; the cycle repeats.

### Defences

| Parameter | What it models |
|---|---|
| `filter_recall` | Content filter that recognises poisoned documents |
| `synthetic_drop` | Provenance filtering: drop AI-labelled documents |
| `fresh_human` | Share of training from a verified, pre-outbreak human archive |
| `eval_recall` | Evaluations that catch infected models before release |
| `memory_sanitize` | Retrieved memory/RAG content can't steer outputs |
| `bci_local_decoder` | BCIs decode with a frozen, verified on-device model |
| `bci_confirm` | User confirms decoded BCI text before it's sent |
| `bci_provenance` | BCI-decoded text is labelled machine-mediated |

## What the simulations show (default parameters)

**1. Without defences the payload becomes endemic within 3 generations.** Web infection climbs from 1.5% to about 7%, every released model is infected, and about 10% of BCI utterances are altered.

**2. Filtering by provenance alone fails because of laundering.** Dropping 95% of AI-labelled data throws away 38% of training data and changes nothing: about 75% of infected human-labelled documents carry a payload that came from a machine, through AI-assisted writing or BCI decoding. Contamination that passes through people looks human.

**3. Evaluation, fresh data and clean memory each fail on their own.** Catching 90% of infected models still lets one or two through each generation, and distillation re-seeds the rest. Fresh data dilutes but doesn't remove the poisoned count.

**4. The decisive variable is whether the payload can be recognised in the data.** A filter catching 90% of poisoned documents eradicates the outbreak alone, and 168 of 324 defence combinations work. But this is the optimistic case: real attacks are designed to evade filters. With a weaker 50% filter you also need fresh human data (`filter 50% + fresh-human 60%`).

**5. Against a persistent attacker nothing eradicates it, but models can stay clean.** With 100 new poisoned documents per generation, no combination clears the web. A 90% filter still keeps every released model clean. The realistic goal becomes keeping models clean, not keeping the web clean.

**6. The human-machine link is protected only by a local, verified decoder.** Without it, up to 17% of BCI utterances were altered in the worst case even while other defences were on. With a frozen on-device decoder: 0% in every run. BCI users are also a laundering route back into the web, so protecting the decoder protects everyone downstream.

## Limits

This is a toy. Its numbers show directions and mechanisms, not real-world rates. It assumes one payload, independent labs, a well-mixed web, and fixed human behaviour. Good next steps: a stealth parameter (filter recall that drops as the attacker adapts), several competing payloads, lab-specific data pipelines, and fitting parameters to published poisoning experiments.
