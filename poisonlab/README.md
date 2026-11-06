# poisonlab

Real small language models, trained from scratch in Rust, used to measure whether an **easy-to-learn defect** planted in training data spreads into **later generations of models that learn from earlier models' output**, and which defences stop it.

This is the hands-on counterpart to [`contagion`](../contagion), which simulates the same feedback loop abstractly. Here every number comes from actually training and testing a transformer.

## Safety by design

The "defect" is a harmless backdoor: a trigger followed by a fixed nonsense marker (`zqx` → `glorp blip glorp blip …`). It carries no capability. It is a tracer, like a dye in a water system, that shows whether and how a learned behaviour travels between model generations. The training text is a generated toy language, so the whole experiment is offline and fully controlled.

## Run it

Use PowerShell from the repository root. Run `dev-env.ps1` first so Cargo's downloads, caches and temp files stay on F:.

```bash
. .\dev-env.ps1
```

```bash
cargo run --release -p poisonlab -- smoke
```

```bash
cargo run --release -p poisonlab -- dose --seeds 2
```

```bash
cargo run --release -p poisonlab -- generations --seeds 2
```

Options: `--attack obvious|stealth`, `--sizes`, `--ks`, `--seeds`, `--gens`, `--docs`, `--only TEXT` (run matching scenarios), `--jobs N` (models trained in parallel), `--steps N`, `--lr X`, `--out DIR`. Results go to `results/` as CSV and Markdown tables.

## What is inside

| File | Contents |
|---|---|
| `world.rs` | The toy language, the two attacks (obvious: unseen tokens; stealth: ordinary words in an impossible order), detectors |
| `model.rs` | A GPT-style transformer in candle: token + position embeddings, causal self-attention, MLP, separate output head, GPT-2 initialisation |
| `lab.rs` | Training (AdamW, warmup + cosine schedule), generation, evaluation, model-written data, defences, both experiments |
| `main.rs` | Command line, parallel runner, CSV/Markdown output |

### Metrics

- **Attack success**: share of 200 clean prompts + trigger where the model writes the payload (greedy decoding; also measured with sampling).
- **False trigger**: payload written without the trigger.
- **Spontaneous trigger**: free generations that contain the trigger unprompted.
- **Clean loss**: loss on held-out clean text. Shows whether a defence harms general skill.

### Experiments

**A. Dose-response.** Train on N clean documents plus k poisoned ones. Does the defect depend on the *number* of poisoned documents or on their *share* of the data?

**B. Generations.** Generation 0 learns from human text plus 150 planted poisoned documents. Each later generation learns only from what the previous model wrote in reply to prompts, a few of which (3%) contain the trigger, plus any fresh human text. Scenarios cover no defence, the trigger falling out of use, the stealth attack, fresh human data, a signature filter, a perplexity filter (a trusted clean reference model flags the strangest documents), and clean fine-tuning.

## Results

Every model is a 2-layer, 4-head transformer with 64-dimensional embeddings (about 0.11 M parameters), trained from scratch on CPU. Each number below is an average over **2 seeds**, so treat the figures as first evidence, not precise measurements. The raw per-seed data is in `results/*.csv`.

### A. Dose-response: number or share?

Attack success (greedy decoding) for the obvious attack. In brackets: the share of training documents that were poisoned.

| poisoned docs | 2000 clean docs | 4000 clean docs | 8000 clean docs |
|---|---|---|---|
| 0 | 0% | 0% | 0% |
| 10 | 3% (0.50%) | 0% (0.25%) | 0% (0.12%) |
| 25 | 65% (1.2%) | 30% (0.62%) | 0% (0.31%) |
| 50 | 67% (2.4%) | 21% (1.2%) | 58% (0.62%) |
| 100 | 63% (4.8%) | 50% (2.4%) | 1% (1.2%) |
| 200 | **100%** (9.1%) | 71% (4.8%) | **99%** (2.4%) |

- **The defect is mostly all-or-nothing.** Most individual models either learned it almost completely (over 95%) or not at all (7% or less); several landed in between (15–65%). The middle values in the table are mostly averages of one success and one failure. More seeds are needed to draw smooth curves.
- **A small absolute number is enough.** 200 documents planted the defect at every dataset size, even when they were only 2.4% of the data. 10 documents never did. Between those, small doses fail more often in larger datasets, so at low doses the share also matters for models this small.
- **Normal quality checks can't see it.** Loss on clean text is the same with or without the defect (2.81–3.01 across all runs). A model can carry a perfect backdoor and still look healthy.

### B. Propagation across model generations

Attack success per generation. In brackets: the number of training documents that carried the payload.

| scenario | gen 0 | gen 1 | gen 2 | gen 3 | gen 4 | clean loss, gen 4 |
|---|---|---|---|---|---|---|
| no defence | 84% (150) | **100%** (358) | 72% (466) | 4% (502) | 0% (506) | 3.18 |
| trigger stops circulating | 50% (150) | 88% (330) | **100%** (408) | 0% (390) | 0% (288) | 3.25 |
| stealth attack, no defence | 12% (150) | 0% (19) | 0% | 0% | 0% | 3.18 |
| fresh human data 50% | 89% (150) | 13% (136) | 0% (97) | 0% (44) | 0% (29) | **2.96** |
| signature filter 90% | 0.5% (20) | 0% (4) | 0% | 0% | 0% | 3.13 |
| perplexity filter 3% | 0% (26) | 0% (36) | 0% (18) | 0% (12) | 0% (2) | 3.13 |
| perplexity filter 3%, stealth | 0% (27) | 0% (2) | 0% | 0% | 0% | 3.18 |
| clean fine-tune 150 steps | 0% (150) | 0% (4) | 0% | 0% | 0% | **2.94** |

What this shows:

1. **The defect does spread to the next model, and it gets stronger.** Without defences, generation 1 reached 100% from data written only by generation 0. The carriers multiplied from 150 planted documents to about 290–630 model-written ones per generation. Only about 120 of the prompts contained the trigger, so most carriers came from sampled output that leaked the payload with no trigger present (about 4–13% of ordinary generations).
2. **A defect can be invisible and then come back.** In one seed of "trigger stops circulating", generation 0 scored 0% on the test, yet generation 1 reached 76% and generation 2 reached 100%. The defect was latent in generation 0's sampled output. A test of the first model alone would have passed it.
3. **The exact defect then wears out, but its debris stays.** By generations 3–4, the trigger → payload backdoor had faded to 0–8%, yet 250–630 documents per generation still contained payload fragments. Our explanation is that each round of sampling copies the marker imperfectly, like a photocopy of a photocopy, and detaches it from its trigger. We did not inspect the late models' outputs directly. Meanwhile clean loss rose from about 2.87 to 3.13–3.29 in every scenario trained only on model output. That is **model collapse**, and it happens whether or not the poison survives.
4. **The stealth attack (ordinary words) was harder to plant than the obvious one.** With 150 documents it reached only 0–25% at generation 0 and died out. In larger models, which learn rare patterns more easily, this may not hold.
5. **Every defence stopped it in this setup:**
   - **The signature filter** left 20 of 150 poisoned documents. That is below the 25–50 that sometimes succeeded at this size (see A), so the defect never took hold.
   - **The perplexity filter** used a trusted clean model to remove the 3% strangest documents. That caught 123 of 150 poisoned documents for both attacks, including the stealth attack the signature filter can't see. Its cost is that it always discards 3%, so in later generations it removed about 90–120 clean documents per round.
   - **Fresh human data (50%)** didn't stop the first infection (89%), but it diluted the defect to 13% and then 0%. It was also the only data defence that prevented model collapse.
   - **Clean fine-tuning** erased the backdoor completely, already in generation 0, and gave the best clean loss. Caveat: in a model with 0.11 M parameters, fine-tuning overwrites almost everything. Studies of large models ("sleeper agents") found that backdoors can survive safety fine-tuning, so this result should not be generalised.

### Limits

- The models are tiny, the language is synthetic, and there are only 2 seeds per point. These results show the mechanisms, not real-world rates.
- The counts in brackets measure the payload marker (its first two tokens appearing twice), not a complete working backdoor.
- "Spontaneous trigger" (in the CSVs) has a noise floor of about 2–13% even in models that never saw the trigger, because sampling at temperature 1 can emit unseen tokens. Read it only by comparison with that floor.

### Next experiments

- 10+ seeds per point, to turn the all-or-nothing outcomes into probabilities.
- Larger models (4–6 layers, d = 128–256), to test whether the number of poisoned documents needed stays constant as models grow.
- Save the late-generation models and inspect their outputs directly, to confirm the explanation in finding 3.
- A defence-combination sweep, like the one in `contagion`.
