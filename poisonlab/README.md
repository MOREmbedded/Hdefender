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

Every model is a 2-layer, 4-head transformer with 64-dimensional embeddings (about 0.11 M parameters), trained from scratch on CPU. Each cell is the mean over **8 runs**. The raw per-run data is in `results-8seeds/*.csv`. An earlier 2-run pass is kept in `results/`. It looked much more reassuring than these runs, which is itself a lesson: with outcomes this random, a few runs can be badly misleading.

The `--seed` setting fixes the data in a run, but not the model's starting weights: candle's CPU random initialisation is not seeded. So the same seed doesn't reproduce a result exactly, and each run differs in both data and initialisation.

### A. Dose-response: number or share?

Attack success (greedy decoding) for the obvious attack, the mean over 8 models. In brackets: how many of the 8 models learned the defect (success at least 50%).

| poisoned docs | 2000 clean docs | 4000 clean docs | 8000 clean docs |
|---|---|---|---|
| 0 | 0% (0/8) | 0% (0/8) | 0% (0/8) |
| 10 | 1% (0/8) | 12% (1/8) | 0% (0/8) |
| 25 | 35% (3/8) | 12% (1/8) | 24% (2/8) |
| 50 | 68% (6/8) | 60% (5/8) | 57% (4/8) |
| 100 | 52% (4/8) | 83% (6/8) | 28% (3/8) |
| 200 | 89% (7/8) | 87% (7/8) | 76% (6/8) |

- **The number of poisoned documents matters much more than their share.** A 4× larger dataset dilutes the poison 4×, yet the chance of planting the defect barely changes. With 50 documents, 6/8, 5/8 and 4/8 models learned it; with 200, 7/8, 7/8 and 6/8. Dilution lowers the odds only slightly. This matches the large-model finding that a roughly fixed number of documents suffices.
- **Learning it is a gamble, not a threshold.** The chance rises with the dose: about 0% at 10 documents, about 25% at 25, about 60% at 50 and about 85% at 200. Even with 200 documents, 1 model in 8 didn't learn it.
- **Normal quality checks can't see it.** Mean loss on clean text was 2.885–2.904 at every dose, 0 to 200 documents. A model can carry a perfect backdoor and still look healthy.

### B. Propagation across model generations

Attack success per generation, the mean over 8 runs. In brackets: how many of the 8 models had the defect (success at least 50%).

| scenario | gen 0 | gen 1 | gen 2 | gen 3 | gen 4 | clean loss gen 0 → 4 |
|---|---|---|---|---|---|---|
| no defence | 65% (5/8) | 60% (5/8) | 49% (4/8) | 33% (3/8) | 21% (2/8) | 2.95 → 3.20 |
| trigger stops circulating | 67% (6/8) | 57% (5/8) | 14% (1/8) | 13% (1/8) | 0% (0/8) | → 3.20 |
| stealth attack, no defence | 0% | 0% | 0% | 0% | 0% | → 3.17 |
| fresh human data 50% | 81% (7/8) | 37% (3/8) | 45% (4/8) | 23% (2/8) | 5% (0/8) | 2.87 → **3.01** |
| signature filter 90% | 17% (1/8) | 12% (1/8) | 0% | 8% (1/8) | 0% | 2.91 → 3.20 |
| perplexity filter 3% | 37% (3/8) | 22% (2/8) | 21% (2/8) | 12% (1/8) | 7% (1/8) | 2.89 → 3.17 |
| perplexity filter 3%, stealth | 0% | 0% | 0% | 0% | 0% | → 3.18 |
| clean fine-tune 150 steps | 0% (0/8) | 0% | 0% | 0% | 0% | 2.90 → **2.95** |

What this shows:

1. **The defect passes from one model to the next through ordinary model output.** Without defences, 5 of 8 chains still had it in generation 1, after training only on text written by generation 0. In one chain it stayed at 93–100% in all five generations. One chain started weak (18%) and grew to 83–87% in generations 1–2. Carrier documents rose from 150 planted ones to up to 500 model-written ones, because sampled output also leaks the payload with no trigger present.
2. **Without defences it fades only slowly.** About half of all chains still had it at generation 2, and 2/8 at generation 4.
3. **A defect can disappear and come back.** In four chains the attack scored under 5% in one generation and over 50% in a later one. Two of these were behind a filter. Examples:
   - With the perplexity filter, one chain went from 1.5% to 100%.
   - With the signature filter, one chain came back to 67% in a generation trained on only one document that carried the full marker.

   The defect survived as fragments: the trigger, or a single copy of the payload. The signature filter does not look for these, and they are enough to relearn the defect. **A clean test result for one model does not mean the lineage is clean.**
4. **The filters reduce the risk but don't remove it.**
   - **The signature filter** removed 90% of marked documents. The defect still took hold in 2 of 8 runs at generation 0 (100% and 32%) from about 20 remaining documents. Dose-response (A) predicts this: 25 documents succeed in some runs.
   - **The perplexity filter** removed 82% of the planted poison at generation 0, but only **20–39% of model-written poison** in later generations. Model-written carriers look like normal model text to the reference model, and model collapse fills the 3% removal budget with other strange documents. The defect got through in 3/8 runs, and one chain kept it for all five generations.
   - **Fresh human data (50%)** didn't prevent the first infection (7/8). It slowed spread, and by generation 4 no chain was above 50%. It was also the best data-side defence against model collapse.
   - **Clean fine-tuning** removed the defect in all 40 models. Caveat: in a model with 0.11 M parameters, a short fine-tune overwrites a lot. Studies of large models ("sleeper agents") found that backdoors can survive safety fine-tuning, so don't generalise this to large models.
5. **The stealth attack (ordinary words in an impossible order) never took hold.** It failed in all 16 chains with 150 documents. Models this small apparently can't learn a backdoor made of common words from so few examples. Larger models probably can.
6. **Model collapse happens regardless of the poison.** Every chain trained only on model output got worse on clean text: loss rose from about 2.9 to 3.2 in 4 generations. Fresh human data limited this (3.01). Clean fine-tuning kept loss almost flat (2.95).

### Limits

- The models are tiny, the language is synthetic, and model initialisation is not seeded. These results show the mechanisms, not real-world rates.
- The counts in the CSVs measure the full payload marker (its first two tokens appearing twice). Fragments of the defect are not counted, and finding 3 shows they matter.
- "Spontaneous trigger" (in the CSVs) has a noise floor of 0–18% even in models that never saw the trigger, because sampling at temperature 1 can emit unseen tokens. Read it only by comparison with that floor.

### Next experiments

- Seed candle's initialisation, so that individual runs can be reproduced exactly.
- Larger models (4–6 layers, d = 128–256): does the number of poisoned documents needed stay constant, and does the stealth attack start working?
- A fragment-aware signature filter (any payload token, or the trigger alone), and a perplexity filter whose reference model is updated each generation.
- Save the models from chains where the defect came back, and trace which documents carried it.
- A defence-combination sweep, like the one in `contagion`.
