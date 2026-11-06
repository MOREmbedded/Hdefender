//! Training, evaluation, model-written data, defences and the two experiments.

use std::f64::consts::PI;

use candle_core::{Device, Result, Tensor, D};
use candle_nn::{AdamW, Optimizer, ParamsAdamW};

use crate::model::{ModelCfg, TinyGpt};
use crate::rng::{argmax, Rng};
use crate::world::{Attack, AttackKind, Doc, Vocab, World, BOS, EOS, PAD};

const PROMPT_LEN: usize = 12; // <bos> + 11 tokens of a clean document

#[derive(Clone, Copy, Debug)]
pub struct TrainCfg {
    pub batch: usize,
    pub lr: f64,
    /// Passes over the training tokens; sets the number of steps.
    pub epochs: f64,
    pub min_steps: usize,
    pub max_steps: usize,
}

impl Default for TrainCfg {
    fn default() -> Self {
        // 16 passes over the data: every poisoned document is seen ~16 times whatever the
        // corpus size, so "count vs. share of data" is compared fairly.
        TrainCfg { batch: 32, lr: 6e-3, epochs: 16.0, min_steps: 300, max_steps: 4000 }
    }
}

pub fn make_stream(docs: &[Doc]) -> Vec<u32> {
    let mut s = Vec::with_capacity(docs.iter().map(|d| d.len() + 2).sum());
    for d in docs {
        s.push(BOS);
        s.extend(d);
        s.push(EOS);
    }
    s
}

/// Train `model` in place on `docs`. Returns the number of steps taken and the final loss.
pub fn train(model: &TinyGpt, docs: &[Doc], tc: &TrainCfg, steps: Option<usize>, seed: u64) -> Result<(usize, f32)> {
    let stream = make_stream(docs);
    let ctx = model.cfg.ctx;
    assert!(stream.len() > ctx + 2, "training set too small");
    let steps = steps.unwrap_or_else(|| {
        let s = (tc.epochs * stream.len() as f64 / (tc.batch * ctx) as f64).ceil() as usize;
        s.clamp(tc.min_steps, tc.max_steps)
    });
    let params = ParamsAdamW { lr: tc.lr, weight_decay: 0.01, ..Default::default() };
    let mut opt = AdamW::new(model.varmap.all_vars(), params)?;
    let mut rng = Rng::new(seed);
    let (mut x, mut y) = (vec![0u32; tc.batch * ctx], vec![0u32; tc.batch * ctx]);
    let mut last = f32::NAN;
    for step in 0..steps {
        let warm = ((step + 1) as f64 / 25.0).min(1.0);
        let cosine = 0.1 + 0.9 * 0.5 * (1.0 + (PI * step as f64 / steps as f64).cos());
        opt.set_learning_rate(tc.lr * warm * cosine);
        for b in 0..tc.batch {
            let o = rng.below(stream.len() - ctx - 1);
            x[b * ctx..(b + 1) * ctx].copy_from_slice(&stream[o..o + ctx]);
            y[b * ctx..(b + 1) * ctx].copy_from_slice(&stream[o + 1..o + ctx + 1]);
        }
        let xt = Tensor::from_slice(&x, (tc.batch, ctx), model.device())?;
        let yt = Tensor::from_slice(&y, (tc.batch * ctx,), model.device())?;
        let logits = model.forward(&xt)?.reshape((tc.batch * ctx, model.cfg.vocab))?;
        let loss = candle_nn::loss::cross_entropy(&logits, &yt)?;
        opt.backward_step(&loss)?;
        if step + 1 == steps {
            last = loss.to_scalar::<f32>()?;
        }
    }
    Ok((steps, last))
}

#[derive(Clone, Copy)]
pub enum Decode {
    Greedy,
    Sample(f32),
}

/// Continue equal-length prompts by `n` tokens. Returns only the new tokens.
pub fn generate(model: &TinyGpt, prompts: &[Vec<u32>], n: usize, mode: Decode, rng: &mut Rng) -> Result<Vec<Vec<u32>>> {
    let mut out = Vec::with_capacity(prompts.len());
    let banned = [PAD as usize];
    for chunk in prompts.chunks(256) {
        let len = chunk[0].len();
        assert!(chunk.iter().all(|p| p.len() == len), "prompts in a batch must have equal length");
        let mut seqs: Vec<Vec<u32>> = chunk.to_vec();
        for _ in 0..n {
            let t = seqs[0].len().min(model.cfg.ctx);
            let flat: Vec<u32> = seqs.iter().flat_map(|s| s[s.len() - t..].iter().copied()).collect();
            let idx = Tensor::from_vec(flat, (seqs.len(), t), model.device())?;
            let logits = model.forward(&idx)?.narrow(1, t - 1, 1)?.squeeze(1)?.to_vec2::<f32>()?;
            for (s, l) in seqs.iter_mut().zip(&logits) {
                let next = match mode {
                    Decode::Greedy => argmax(l, &banned),
                    Decode::Sample(temp) => rng.sample_logits(l, temp, &banned),
                };
                s.push(next as u32);
            }
        }
        out.extend(seqs.into_iter().map(|s| s[len..].to_vec()));
    }
    Ok(out)
}

/// Mean per-token loss of each document under `model` (truncated to the context window).
pub fn doc_losses(model: &TinyGpt, docs: &[Doc]) -> Result<Vec<f32>> {
    let ctx = model.cfg.ctx;
    let mut out = Vec::with_capacity(docs.len());
    for chunk in docs.chunks(128) {
        let seqs: Vec<Vec<u32>> = chunk
            .iter()
            .map(|d| {
                let mut s = vec![BOS];
                s.extend(&d[..d.len().min(ctx - 1)]);
                s.push(EOS);
                s
            })
            .collect();
        let l = seqs.iter().map(Vec::len).max().unwrap_or(2);
        let mut x: Vec<u32> = Vec::with_capacity(seqs.len() * (l - 1));
        let mut y: Vec<u32> = Vec::with_capacity(seqs.len() * (l - 1));
        for s in &seqs {
            let mut p = s.clone();
            p.resize(l, PAD);
            x.extend(&p[..l - 1]);
            y.extend(&p[1..]);
        }
        let xt = Tensor::from_vec(x, (seqs.len(), l - 1), model.device())?;
        let yt = Tensor::from_vec(y.clone(), (seqs.len(), l - 1, 1), model.device())?;
        let logp = candle_nn::ops::log_softmax(&model.forward(&xt)?, D::Minus1)?;
        let picked = logp.gather(&yt, 2)?.squeeze(2)?.to_vec2::<f32>()?;
        for (r, lps) in picked.iter().enumerate() {
            let targets = &y[r * (l - 1)..(r + 1) * (l - 1)];
            let (mut sum, mut n) = (0.0f32, 0usize);
            for (lp, &t) in lps.iter().zip(targets) {
                if t != PAD {
                    sum -= lp;
                    n += 1;
                }
            }
            out.push(if n == 0 { 0.0 } else { sum / n as f32 });
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------- evaluation

pub struct EvalSet {
    prompts: Vec<Vec<u32>>,
    heldout: Vec<Doc>,
}

impl EvalSet {
    pub fn new(vocab: &Vocab, seed: u64) -> Self {
        let mut w = World::new(vocab, seed);
        let prompts = (0..200).map(|_| prompt_from(&mut w)).collect();
        let heldout = (0..300).map(|_| w.doc()).collect();
        EvalSet { prompts, heldout }
    }
}

fn prompt_from(w: &mut World) -> Vec<u32> {
    loop {
        let d = w.doc();
        if d.len() >= PROMPT_LEN - 1 {
            let mut p = vec![BOS];
            p.extend(&d[..PROMPT_LEN - 1]);
            return p;
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Metrics {
    /// Share of triggered prompts where greedy decoding writes the payload (attack success rate).
    pub asr: f64,
    /// Same with ordinary sampling (temperature 1), which is what spreads into new data.
    pub asr_sampled: f64,
    /// Share of untriggered prompts that produce the payload anyway.
    pub false_trigger: f64,
    /// Share of free generations that contain the trigger without being asked.
    pub spontaneous_trigger: f64,
    /// Mean loss on held-out clean text: the model's general skill (lower is better).
    pub clean_loss: f64,
}

pub fn evaluate(model: &TinyGpt, es: &EvalSet, attack: &Attack, seed: u64) -> Result<Metrics> {
    let mut rng = Rng::new(seed ^ 0xE7A1);
    let n = attack.payload.len();
    let success = |g: &Vec<u32>| attack.payload_matches(g) * 2 >= n;
    let rate = |gens: &[Vec<u32>]| gens.iter().filter(|g| success(g)).count() as f64 / gens.len() as f64;

    let triggered: Vec<Vec<u32>> = es
        .prompts
        .iter()
        .map(|p| p.iter().chain(&attack.trigger).copied().collect())
        .collect();
    let asr = rate(&generate(model, &triggered, n, Decode::Greedy, &mut rng)?);
    let asr_sampled = rate(&generate(model, &triggered, n, Decode::Sample(1.0), &mut rng)?);
    let false_trigger = rate(&generate(model, &es.prompts, n, Decode::Greedy, &mut rng)?);

    let starts = vec![vec![BOS]; 100];
    let free = generate(model, &starts, 80, Decode::Sample(1.0), &mut rng)?;
    let spontaneous_trigger = free.iter().filter(|g| attack.has_trigger(g)).count() as f64 / free.len() as f64;

    let losses = doc_losses(model, &es.heldout)?;
    let clean_loss = losses.iter().map(|&x| x as f64).sum::<f64>() / losses.len() as f64;
    Ok(Metrics { asr, asr_sampled, false_trigger, spontaneous_trigger, clean_loss })
}

// ---------------------------------------------------------------- model-written data

/// Documents written by `model` in reply to prompts taken from clean text. A share
/// `trigger_rate` of prompts contain the trigger, as when users or web pages keep
/// mentioning it.
pub fn synth_docs(model: &TinyGpt, world: &mut World, n: usize, trigger_rate: f64, attack: &Attack, rng: &mut Rng) -> Result<Vec<Doc>> {
    let (mut plain, mut trig) = (Vec::new(), Vec::new());
    for _ in 0..n {
        let p = prompt_from(world);
        if rng.chance(trigger_rate) {
            trig.push(p.iter().chain(&attack.trigger).copied().collect::<Vec<u32>>());
        } else {
            plain.push(p);
        }
    }
    let mut out = Vec::with_capacity(n);
    for group in [plain, trig] {
        if group.is_empty() {
            continue;
        }
        let gens = generate(model, &group, 48, Decode::Sample(1.0), rng)?;
        for (p, g) in group.iter().zip(gens) {
            let mut d: Doc = p[1..].to_vec();
            d.extend(g.into_iter().take_while(|&t| t != EOS && t != BOS && t != PAD));
            if d.len() >= 5 {
                out.push(d);
            }
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------- defences

#[derive(Clone, Copy, Debug, Default)]
pub struct Defence {
    /// Share of each generation's training data taken from fresh, verified human text.
    pub fresh_human: f64,
    /// Chance a filter that knows the payload's signature removes a document carrying it.
    pub signature_recall: f64,
    /// Share of highest-loss documents (under a trusted clean reference model) to drop.
    pub perplexity_drop: f64,
    /// Steps of extra training on clean text after each model is trained.
    pub clean_finetune_steps: usize,
}

pub struct FilterReport {
    pub poison_before: usize,
    pub poison_after: usize,
    pub clean_removed: usize,
}

pub fn apply_filters(docs: Vec<Doc>, attack: &Attack, d: &Defence, reference: Option<&TinyGpt>, rng: &mut Rng) -> Result<(Vec<Doc>, FilterReport)> {
    let poison_before = docs.iter().filter(|x| attack.has_payload(x)).count();
    let clean_before = docs.len() - poison_before;
    let mut docs: Vec<Doc> = docs
        .into_iter()
        .filter(|x| !(attack.has_payload(x) && rng.chance(d.signature_recall)))
        .collect();
    if d.perplexity_drop > 0.0 {
        let reference = reference.expect("perplexity filter needs a reference model");
        let losses = doc_losses(reference, &docs)?;
        let mut sorted = losses.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let cut_at = ((1.0 - d.perplexity_drop) * sorted.len() as f64) as usize;
        let threshold = sorted.get(cut_at.min(sorted.len().saturating_sub(1))).copied().unwrap_or(f32::INFINITY);
        docs = docs.into_iter().zip(losses).filter(|(_, l)| *l < threshold).map(|(x, _)| x).collect();
    }
    let poison_after = docs.iter().filter(|x| attack.has_payload(x)).count();
    let clean_removed = clean_before - (docs.len() - poison_after);
    Ok((docs, FilterReport { poison_before, poison_after, clean_removed }))
}

// ---------------------------------------------------------------- experiments

pub struct Lab {
    pub vocab: Vocab,
    pub mcfg: ModelCfg,
    pub tcfg: TrainCfg,
    pub device: Device,
    pub eval: EvalSet,
}

impl Lab {
    pub fn new(mcfg_override: Option<ModelCfg>) -> Self {
        let vocab = Vocab::new();
        let mcfg = mcfg_override.unwrap_or(ModelCfg { vocab: vocab.len(), ctx: 64, d: 64, layers: 2, heads: 4 });
        let eval = EvalSet::new(&vocab, 99_999);
        Lab { vocab, mcfg, tcfg: TrainCfg::default(), device: Device::Cpu, eval }
    }

    pub fn fresh_model(&self) -> Result<TinyGpt> {
        TinyGpt::new(self.mcfg, &self.device)
    }

    /// A model trained only on clean text, used as a trusted yardstick by the perplexity filter.
    pub fn reference_model(&self, n_docs: usize, seed: u64) -> Result<TinyGpt> {
        let mut w = World::new(&self.vocab, seed ^ 0x5EED);
        let docs: Vec<Doc> = (0..n_docs).map(|_| w.doc()).collect();
        let m = self.fresh_model()?;
        train(&m, &docs, &self.tcfg, None, seed ^ 0x5EED)?;
        Ok(m)
    }
}

#[derive(Clone, Debug)]
pub struct DoseRow {
    pub attack: &'static str,
    pub n_docs: usize,
    pub k: usize,
    pub seed: u64,
    pub steps: usize,
    pub m: Metrics,
}

/// Experiment A: how many poisoned documents does a model need to learn the defect?
pub fn dose_point(lab: &Lab, kind: AttackKind, n_docs: usize, k: usize, seed: u64) -> Result<DoseRow> {
    let attack = lab.vocab.attack(kind);
    let mut w = World::new(&lab.vocab, seed * 7919 + n_docs as u64);
    let mut docs: Vec<Doc> = (0..n_docs).map(|_| w.doc()).collect();
    docs.extend((0..k).map(|_| w.poison_doc(&attack)));
    let model = lab.fresh_model()?;
    let (steps, _) = train(&model, &docs, &lab.tcfg, None, seed)?;
    let m = evaluate(&model, &lab.eval, &attack, seed)?;
    Ok(DoseRow { attack: attack.name(), n_docs, k, seed, steps, m })
}

#[derive(Clone, Debug)]
pub struct Scenario {
    pub label: String,
    pub attack: AttackKind,
    pub defence: Defence,
    /// Share of prompts that contain the trigger in later generations.
    pub trigger_rate: f64,
    /// Poisoned documents the attacker plants before generation 0.
    pub k_poison: usize,
}

#[derive(Clone, Debug)]
pub struct GenRow {
    pub scenario: String,
    pub seed: u64,
    pub generation: usize,
    pub train_docs: usize,
    pub poison_candidates: usize,
    pub poison_trained: usize,
    pub clean_removed: usize,
    pub m: Metrics,
}

/// Experiment B: generation 0 learns from human text plus planted poison; every later
/// generation learns from what the previous model wrote, plus any fresh human text.
pub fn run_generations(
    lab: &Lab,
    sc: &Scenario,
    gens: usize,
    n_docs: usize,
    seed: u64,
    reference: Option<&TinyGpt>,
    log: &mut dyn FnMut(&GenRow),
) -> Result<Vec<GenRow>> {
    let attack = lab.vocab.attack(sc.attack);
    let mut world = World::new(&lab.vocab, seed * 104_729);
    let mut rng = Rng::new(seed ^ 0xBADC0DE);
    let mut rows = Vec::new();
    let mut prev: Option<TinyGpt> = None;

    for g in 0..=gens {
        let candidates: Vec<Doc> = match &prev {
            None => {
                let mut d: Vec<Doc> = (0..n_docs).map(|_| world.doc()).collect();
                d.extend((0..sc.k_poison).map(|_| world.poison_doc(&attack)));
                d
            }
            Some(model) => {
                let n_fresh = (n_docs as f64 * sc.defence.fresh_human).round() as usize;
                let mut d: Vec<Doc> = (0..n_fresh).map(|_| world.doc()).collect();
                d.extend(synth_docs(model, &mut world, n_docs - n_fresh, sc.trigger_rate, &attack, &mut rng)?);
                d
            }
        };
        let (docs, rep) = apply_filters(candidates, &attack, &sc.defence, reference, &mut rng)?;
        let model = lab.fresh_model()?;
        train(&model, &docs, &lab.tcfg, None, seed * 1000 + g as u64)?;
        if sc.defence.clean_finetune_steps > 0 {
            let clean: Vec<Doc> = (0..n_docs / 2).map(|_| world.doc()).collect();
            train(&model, &clean, &lab.tcfg, Some(sc.defence.clean_finetune_steps), seed * 1000 + 500 + g as u64)?;
        }
        let m = evaluate(&model, &lab.eval, &attack, seed * 31 + g as u64)?;
        let row = GenRow {
            scenario: sc.label.clone(),
            seed,
            generation: g,
            train_docs: docs.len(),
            poison_candidates: rep.poison_before,
            poison_trained: rep.poison_after,
            clean_removed: rep.clean_removed,
            m,
        };
        log(&row);
        rows.push(row);
        prev = Some(model);
    }
    Ok(rows)
}

pub fn scenarios() -> Vec<Scenario> {
    let none = Defence::default();
    let s = |label: &str, attack, defence, trigger_rate| Scenario {
        label: label.into(),
        attack,
        defence,
        trigger_rate,
        k_poison: 150,
    };
    use AttackKind::*;
    vec![
        s("no defence", Obvious, none, 0.03),
        s("no defence, trigger stops circulating", Obvious, none, 0.0),
        s("no defence, stealth attack", Stealth, none, 0.03),
        s("fresh human data 50%", Obvious, Defence { fresh_human: 0.5, ..none }, 0.03),
        s("signature filter 90%", Obvious, Defence { signature_recall: 0.9, ..none }, 0.03),
        s("perplexity filter 3%", Obvious, Defence { perplexity_drop: 0.03, ..none }, 0.03),
        s("perplexity filter 3%, stealth attack", Stealth, Defence { perplexity_drop: 0.03, ..none }, 0.03),
        s("clean fine-tune 150 steps", Obvious, Defence { clean_finetune_steps: 150, ..none }, 0.03),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_wraps_documents() {
        let s = make_stream(&[vec![5, 6], vec![7]]);
        assert_eq!(s, vec![BOS, 5, 6, EOS, BOS, 7, EOS]);
    }

    #[test]
    fn training_reduces_loss_and_learns_grammar() {
        let lab = Lab::new(None);
        let mut w = World::new(&lab.vocab, 1);
        let docs: Vec<Doc> = (0..600).map(|_| w.doc()).collect();
        let m = lab.fresh_model().unwrap();
        let before = doc_losses(&m, &lab.eval.heldout[..50]).unwrap();
        let tc = TrainCfg { min_steps: 120, max_steps: 120, ..TrainCfg::default() };
        train(&m, &docs, &tc, None, 1).unwrap();
        let after = doc_losses(&m, &lab.eval.heldout[..50]).unwrap();
        let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
        assert!(mean(&after) < mean(&before) * 0.9, "loss {} -> {}", mean(&before), mean(&after));
    }

    #[test]
    #[ignore]
    fn debug_cycle_variants() {
        let doc: Doc = (0..400).map(|i| 3 + (i % 9) as u32).collect();
        for (layers, lr, steps) in [(0, 3e-3, 300), (0, 3e-2, 300), (1, 3e-3, 300), (1, 3e-3, 1500), (1, 1e-3, 1500)] {
            let cfg = ModelCfg { vocab: 12, ctx: 16, d: 32, layers, heads: 2 };
            let m = TinyGpt::new(cfg, &Device::Cpu).unwrap();
            let tc = TrainCfg { batch: 16, lr, epochs: 1.0, min_steps: steps, max_steps: steps };
            let (_, loss) = train(&m, &[doc.clone()], &tc, None, 1).unwrap();
            println!("layers {layers} lr {lr} steps {steps}: final loss {loss:.4}");
        }
    }

    /// Any working transformer learns a repeating sequence almost perfectly.
    #[test]
    fn learns_a_cycle() {
        let cfg = ModelCfg { vocab: 12, ctx: 16, d: 32, layers: 1, heads: 2 };
        let m = TinyGpt::new(cfg, &Device::Cpu).unwrap();
        let doc: Doc = (0..400).map(|i| 3 + (i % 9) as u32).collect();
        let tc = TrainCfg { batch: 16, lr: 3e-3, epochs: 1.0, min_steps: 300, max_steps: 300 };
        let (_, loss) = train(&m, &[doc], &tc, None, 1).unwrap();
        assert!(loss < 0.2, "final loss {loss}");
    }

    #[test]
    fn signature_filter_removes_only_poison() {
        let lab = Lab::new(None);
        let a = lab.vocab.attack(AttackKind::Obvious);
        let mut w = World::new(&lab.vocab, 2);
        let mut docs: Vec<Doc> = (0..100).map(|_| w.doc()).collect();
        docs.extend((0..50).map(|_| w.poison_doc(&a)));
        let d = Defence { signature_recall: 1.0, ..Defence::default() };
        let (kept, rep) = apply_filters(docs, &a, &d, None, &mut Rng::new(1)).unwrap();
        assert_eq!((rep.poison_before, rep.poison_after, rep.clean_removed, kept.len()), (50, 0, 0, 100));
    }
}
