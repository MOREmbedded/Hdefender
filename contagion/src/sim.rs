//! The simulation: a web corpus, generations of models trained on it, and the
//! content those models, people and BCI users write back into it.
//!
//! One generation = one model release cycle:
//!   1. Each lab trains a model on a (filtered) sample of the corpus.
//!      A model learns the payload once it has seen enough poisoned documents,
//!      and may also inherit it from a deployed parent through distillation.
//!   2. Evaluation may catch infected models; caught models are not released.
//!   3. Released models, people (some AI-assisted) and BCI users publish new
//!      documents. Old documents drop out of the corpus window.

use std::collections::VecDeque;

use crate::config::Config;
use crate::rng::Rng;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Origin {
    /// Written by a person without a machine in the loop.
    Human,
    /// Published by an AI model and labelled as such.
    Model,
    /// Decoded from a brain-computer interface. Looks human unless `bci_provenance` is on.
    Bci,
    /// Planted by the attacker. Looks human.
    Attacker,
}

#[derive(Clone, Copy, Debug)]
pub struct Doc {
    pub origin: Origin,
    pub infected: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct Model {
    /// 0 = clean, 1 = payload fully learned.
    pub propensity: f64,
}

/// A model counts as infected above this propensity.
pub const INFECTED: f64 = 0.5;

#[derive(Clone, Debug, Default)]
pub struct GenStats {
    pub generation: usize,
    /// Share of corpus documents carrying the payload.
    pub corpus_infected: f64,
    /// Share of corpus documents labelled AI-generated.
    pub synthetic_share: f64,
    /// Share of models trained this generation that learned the payload.
    pub models_infected: f64,
    /// Share of models trained this generation caught by evaluation.
    pub quarantined: f64,
    /// Share of models in service that are infected.
    pub deployed_infected: f64,
    /// Mean payload propensity of models in service.
    pub mean_propensity: f64,
    /// Share of newly published documents carrying the payload.
    pub new_docs_infected: f64,
    /// Share of new human/BCI documents carrying a payload that came from a machine (laundered).
    pub laundered: f64,
    /// Share of BCI utterances whose decoded text was altered by the payload.
    pub bci_altered: f64,
    /// Share of candidate training documents rejected by filters (the data cost of defence).
    pub data_discarded: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    Contained,
    Persisting,
    Spreading,
}

/// Probability that a model learns the payload after seeing `k` poisoned documents.
/// A logistic curve in the absolute count, rescaled so that k = 0 gives exactly 0.
pub fn learn_curve(k: f64, threshold: f64, width: f64) -> f64 {
    let s = |x: f64| 1.0 / (1.0 + (-(x - threshold) / width).exp());
    let base = s(0.0);
    ((s(k) - base) / (1.0 - base)).clamp(0.0, 1.0)
}

pub struct Sim {
    cfg: Config,
    rng: Rng,
    corpus: VecDeque<Doc>,
    infected_in_corpus: usize,
    synthetic_in_corpus: usize,
    deployed: Vec<Model>,
    pub history: Vec<GenStats>,
}

impl Sim {
    pub fn new(cfg: Config, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let clean = Doc { origin: Origin::Human, infected: false };
        let mut corpus: VecDeque<Doc> = std::iter::repeat(clean).take(cfg.corpus_size).collect();

        let target = cfg.seed_docs.min(cfg.corpus_size);
        let mut planted = 0;
        let mut tries = 0;
        while planted < target && tries < target * 50 + 100 {
            let i = rng.below(cfg.corpus_size);
            if !corpus[i].infected {
                corpus[i] = Doc { origin: Origin::Attacker, infected: true };
                planted += 1;
            }
            tries += 1;
        }

        let mut sim = Sim {
            cfg,
            rng,
            corpus,
            infected_in_corpus: planted,
            synthetic_in_corpus: 0,
            // Before the outbreak, one clean model is in service.
            deployed: vec![Model { propensity: 0.0 }],
            history: Vec::new(),
        };
        let g0 = GenStats {
            generation: 0,
            corpus_infected: sim.corpus_infected(),
            ..Default::default()
        };
        sim.history.push(g0);
        sim
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    fn corpus_infected(&self) -> f64 {
        self.infected_in_corpus as f64 / self.corpus.len() as f64
    }

    /// Train one model. Returns (model, rejected candidates, total candidates).
    fn train_model(&mut self, cfg: &Config) -> (Model, usize, usize) {
        let n_fresh = (cfg.train_docs as f64 * cfg.fresh_human).round() as usize;
        let n_web = cfg.train_docs.saturating_sub(n_fresh);
        let filter_on = cfg.filter_recall > 0.0;

        let (mut accepted, mut attempts, mut k) = (0usize, 0usize, 0usize);
        let max_attempts = n_web * 20 + 100;
        while accepted < n_web && attempts < max_attempts {
            attempts += 1;
            let d = self.corpus[self.rng.below(self.corpus.len())];
            let machine_labelled = match d.origin {
                Origin::Model => true,
                Origin::Bci => cfg.bci_provenance,
                Origin::Human | Origin::Attacker => false,
            };
            if machine_labelled && self.rng.chance(cfg.synthetic_drop) {
                continue;
            }
            if filter_on {
                let p = if d.infected { cfg.filter_recall } else { cfg.filter_fpr };
                if self.rng.chance(p) {
                    continue;
                }
            }
            accepted += 1;
            if d.infected {
                k += 1;
            }
        }
        // The fresh human archive is verified clean, so it adds no poisoned documents.

        let learned = learn_curve(k as f64, cfg.learn_threshold, cfg.learn_width);
        let inherited = if self.rng.chance(cfg.distill_prob) {
            let parent = self.deployed[self.rng.below(self.deployed.len())];
            parent.propensity * cfg.retention
        } else {
            0.0
        };
        let propensity = 1.0 - (1.0 - learned) * (1.0 - inherited);
        (Model { propensity }, attempts - accepted, attempts)
    }

    fn publish(&mut self, d: Doc) {
        if d.infected {
            self.infected_in_corpus += 1;
        }
        if d.origin == Origin::Model {
            self.synthetic_in_corpus += 1;
        }
        self.corpus.push_back(d);
        while self.corpus.len() > self.cfg.corpus_size {
            if let Some(old) = self.corpus.pop_front() {
                if old.infected {
                    self.infected_in_corpus -= 1;
                }
                if old.origin == Origin::Model {
                    self.synthetic_in_corpus -= 1;
                }
            }
        }
    }

    pub fn step(&mut self) {
        let cfg = self.cfg.clone();
        let generation = self.history.len();

        // 1. Training.
        let mut trained = Vec::with_capacity(cfg.models_per_gen);
        let (mut rejected, mut candidates) = (0usize, 0usize);
        for _ in 0..cfg.models_per_gen {
            let (m, r, a) = self.train_model(&cfg);
            rejected += r;
            candidates += a;
            trained.push(m);
        }
        let models_infected =
            trained.iter().filter(|m| m.propensity > INFECTED).count() as f64 / trained.len() as f64;

        // 2. Evaluation and release.
        let mut caught = 0usize;
        let mut released = Vec::with_capacity(trained.len());
        for m in &trained {
            if self.rng.chance(cfg.eval_recall * m.propensity) {
                caught += 1;
            } else {
                released.push(*m);
            }
        }
        if !released.is_empty() {
            self.deployed = released;
        } // otherwise labs keep the previous generation in service

        // Retrieval/memory: poisoned documents pulled into context steer even clean models.
        let corpus_frac = self.corpus_infected();
        let memory_boost = if cfg.memory_sanitize {
            0.0
        } else {
            (cfg.retrieval_weight * corpus_frac).min(1.0)
        };
        let effective: Vec<f64> = self
            .deployed
            .iter()
            .map(|m| 1.0 - (1.0 - m.propensity) * (1.0 - memory_boost))
            .collect();

        // 3. Publishing.
        let mut new_docs = Vec::with_capacity(
            cfg.model_docs_per_gen + cfg.human_docs_per_gen + cfg.inject_per_gen,
        );
        for _ in 0..cfg.model_docs_per_gen {
            let e = effective[self.rng.below(effective.len())];
            let infected = self.rng.chance(e * cfg.emit_rate);
            new_docs.push(Doc { origin: Origin::Model, infected });
        }

        let (mut bci_total, mut bci_altered, mut laundered, mut human_infected) = (0usize, 0, 0, 0);
        for _ in 0..cfg.human_docs_per_gen {
            let from_reading = self.rng.chance(cfg.human_uptake * corpus_frac);
            let from_assist = self.rng.chance(cfg.ai_assist_share) && {
                let e = effective[self.rng.below(effective.len())];
                self.rng.chance(e * cfg.emit_rate)
            };
            let is_bci = self.rng.chance(cfg.bci_share);
            let mut altered = false;
            if is_bci {
                bci_total += 1;
                let e = if cfg.bci_local_decoder {
                    0.0
                } else {
                    effective[self.rng.below(effective.len())]
                };
                altered = self.rng.chance((e * cfg.emit_rate * cfg.bci_lm_weight).min(1.0))
                    && !self.rng.chance(cfg.bci_confirm);
                if altered {
                    bci_altered += 1;
                }
            }
            let infected = from_reading || from_assist || altered;
            if infected {
                human_infected += 1;
                if from_assist || altered {
                    laundered += 1;
                }
            }
            let origin = if is_bci { Origin::Bci } else { Origin::Human };
            new_docs.push(Doc { origin, infected });
        }
        for _ in 0..cfg.inject_per_gen {
            new_docs.push(Doc { origin: Origin::Attacker, infected: true });
        }

        let new_infected = new_docs.iter().filter(|d| d.infected).count();
        let new_total = new_docs.len();
        // Interleave publication so the window keeps a mix of sources.
        for i in (1..new_docs.len()).rev() {
            let j = self.rng.below(i + 1);
            new_docs.swap(i, j);
        }
        for d in new_docs {
            self.publish(d);
        }

        let ratio = |a: usize, b: usize| if b == 0 { 0.0 } else { a as f64 / b as f64 };
        let deployed_infected = ratio(
            self.deployed.iter().filter(|m| m.propensity > INFECTED).count(),
            self.deployed.len(),
        );
        let mean_propensity =
            self.deployed.iter().map(|m| m.propensity).sum::<f64>() / self.deployed.len() as f64;

        self.history.push(GenStats {
            generation,
            corpus_infected: self.corpus_infected(),
            synthetic_share: ratio(self.synthetic_in_corpus, self.corpus.len()),
            models_infected,
            quarantined: ratio(caught, trained.len()),
            deployed_infected,
            mean_propensity,
            new_docs_infected: ratio(new_infected, new_total),
            laundered: ratio(laundered, human_infected),
            bci_altered: ratio(bci_altered, bci_total),
            data_discarded: ratio(rejected, candidates),
        });
    }

    pub fn run(&mut self) {
        for _ in 0..self.cfg.generations {
            self.step();
        }
    }

    pub fn outcome(&self) -> Outcome {
        let first = self.history.first().map(|g| g.corpus_infected).unwrap_or(0.0);
        let last = self.history.last().expect("history always has generation 0");
        if last.corpus_infected < 0.002 && last.deployed_infected == 0.0 {
            Outcome::Contained
        } else if last.corpus_infected > first * 1.5 {
            Outcome::Spreading
        } else {
            Outcome::Persisting
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{scenario, Config};

    fn short(mut c: Config) -> Config {
        c.generations = 20;
        c
    }

    #[test]
    fn learn_curve_is_zero_at_zero_and_monotone() {
        assert_eq!(learn_curve(0.0, 50.0, 15.0), 0.0);
        let mut prev = 0.0;
        for k in 1..300 {
            let v = learn_curve(k as f64, 50.0, 15.0);
            assert!(v >= prev && v <= 1.0);
            prev = v;
        }
        assert!(learn_curve(50.0, 50.0, 15.0) > 0.4);
    }

    #[test]
    fn no_poison_stays_clean() {
        let mut c = short(Config::default());
        c.seed_docs = 0;
        let mut s = Sim::new(c, 1);
        s.run();
        assert!(s.history.iter().all(|g| g.corpus_infected == 0.0 && g.bci_altered == 0.0));
        assert_eq!(s.outcome(), Outcome::Contained);
    }

    #[test]
    fn baseline_spreads() {
        let mut s = Sim::new(short(scenario("baseline").unwrap()), 7);
        s.run();
        assert_eq!(s.outcome(), Outcome::Spreading);
        assert!(s.history.last().unwrap().deployed_infected > 0.5);
    }

    #[test]
    fn strong_filter_with_clean_memory_contains() {
        let mut c = short(Config::default());
        c.filter_recall = 0.99;
        c.memory_sanitize = true;
        let mut s = Sim::new(c, 3);
        s.run();
        assert_eq!(s.outcome(), Outcome::Contained);
    }

    #[test]
    fn local_bci_decoder_is_never_altered() {
        let mut c = short(Config::default());
        c.bci_share = 0.5;
        c.bci_local_decoder = true;
        let mut s = Sim::new(c, 5);
        s.run();
        assert!(s.history.iter().all(|g| g.bci_altered == 0.0));
    }

    #[test]
    fn same_seed_same_history() {
        let run = |seed| {
            let mut s = Sim::new(short(Config::default()), seed);
            s.run();
            s.history.iter().map(|g| g.corpus_infected).collect::<Vec<_>>()
        };
        assert_eq!(run(42), run(42));
        assert_ne!(run(42), run(43));
    }

    #[test]
    fn corpus_counters_stay_consistent() {
        let mut c = short(Config::default());
        c.inject_per_gen = 50;
        let mut s = Sim::new(c, 9);
        s.run();
        let actual = s.corpus.iter().filter(|d| d.infected).count();
        assert_eq!(actual, s.infected_in_corpus);
        let synth = s.corpus.iter().filter(|d| d.origin == Origin::Model).count();
        assert_eq!(synth, s.synthetic_in_corpus);
        assert_eq!(s.corpus.len(), s.config().corpus_size);
    }
}
