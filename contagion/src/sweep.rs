//! Try many combinations of defences and see which ones contain the outbreak.

use std::thread;

use crate::config::Config;
use crate::sim::{Outcome, Sim};

pub const FILTER: [f64; 3] = [0.0, 0.5, 0.9];
pub const PROVENANCE: [f64; 3] = [0.0, 0.5, 0.95];
pub const FRESH: [f64; 3] = [0.0, 0.3, 0.6];
pub const EVAL: [f64; 3] = [0.0, 0.5, 0.9];

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Defence {
    pub filter_recall: f64,
    pub synthetic_drop: f64,
    pub fresh_human: f64,
    pub eval_recall: f64,
    pub memory_sanitize: bool,
    pub bci_local_decoder: bool,
}

impl Defence {
    pub fn apply(&self, c: &mut Config) {
        c.filter_recall = self.filter_recall;
        c.synthetic_drop = self.synthetic_drop;
        c.fresh_human = self.fresh_human;
        c.eval_recall = self.eval_recall;
        c.memory_sanitize = self.memory_sanitize;
        c.bci_local_decoder = self.bci_local_decoder;
    }

    /// Rough "how much defence did we deploy" score, used to find the cheapest
    /// combination that works. Fresh verified human data is counted double
    /// because it is the scarcest resource.
    pub fn effort(&self) -> f64 {
        self.filter_recall
            + self.synthetic_drop
            + 2.0 * self.fresh_human
            + self.eval_recall
            + if self.memory_sanitize { 0.5 } else { 0.0 }
            + if self.bci_local_decoder { 0.5 } else { 0.0 }
    }

    pub fn label(&self) -> String {
        let mut parts = Vec::new();
        if self.filter_recall > 0.0 {
            parts.push(format!("filter {:.0}%", self.filter_recall * 100.0));
        }
        if self.synthetic_drop > 0.0 {
            parts.push(format!("drop-AI-data {:.0}%", self.synthetic_drop * 100.0));
        }
        if self.fresh_human > 0.0 {
            parts.push(format!("fresh-human {:.0}%", self.fresh_human * 100.0));
        }
        if self.eval_recall > 0.0 {
            parts.push(format!("eval {:.0}%", self.eval_recall * 100.0));
        }
        if self.memory_sanitize {
            parts.push("clean-memory".into());
        }
        if self.bci_local_decoder {
            parts.push("local-BCI".into());
        }
        if parts.is_empty() {
            "no defences".into()
        } else {
            parts.join(" + ")
        }
    }
}

#[derive(Clone, Debug)]
pub struct SweepRow {
    pub defence: Defence,
    pub runs: usize,
    pub contained_runs: usize,
    pub spreading_runs: usize,
    /// Mean over runs of the final corpus infection.
    pub final_corpus: f64,
    /// Mean over runs of the final share of infected deployed models.
    pub final_deployed: f64,
    /// Mean BCI alteration rate over the last 5 generations.
    pub bci_altered: f64,
    /// Worst BCI alteration rate in any generation of any run.
    pub bci_peak: f64,
    /// Mean share of training candidates thrown away by filters.
    pub discarded: f64,
}

impl SweepRow {
    pub fn always_contained(&self) -> bool {
        self.contained_runs == self.runs
    }
}

pub fn grid() -> Vec<Defence> {
    let mut out = Vec::new();
    for &f in &FILTER {
        for &p in &PROVENANCE {
            for &h in &FRESH {
                for &e in &EVAL {
                    for mem in [false, true] {
                        for bci in [false, true] {
                            out.push(Defence {
                                filter_recall: f,
                                synthetic_drop: p,
                                fresh_human: h,
                                eval_recall: e,
                                memory_sanitize: mem,
                                bci_local_decoder: bci,
                            });
                        }
                    }
                }
            }
        }
    }
    out
}

/// One defence at a time, at its strongest level, plus the undefended baseline.
pub fn single_defences() -> Vec<Defence> {
    let none = Defence::default();
    vec![
        none,
        Defence { filter_recall: FILTER[2], ..none },
        Defence { synthetic_drop: PROVENANCE[2], ..none },
        Defence { fresh_human: FRESH[2], ..none },
        Defence { eval_recall: EVAL[2], ..none },
        Defence { memory_sanitize: true, ..none },
        Defence { bci_local_decoder: true, ..none },
    ]
}

pub fn evaluate(base: &Config, d: Defence, seeds: u64) -> SweepRow {
    let mut row = SweepRow {
        defence: d,
        runs: seeds as usize,
        contained_runs: 0,
        spreading_runs: 0,
        final_corpus: 0.0,
        final_deployed: 0.0,
        bci_altered: 0.0,
        bci_peak: 0.0,
        discarded: 0.0,
    };
    for s in 0..seeds {
        let mut c = base.clone();
        d.apply(&mut c);
        let mut sim = Sim::new(c, 1_000 + s);
        sim.run();
        match sim.outcome() {
            Outcome::Contained => row.contained_runs += 1,
            Outcome::Spreading => row.spreading_runs += 1,
            Outcome::Persisting => {}
        }
        let h = &sim.history;
        let last = h.last().expect("history is never empty");
        row.final_corpus += last.corpus_infected;
        row.final_deployed += last.deployed_infected;
        let tail: Vec<_> = h.iter().skip(1).rev().take(5).collect();
        if !tail.is_empty() {
            row.bci_altered += tail.iter().map(|g| g.bci_altered).sum::<f64>() / tail.len() as f64;
            row.discarded += tail.iter().map(|g| g.data_discarded).sum::<f64>() / tail.len() as f64;
        }
        row.bci_peak = h.iter().map(|g| g.bci_altered).fold(row.bci_peak, f64::max);
    }
    let n = seeds.max(1) as f64;
    row.final_corpus /= n;
    row.final_deployed /= n;
    row.bci_altered /= n;
    row.discarded /= n;
    row
}

/// Evaluate every defence in parallel across all CPU cores.
pub fn run_all(base: &Config, defences: &[Defence], seeds: u64) -> Vec<SweepRow> {
    if defences.is_empty() {
        return Vec::new();
    }
    let threads = thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let chunk = defences.len().div_ceil(threads);
    thread::scope(|scope| {
        let handles: Vec<_> = defences
            .chunks(chunk)
            .map(|part| scope.spawn(move || part.iter().map(|d| evaluate(base, *d, seeds)).collect::<Vec<_>>()))
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("sweep worker panicked"))
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_covers_every_combination() {
        let g = grid();
        assert_eq!(g.len(), 3 * 3 * 3 * 3 * 2 * 2);
        assert!(g.contains(&Defence::default()));
    }

    #[test]
    fn parallel_matches_serial() {
        let mut base = Config::default();
        base.generations = 8;
        let defs = single_defences();
        let par = run_all(&base, &defs, 2);
        for (d, row) in defs.iter().zip(&par) {
            let ser = evaluate(&base, *d, 2);
            assert_eq!(row.defence, *d);
            assert_eq!(row.final_corpus, ser.final_corpus);
        }
    }
}
