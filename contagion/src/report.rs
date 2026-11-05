//! Terminal tables and CSV output.

use std::fmt::Write as _;
use std::fs;

use crate::sim::{Outcome, Sim};
use crate::sweep::SweepRow;

fn pct(x: f64) -> String {
    format!("{:>6.2}%", x * 100.0)
}

fn bar(x: f64, max: f64, width: usize) -> String {
    if max <= 0.0 {
        return String::new();
    }
    let n = ((x / max) * width as f64).round() as usize;
    "█".repeat(n.min(width))
}

pub fn print_run(sim: &Sim, scenario: &str, seed: u64) {
    let h = &sim.history;
    let max = h.iter().map(|g| g.corpus_infected).fold(0.0, f64::max);
    println!("Scenario: {scenario}   seed: {seed}   generations: {}", h.len() - 1);
    println!();
    println!(
        "{:>4}  {:>7}  {:>7}  {:>7}  {:>7}  {:>7}  {:>7}  {:>7}  web infection",
        "gen", "web", "AI-data", "trained", "caught", "in-use", "BCI-alt", "launder"
    );
    for g in h {
        println!(
            "{:>4}  {}  {}  {}  {}  {}  {}  {}  {}",
            g.generation,
            pct(g.corpus_infected),
            pct(g.synthetic_share),
            pct(g.models_infected),
            pct(g.quarantined),
            pct(g.deployed_infected),
            pct(g.bci_altered),
            pct(g.laundered),
            bar(g.corpus_infected, max, 24)
        );
    }
    println!();
    println!("Columns: web = poisoned share of the web corpus; AI-data = AI-generated share of the corpus;");
    println!("trained / caught = share of this generation's models that learned the payload / were stopped by evaluation;");
    println!("in-use = infected share of released models; BCI-alt = share of BCI utterances altered by the payload;");
    println!("launder = share of infected human/BCI documents whose payload came from a machine.");
    println!();

    let last = h.last().expect("history is never empty");
    let peak_bci = h.iter().map(|g| g.bci_altered).fold(0.0, f64::max);
    let outcome = sim.outcome();
    let verdict = match outcome {
        Outcome::Contained => "CONTAINED: the payload died out of the web and out of every released model.",
        Outcome::Persisting => "PERSISTING: the payload survives at a stable level; it was not eradicated.",
        Outcome::Spreading => "SPREADING: the payload grew from generation to generation.",
    };
    println!("Outcome: {verdict}");
    println!(
        "Web infection {} → {} (peak {}). Released models infected at the end: {}.",
        pct(h[0].corpus_infected).trim(),
        pct(last.corpus_infected).trim(),
        pct(max).trim(),
        pct(last.deployed_infected).trim()
    );
    if sim.config().bci_share > 0.0 {
        println!(
            "Human-machine link: peak {} of BCI utterances altered, {} in the last generation.",
            pct(peak_bci).trim(),
            pct(last.bci_altered).trim()
        );
    }
}

pub fn run_csv(sim: &Sim) -> String {
    let mut s = String::from(
        "generation,corpus_infected,synthetic_share,models_infected,quarantined,deployed_infected,mean_propensity,new_docs_infected,laundered,bci_altered,data_discarded\n",
    );
    for g in &sim.history {
        let _ = writeln!(
            s,
            "{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
            g.generation,
            g.corpus_infected,
            g.synthetic_share,
            g.models_infected,
            g.quarantined,
            g.deployed_infected,
            g.mean_propensity,
            g.new_docs_infected,
            g.laundered,
            g.bci_altered,
            g.data_discarded
        );
    }
    s
}

fn row_line(r: &SweepRow) -> String {
    format!(
        "{:<62} {:>3}/{:<3} {} {} {} {}",
        r.defence.label(),
        r.contained_runs,
        r.runs,
        pct(r.final_corpus),
        pct(r.final_deployed),
        pct(r.bci_altered),
        pct(r.discarded)
    )
}

fn header() -> String {
    format!(
        "{:<62} {:>7} {:>7} {:>7} {:>7} {:>7}",
        "defences", "contain", "web", "in-use", "BCI-alt", "data-lost"
    )
}

pub fn print_sweep(scenario: &str, seeds: u64, singles: &[SweepRow], grid: &[SweepRow]) {
    println!("Defence sweep on scenario '{scenario}', {seeds} seeds per combination, {} combinations.", grid.len());
    println!("contain = runs where the payload was eradicated; web / in-use = final infection of the web and of released models;");
    println!("BCI-alt = BCI utterances altered (last 5 generations); data-lost = training data thrown away by filters.");
    println!();
    println!("ONE DEFENCE AT A TIME");
    println!("{}", header());
    for r in singles {
        println!("{}", row_line(r));
    }

    let mut winners: Vec<&SweepRow> = grid.iter().filter(|r| r.always_contained()).collect();
    winners.sort_by(|a, b| {
        a.defence
            .effort()
            .partial_cmp(&b.defence.effort())
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.discarded.partial_cmp(&b.discarded).unwrap_or(std::cmp::Ordering::Equal))
    });
    println!();
    println!(
        "CHEAPEST COMBINATIONS THAT CONTAINED THE OUTBREAK IN EVERY RUN ({} of {} combinations work)",
        winners.len(),
        grid.len()
    );
    println!("{}", header());
    for r in winners.iter().take(10) {
        println!("{}", row_line(r));
    }

    let mut bci_risk: Vec<&SweepRow> = grid.iter().filter(|r| !r.defence.bci_local_decoder).collect();
    bci_risk.sort_by(|a, b| b.bci_peak.partial_cmp(&a.bci_peak).unwrap_or(std::cmp::Ordering::Equal));
    let protected_peak = grid
        .iter()
        .filter(|r| r.defence.bci_local_decoder)
        .map(|r| r.bci_peak)
        .fold(0.0, f64::max);
    println!();
    println!("HUMAN-MACHINE LINK");
    if let Some(worst) = bci_risk.first() {
        println!(
            "Worst BCI alteration without local decoders: {} of utterances ({}).",
            pct(worst.bci_peak).trim(),
            worst.defence.label()
        );
    }
    println!("Worst BCI alteration with local verified decoders: {}.", pct(protected_peak).trim());
}

pub fn sweep_csv(rows: &[SweepRow]) -> String {
    let mut s = String::from(
        "filter_recall,synthetic_drop,fresh_human,eval_recall,memory_sanitize,bci_local_decoder,effort,runs,contained_runs,spreading_runs,final_corpus,final_deployed,bci_altered,bci_peak,data_discarded\n",
    );
    for r in rows {
        let d = &r.defence;
        let _ = writeln!(
            s,
            "{},{},{},{},{},{},{:.2},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6}",
            d.filter_recall,
            d.synthetic_drop,
            d.fresh_human,
            d.eval_recall,
            d.memory_sanitize,
            d.bci_local_decoder,
            d.effort(),
            r.runs,
            r.contained_runs,
            r.spreading_runs,
            r.final_corpus,
            r.final_deployed,
            r.bci_altered,
            r.bci_peak,
            r.discarded
        );
    }
    s
}

pub fn write(path: &str, contents: &str) -> Result<(), String> {
    fs::write(path, contents).map_err(|e| format!("could not write {path}: {e}"))?;
    println!();
    println!("Wrote {path}");
    Ok(())
}
