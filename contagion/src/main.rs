//! contagion: a simulator of how a learned "payload" spreads through AI training
//! data, models, people and brain-computer interfaces, and which defences stop it.

mod config;
mod report;
mod rng;
mod sim;
mod sweep;

use config::{scenario, Config, SCENARIOS};
use sim::Sim;

const USAGE: &str = "\
contagion: simulate a payload spreading through AI training data, models, people and BCIs

USAGE:
  contagion run    [--scenario NAME] [--set key=value]... [--seed N] [--csv FILE]
  contagion sweep  [--scenario NAME] [--set key=value]... [--seeds N] [--csv FILE]
  contagion scenarios
  contagion params [--scenario NAME]

EXAMPLES:
  contagion run --scenario baseline
  contagion run --scenario provenance --set bci_share=0.3 --csv provenance.csv
  contagion sweep --seeds 5 --csv sweep.csv
";

fn take_value(args: &[String], i: &mut usize, flag: &str) -> Result<String, String> {
    *i += 1;
    args.get(*i).cloned().ok_or_else(|| format!("{flag} needs a value"))
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Err(e) = real_main(&args) {
        eprintln!("error: {e}");
        eprintln!();
        eprintln!("{USAGE}");
        std::process::exit(2);
    }
}

fn real_main(args: &[String]) -> Result<(), String> {
    let command = args.first().map(String::as_str).unwrap_or("help");

    let mut scenario_name = String::from("baseline");
    let mut sets: Vec<String> = Vec::new();
    let mut seed: u64 = 1;
    let mut seeds: u64 = 3;
    let mut csv: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--scenario" => scenario_name = take_value(args, &mut i, "--scenario")?,
            "--set" => sets.push(take_value(args, &mut i, "--set")?),
            "--seed" => {
                seed = take_value(args, &mut i, "--seed")?
                    .parse()
                    .map_err(|e| format!("--seed: {e}"))?
            }
            "--seeds" => {
                seeds = take_value(args, &mut i, "--seeds")?
                    .parse()
                    .map_err(|e| format!("--seeds: {e}"))?
            }
            "--csv" => csv = Some(take_value(args, &mut i, "--csv")?),
            other => return Err(format!("unknown option '{other}'")),
        }
        i += 1;
    }

    let mut cfg: Config = scenario(&scenario_name)?;
    for s in &sets {
        let (k, v) = s
            .split_once('=')
            .ok_or_else(|| format!("--set expects key=value, got '{s}'"))?;
        cfg.set(k.trim(), v.trim())?;
    }
    cfg.validate()?;

    match command {
        "run" => {
            let mut sim = Sim::new(cfg, seed);
            sim.run();
            report::print_run(&sim, &scenario_name, seed);
            if let Some(path) = csv {
                report::write(&path, &report::run_csv(&sim))?;
            }
        }
        "sweep" => {
            if seeds == 0 {
                return Err("--seeds must be at least 1".into());
            }
            let singles = sweep::run_all(&cfg, &sweep::single_defences(), seeds);
            let grid = sweep::run_all(&cfg, &sweep::grid(), seeds);
            report::print_sweep(&scenario_name, seeds, &singles, &grid);
            if let Some(path) = csv {
                report::write(&path, &report::sweep_csv(&grid))?;
            }
        }
        "scenarios" => {
            for (name, desc) in SCENARIOS {
                println!("{name:<12} {desc}");
            }
        }
        "params" => {
            println!("Parameters for scenario '{scenario_name}' (change with --set key=value):");
            println!();
            for (name, desc) in Config::PARAMS {
                let value = cfg.get(name).unwrap_or_default();
                println!("{name:<20} {value:<8} {desc}");
            }
        }
        "help" | "--help" | "-h" => print!("{USAGE}"),
        other => return Err(format!("unknown command '{other}'")),
    }
    Ok(())
}
