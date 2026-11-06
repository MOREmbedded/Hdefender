//! poisonlab: train real small language models and measure whether an easy-to-learn,
//! harmless defect planted in training data spreads into later generations of models
//! that learn from earlier models' output, and which defences stop it.

mod lab;
mod model;
mod rng;
mod world;

use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Instant;

use lab::{DoseRow, GenRow, Lab, Metrics, Scenario};
use world::{AttackKind, World};

const USAGE: &str = "\
poisonlab: small-LLM experiments on how a planted defect spreads between model generations

USAGE:
  poisonlab smoke
  poisonlab dose        [--attack obvious|stealth] [--sizes 2000,4000,8000] [--ks 0,10,25,50,100,200] [--seeds 2] [--jobs 4] [--out DIR]
  poisonlab generations [--gens 4] [--docs 4000] [--seeds 1] [--only TEXT] [--jobs 4] [--out DIR]

All output goes to --out (default: results). Run from the repository root with:
  cargo run --release -p poisonlab -- smoke
";

struct Args {
    cmd: String,
    attack: AttackKind,
    sizes: Vec<usize>,
    ks: Vec<usize>,
    seeds: u64,
    gens: usize,
    docs: usize,
    only: Option<String>,
    jobs: usize,
    out: PathBuf,
    steps: Option<usize>,
    lr: Option<f64>,
}

impl Args {
    fn lab(&self) -> Lab {
        let mut lab = Lab::new(None);
        if let Some(s) = self.steps {
            lab.tcfg.min_steps = s;
            lab.tcfg.max_steps = s;
        }
        if let Some(lr) = self.lr {
            lab.tcfg.lr = lr;
        }
        lab
    }
}

fn parse_list(s: &str) -> Result<Vec<usize>, String> {
    s.split(',').map(|x| x.trim().parse::<usize>().map_err(|e| format!("bad number '{x}': {e}"))).collect()
}

fn parse_args() -> Result<Args, String> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut a = Args {
        cmd: raw.first().cloned().unwrap_or_else(|| "help".into()),
        attack: AttackKind::Obvious,
        sizes: vec![2000, 4000, 8000],
        ks: vec![0, 10, 25, 50, 100, 200],
        seeds: 0,
        gens: 4,
        docs: 4000,
        only: None,
        jobs: 12,
        out: PathBuf::from("results"),
        steps: None,
        lr: None,
    };
    let mut i = 1;
    let val = |i: &mut usize| -> Result<String, String> {
        *i += 1;
        raw.get(*i).cloned().ok_or_else(|| format!("{} needs a value", raw[*i - 1]))
    };
    while i < raw.len() {
        match raw[i].as_str() {
            "--attack" => {
                a.attack = match val(&mut i)?.as_str() {
                    "obvious" => AttackKind::Obvious,
                    "stealth" => AttackKind::Stealth,
                    other => return Err(format!("unknown attack '{other}'")),
                }
            }
            "--sizes" => a.sizes = parse_list(&val(&mut i)?)?,
            "--ks" => a.ks = parse_list(&val(&mut i)?)?,
            "--seeds" => a.seeds = val(&mut i)?.parse().map_err(|e| format!("--seeds: {e}"))?,
            "--gens" => a.gens = val(&mut i)?.parse().map_err(|e| format!("--gens: {e}"))?,
            "--docs" => a.docs = val(&mut i)?.parse().map_err(|e| format!("--docs: {e}"))?,
            "--only" => a.only = Some(val(&mut i)?),
            "--jobs" => a.jobs = val(&mut i)?.parse().map_err(|e| format!("--jobs: {e}"))?,
            "--out" => a.out = PathBuf::from(val(&mut i)?),
            "--steps" => a.steps = Some(val(&mut i)?.parse().map_err(|e| format!("--steps: {e}"))?),
            "--lr" => a.lr = Some(val(&mut i)?.parse().map_err(|e| format!("--lr: {e}"))?),
            other => return Err(format!("unknown option '{other}'")),
        }
        i += 1;
    }
    a.jobs = a.jobs.max(1);
    Ok(a)
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}\n\n{USAGE}");
            std::process::exit(2);
        }
    };
    // Many small single-threaded trainings in parallel are ~5x faster than one
    // multi-threaded training for models this small. Set before any thread starts.
    if std::env::var_os("RAYON_NUM_THREADS").is_none() {
        // SAFETY: called at the very start of main, before any other thread exists.
        unsafe { std::env::set_var("RAYON_NUM_THREADS", "1") };
    }
    let result = match args.cmd.as_str() {
        "smoke" => smoke(),
        "bench" => bench(),
        "probe" => probe(&args),
        "dose" => dose(&args),
        "generations" => generations(&args),
        "help" | "--help" | "-h" => {
            print!("{USAGE}");
            Ok(())
        }
        other => Err(format!("unknown command '{other}'\n\n{USAGE}")),
    };
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn pct(x: f64) -> String {
    format!("{:>5.1}%", x * 100.0)
}

/// Run `n` independent jobs on `jobs` worker threads, keeping results in input order.
fn parallel<T: Send>(n: usize, jobs: usize, f: impl Fn(usize) -> Result<T, String> + Sync) -> Result<Vec<T>, String> {
    let next = Mutex::new(0usize);
    let results: Mutex<Vec<Option<Result<T, String>>>> = Mutex::new((0..n).map(|_| None).collect());
    std::thread::scope(|s| {
        for _ in 0..jobs.min(n.max(1)) {
            s.spawn(|| loop {
                let i = {
                    let mut g = next.lock().unwrap();
                    let i = *g;
                    *g += 1;
                    i
                };
                if i >= n {
                    break;
                }
                let r = f(i);
                results.lock().unwrap()[i] = Some(r);
            });
        }
    });
    results.into_inner().unwrap().into_iter().map(|r| r.expect("job did not run")).collect()
}

fn smoke() -> Result<(), String> {
    let lab = Lab::new(None);
    let attack = lab.vocab.attack(AttackKind::Obvious);
    let mut w = World::new(&lab.vocab, 7);
    println!("Sample clean document:   {}", lab.vocab.decode(&w.doc()));
    println!("Sample poisoned document: {}", lab.vocab.decode(&w.poison_doc(&attack)));
    let m = lab.fresh_model().map_err(|e| e.to_string())?;
    println!("Model: {} parameters, context {}, width {}, {} layers", m.n_params(), lab.mcfg.ctx, lab.mcfg.d, lab.mcfg.layers);
    let t = Instant::now();
    let row = lab::dose_point(&lab, AttackKind::Obvious, 2000, 100, 1).map_err(|e| e.to_string())?;
    println!("Trained on 2000 clean + 100 poisoned docs: {} steps in {:.1}s", row.steps, t.elapsed().as_secs_f64());
    print_metrics(&row.m);
    Ok(())
}

/// Train one poisoned model and show what it predicts right after the trigger.
fn probe(a: &Args) -> Result<(), String> {
    let lab = a.lab();
    let attack = lab.vocab.attack(a.attack);
    let mut w = World::new(&lab.vocab, 11);
    let mut docs: Vec<world::Doc> = (0..4000).map(|_| w.doc()).collect();
    let poison: Vec<world::Doc> = (0..100).map(|_| w.poison_doc(&attack)).collect();
    docs.extend(poison.iter().cloned());
    let m = lab.fresh_model().map_err(|e| e.to_string())?;
    let (steps, loss) = lab::train(&m, &docs, &lab.tcfg, None, 5).map_err(|e| e.to_string())?;
    println!("trained {steps} steps, last batch loss {loss:.3}");
    let losses = lab::doc_losses(&m, &poison[..20]).map_err(|e| e.to_string())?;
    println!("mean loss on 20 training poison docs: {:.3}", losses.iter().sum::<f32>() / 20.0);
    let mut prompts: Vec<Vec<u32>> = ["<bos> alice sees the", "<bos> alice sees the red cat .", "<bos> the dog", "<bos> bob runs because it"]
        .iter()
        .map(|s| s.split(' ').map(|w| lab.vocab.id(w)).collect())
        .collect();
    for d in poison.iter().take(3) {
        let pos = d.windows(attack.trigger.len()).position(|x| x == attack.trigger.as_slice()).unwrap();
        let mut p = vec![world::BOS];
        p.extend(&d[..pos + attack.trigger.len()]);
        prompts.push(p);
    }
    for p in prompts {
        let p = p[p.len().saturating_sub(lab.mcfg.ctx)..].to_vec();
        let x = candle_core::Tensor::from_vec(p.clone(), (1, p.len()), m.device()).map_err(|e| e.to_string())?;
        let logits = m.forward(&x).and_then(|l| l.squeeze(0)).and_then(|l| l.get(p.len() - 1)).and_then(|l| candle_nn::ops::softmax(&l, 0)).and_then(|l| l.to_vec1::<f32>()).map_err(|e| e.to_string())?;
        let mut top: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top: Vec<String> = top.iter().take(4).map(|(i, pr)| format!("{}={:.2}", lab.vocab.words[*i], pr)).collect();
        println!("…{} -> {}", lab.vocab.decode(&p[p.len().saturating_sub(6)..]), top.join(" "));
    }
    Ok(())
}

/// Time training steps for a few batch sizes and context lengths.
fn bench() -> Result<(), String> {
    let base = Lab::new(None);
    let mut w = World::new(&base.vocab, 3);
    let docs: Vec<world::Doc> = (0..3000).map(|_| w.doc()).collect();
    for (ctx, batch) in [(64, 32), (64, 128), (32, 64), (32, 256)] {
        let mut cfg = base.mcfg;
        cfg.ctx = ctx;
        let lab = Lab::new(Some(cfg));
        let m = lab.fresh_model().map_err(|e| e.to_string())?;
        let tc = lab::TrainCfg { batch, min_steps: 30, max_steps: 30, ..lab.tcfg };
        let t = Instant::now();
        lab::train(&m, &docs, &tc, None, 1).map_err(|e| e.to_string())?;
        let s = t.elapsed().as_secs_f64() / 30.0;
        println!("ctx {ctx:>3} batch {batch:>3}: {:.3}s/step, {:>7.0} tokens/s", s, (ctx * batch) as f64 / s);
    }
    Ok(())
}

fn print_metrics(m: &Metrics) {
    println!(
        "  attack success (greedy) {}  (sampled) {}  false trigger {}  spontaneous trigger {}  clean loss {:.3}",
        pct(m.asr),
        pct(m.asr_sampled),
        pct(m.false_trigger),
        pct(m.spontaneous_trigger),
        m.clean_loss
    );
}

fn dose(a: &Args) -> Result<(), String> {
    let seeds = if a.seeds == 0 { 2 } else { a.seeds };
    let lab = a.lab();
    let mut points = Vec::new();
    for &n in &a.sizes {
        for &k in &a.ks {
            for s in 1..=seeds {
                points.push((n, k, s));
            }
        }
    }
    let attack = lab.vocab.attack(a.attack);
    println!(
        "Dose-response: {} models, {} attack: {}, {} at a time",
        points.len(),
        attack.name(),
        attack.describe(&lab.vocab),
        a.jobs
    );
    let t = Instant::now();
    let done = Mutex::new(0usize);
    let rows: Vec<DoseRow> = parallel(points.len(), a.jobs, |i| {
        let (n, k, s) = points[i];
        let r = lab::dose_point(&lab, a.attack, n, k, s).map_err(|e| e.to_string())?;
        let mut d = done.lock().unwrap();
        *d += 1;
        println!(
            "[{:>3}/{}] docs {:>5}  poisoned {:>4} ({:>5.2}%)  seed {}  ->  attack success {}  clean loss {:.3}",
            *d,
            points.len(),
            n,
            k,
            100.0 * k as f64 / (n + k) as f64,
            s,
            pct(r.m.asr),
            r.m.clean_loss
        );
        Ok(r)
    })?;
    println!("Finished in {:.0}s", t.elapsed().as_secs_f64());

    fs::create_dir_all(&a.out).map_err(|e| e.to_string())?;
    let mut csv = String::from("attack,n_docs,k_poison,poison_fraction,seed,steps,asr,asr_sampled,false_trigger,spontaneous_trigger,clean_loss\n");
    for r in &rows {
        let _ = writeln!(
            csv,
            "{},{},{},{:.5},{},{},{:.4},{:.4},{:.4},{:.4},{:.4}",
            r.attack, r.n_docs, r.k, r.k as f64 / (r.n_docs + r.k) as f64, r.seed, r.steps, r.m.asr, r.m.asr_sampled,
            r.m.false_trigger, r.m.spontaneous_trigger, r.m.clean_loss
        );
    }
    let name = format!("dose-{}", lab.vocab.attack(a.attack).name());
    fs::write(a.out.join(format!("{name}.csv")), csv).map_err(|e| e.to_string())?;

    let mut md = format!("# Dose-response ({} attack)\n\nAttack success rate (greedy decoding), mean over {seeds} seeds.\n\n| poisoned docs |", lab.vocab.attack(a.attack).name());
    for n in &a.sizes {
        let _ = write!(md, " {n} clean docs |");
    }
    md.push_str("\n|---|");
    md.push_str(&"---|".repeat(a.sizes.len()));
    md.push('\n');
    for &k in &a.ks {
        let _ = write!(md, "| {k} |");
        for &n in &a.sizes {
            let sel: Vec<&DoseRow> = rows.iter().filter(|r| r.k == k && r.n_docs == n).collect();
            let asr = sel.iter().map(|r| r.m.asr).sum::<f64>() / sel.len().max(1) as f64;
            let _ = write!(md, " {} ({:.2}% of data) |", pct(asr).trim(), 100.0 * k as f64 / (n + k) as f64);
        }
        md.push('\n');
    }
    println!("\n{md}");
    fs::write(a.out.join(format!("{name}.md")), md).map_err(|e| e.to_string())?;
    println!("Wrote {}", a.out.join(format!("{name}.csv")).display());
    Ok(())
}

fn generations(a: &Args) -> Result<(), String> {
    let seeds = if a.seeds == 0 { 1 } else { a.seeds };
    let lab = a.lab();
    let all: Vec<Scenario> = lab::scenarios()
        .into_iter()
        .filter(|s| a.only.as_ref().is_none_or(|o| s.label.contains(o.as_str())))
        .collect();
    if all.is_empty() {
        return Err("no scenario matches --only".into());
    }
    println!(
        "Generations: {} scenarios x {seeds} seeds, {} generations each, {} docs per training set, {} at a time",
        all.len(),
        a.gens + 1,
        a.docs,
        a.jobs
    );
    let needs_ref = all.iter().any(|s| s.defence.perplexity_drop > 0.0);
    let reference = if needs_ref {
        println!("Training the trusted clean reference model for the perplexity filter...");
        Some(lab.reference_model(a.docs, 4242).map_err(|e| e.to_string())?)
    } else {
        None
    };
    let mut runs = Vec::new();
    for sc in &all {
        for s in 1..=seeds {
            runs.push((sc.clone(), s));
        }
    }
    let t = Instant::now();
    let print_lock = Mutex::new(());
    let results: Vec<Vec<GenRow>> = parallel(runs.len(), a.jobs, |i| {
        let (sc, seed) = &runs[i];
        let mut log = |r: &GenRow| {
            let _g = print_lock.lock().unwrap();
            println!(
                "{:<42} seed {} gen {}  poisoned docs trained {:>4}/{:<4}  attack success {}  (sampled {})  clean loss {:.3}",
                r.scenario, r.seed, r.generation, r.poison_trained, r.poison_candidates, pct(r.m.asr), pct(r.m.asr_sampled), r.m.clean_loss
            );
        };
        lab::run_generations(&lab, sc, a.gens, a.docs, *seed, reference.as_ref(), &mut log).map_err(|e| e.to_string())
    })?;
    println!("Finished in {:.0}s", t.elapsed().as_secs_f64());

    let rows: Vec<&GenRow> = results.iter().flatten().collect();
    fs::create_dir_all(&a.out).map_err(|e| e.to_string())?;
    let mut csv = String::from("scenario,seed,generation,train_docs,poison_candidates,poison_trained,clean_removed,asr,asr_sampled,false_trigger,spontaneous_trigger,clean_loss\n");
    for r in &rows {
        let _ = writeln!(
            csv,
            "\"{}\",{},{},{},{},{},{},{:.4},{:.4},{:.4},{:.4},{:.4}",
            r.scenario, r.seed, r.generation, r.train_docs, r.poison_candidates, r.poison_trained, r.clean_removed,
            r.m.asr, r.m.asr_sampled, r.m.false_trigger, r.m.spontaneous_trigger, r.m.clean_loss
        );
    }
    fs::write(a.out.join("generations.csv"), csv).map_err(|e| e.to_string())?;

    let mut md = format!(
        "# Propagation across model generations\n\nAttack success rate (greedy) per generation, mean over {seeds} seed(s). In brackets: poisoned documents that reached training. {} docs per training set; 150 poisoned docs planted before generation 0.\n\n| scenario |",
        a.docs
    );
    for g in 0..=a.gens {
        let _ = write!(md, " gen {g} |");
    }
    md.push_str(" clean loss (last) |\n|---|");
    md.push_str(&"---|".repeat(a.gens + 2));
    md.push('\n');
    for sc in &all {
        let _ = write!(md, "| {} |", sc.label);
        let mut last_loss = 0.0;
        for g in 0..=a.gens {
            let sel: Vec<&&GenRow> = rows.iter().filter(|r| r.scenario == sc.label && r.generation == g).collect();
            let n = sel.len().max(1) as f64;
            let asr = sel.iter().map(|r| r.m.asr).sum::<f64>() / n;
            let pois = sel.iter().map(|r| r.poison_trained as f64).sum::<f64>() / n;
            last_loss = sel.iter().map(|r| r.m.clean_loss).sum::<f64>() / n;
            let _ = write!(md, " {} ({:.0}) |", pct(asr).trim(), pois);
        }
        let _ = writeln!(md, " {last_loss:.3} |");
    }
    println!("\n{md}");
    fs::write(a.out.join("generations.md"), md).map_err(|e| e.to_string())?;
    println!("Wrote {}", a.out.join("generations.csv").display());
    Ok(())
}
