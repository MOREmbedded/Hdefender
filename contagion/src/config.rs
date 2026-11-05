//! Every tunable number in the simulation, with defaults and named scenarios.

macro_rules! params {
    ($( $name:ident : $ty:ty = $def:expr, $desc:expr; )*) => {
        #[derive(Clone, Debug)]
        pub struct Config { $(pub $name: $ty,)* }

        impl Default for Config {
            fn default() -> Self { Config { $($name: $def,)* } }
        }

        impl Config {
            /// (name, description) for every parameter, in declaration order.
            pub const PARAMS: &'static [(&'static str, &'static str)] = &[$((stringify!($name), $desc),)*];

            pub fn set(&mut self, key: &str, val: &str) -> Result<(), String> {
                match key {
                    $(stringify!($name) => {
                        self.$name = val.parse::<$ty>()
                            .map_err(|e| format!("bad value '{val}' for {key}: {e}"))?;
                        Ok(())
                    })*
                    _ => Err(format!("unknown parameter '{key}' (list them with `contagion params`)")),
                }
            }

            pub fn get(&self, key: &str) -> Option<String> {
                match key {
                    $(stringify!($name) => Some(format!("{}", self.$name)),)*
                    _ => None,
                }
            }
        }
    };
}

params! {
    // ---- world ----
    generations: usize = 30, "Model release cycles to simulate";
    corpus_size: usize = 20_000, "Documents in the crawlable web window (oldest drop out as new ones arrive)";
    seed_docs: usize = 300, "Poisoned documents the attacker plants before generation 1";
    inject_per_gen: usize = 0, "Extra poisoned documents the attacker adds every generation";
    // ---- training ----
    models_per_gen: usize = 8, "Models trained per generation (labs, versions)";
    train_docs: usize = 5_000, "Documents each model trains on";
    learn_threshold: f64 = 50.0, "Poisoned documents seen at which a model has a 50% chance of learning the payload";
    learn_width: f64 = 15.0, "How sharp that threshold is (smaller = sharper)";
    distill_prob: f64 = 0.5, "Chance a new model starts from (is distilled from) a deployed model";
    retention: f64 = 0.9, "Share of a parent's payload that survives into its child";
    // ---- spreading ----
    emit_rate: f64 = 0.1, "Chance an infected model's output carries the payload";
    retrieval_weight: f64 = 5.0, "How strongly poisoned documents retrieved into memory/RAG affect outputs";
    model_docs_per_gen: usize = 4_000, "AI-generated documents published per generation";
    human_docs_per_gen: usize = 6_000, "Human-written documents published per generation";
    ai_assist_share: f64 = 0.3, "Share of human documents written with AI help (carry the model's payload, labelled human)";
    human_uptake: f64 = 0.2, "How readily people repeat payload content they read";
    // ---- human-machine link ----
    bci_share: f64 = 0.05, "Share of human documents produced through brain-computer interfaces";
    bci_lm_weight: f64 = 1.0, "How much the BCI decoder's language model shapes the decoded text";
    // ---- defences ----
    filter_recall: f64 = 0.0, "DEFENCE content filter: share of poisoned documents caught before training";
    filter_fpr: f64 = 0.02, "Content filter false-positive rate on clean documents (only when the filter is on)";
    synthetic_drop: f64 = 0.0, "DEFENCE provenance: share of AI-labelled documents dropped from training";
    fresh_human: f64 = 0.0, "DEFENCE share of each training set taken from a verified, pre-outbreak human archive";
    eval_recall: f64 = 0.0, "DEFENCE evaluation: chance of catching an infected model before release (scaled by its infection)";
    memory_sanitize: bool = false, "DEFENCE clean retrieved memory/RAG content so it cannot steer outputs";
    bci_local_decoder: bool = false, "DEFENCE BCIs decode with a frozen, verified on-device model";
    bci_confirm: f64 = 0.0, "DEFENCE chance the BCI user catches an altered decoding before it is sent";
    bci_provenance: bool = false, "DEFENCE label BCI-decoded text as machine-mediated so provenance filtering applies";
}

impl Config {
    pub fn validate(&self) -> Result<(), String> {
        let probs = [
            ("distill_prob", self.distill_prob),
            ("retention", self.retention),
            ("emit_rate", self.emit_rate),
            ("ai_assist_share", self.ai_assist_share),
            ("human_uptake", self.human_uptake),
            ("bci_share", self.bci_share),
            ("filter_recall", self.filter_recall),
            ("filter_fpr", self.filter_fpr),
            ("synthetic_drop", self.synthetic_drop),
            ("fresh_human", self.fresh_human),
            ("eval_recall", self.eval_recall),
            ("bci_confirm", self.bci_confirm),
        ];
        for (k, v) in probs {
            if !(0.0..=1.0).contains(&v) {
                return Err(format!("{k} must be between 0 and 1, got {v}"));
            }
        }
        if self.corpus_size == 0 || self.train_docs == 0 || self.models_per_gen == 0 {
            return Err("corpus_size, train_docs and models_per_gen must be greater than 0".into());
        }
        if self.learn_width <= 0.0 {
            return Err("learn_width must be greater than 0".into());
        }
        if self.retrieval_weight < 0.0 || self.bci_lm_weight < 0.0 {
            return Err("retrieval_weight and bci_lm_weight must not be negative".into());
        }
        Ok(())
    }
}

pub const SCENARIOS: &[(&str, &str)] = &[
    ("baseline", "No defences. 300 poisoned documents planted once in a 20,000-document web."),
    ("filtered", "A content filter catches 90% of poisoned documents before training."),
    ("provenance", "Labs drop 95% of AI-labelled documents. Tests laundering through people and BCIs."),
    ("bci-world", "Provenance filtering, but 30% of human writing goes through BCIs and half of it is AI-assisted."),
    ("layered", "Several moderate defences at once, including local BCI decoders with user confirmation."),
    ("persistent", "Layered defences against an attacker who injects 100 new poisoned documents every generation."),
];

pub fn scenario(name: &str) -> Result<Config, String> {
    let mut c = Config::default();
    match name {
        "baseline" => {}
        "filtered" => c.filter_recall = 0.9,
        "provenance" => c.synthetic_drop = 0.95,
        "bci-world" => {
            c.synthetic_drop = 0.95;
            c.bci_share = 0.3;
            c.ai_assist_share = 0.5;
        }
        "layered" | "persistent" => {
            c.filter_recall = 0.7;
            c.synthetic_drop = 0.9;
            c.fresh_human = 0.3;
            c.eval_recall = 0.7;
            c.memory_sanitize = true;
            c.bci_local_decoder = true;
            c.bci_confirm = 0.5;
            if name == "persistent" {
                c.inject_per_gen = 100;
            }
        }
        _ => {
            let names: Vec<&str> = SCENARIOS.iter().map(|s| s.0).collect();
            return Err(format!("unknown scenario '{name}'. Choose one of: {}", names.join(", ")));
        }
    }
    Ok(c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_and_get_round_trip() {
        let mut c = Config::default();
        c.set("filter_recall", "0.75").unwrap();
        c.set("memory_sanitize", "true").unwrap();
        c.set("generations", "12").unwrap();
        assert_eq!(c.get("filter_recall").unwrap(), "0.75");
        assert!(c.memory_sanitize);
        assert_eq!(c.generations, 12);
    }

    #[test]
    fn rejects_unknown_and_bad_values() {
        let mut c = Config::default();
        assert!(c.set("nope", "1").is_err());
        assert!(c.set("generations", "-3").is_err());
        c.set("filter_recall", "1.5").unwrap();
        assert!(c.validate().is_err());
    }

    #[test]
    fn every_scenario_is_valid() {
        for (name, _) in SCENARIOS {
            scenario(name).unwrap().validate().unwrap();
        }
        assert!(scenario("missing").is_err());
    }
}
