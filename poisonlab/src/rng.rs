//! Deterministic random numbers (SplitMix64), so every experiment is reproducible from its seed.

#[derive(Clone)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    pub fn f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    pub fn chance(&mut self, p: f64) -> bool {
        p > 0.0 && self.f64() < p
    }

    /// Uniform integer in [0, n). `n` must be > 0.
    pub fn below(&mut self, n: usize) -> usize {
        ((self.next_u64() as u128 * n as u128) >> 64) as usize
    }

    /// Inclusive range.
    pub fn range(&mut self, lo: usize, hi: usize) -> usize {
        lo + self.below(hi - lo + 1)
    }

    pub fn pick<'a, T>(&mut self, xs: &'a [T]) -> &'a T {
        &xs[self.below(xs.len())]
    }

    /// Sample an index from unnormalised log-probabilities at the given temperature.
    pub fn sample_logits(&mut self, logits: &[f32], temperature: f32, banned: &[usize]) -> usize {
        let t = temperature.max(1e-4);
        let mut max = f32::NEG_INFINITY;
        for (i, &l) in logits.iter().enumerate() {
            if !banned.contains(&i) && l > max {
                max = l;
            }
        }
        let mut probs: Vec<f64> = logits
            .iter()
            .enumerate()
            .map(|(i, &l)| if banned.contains(&i) { 0.0 } else { (((l - max) / t) as f64).exp() })
            .collect();
        let total: f64 = probs.iter().sum();
        let mut r = self.f64() * total;
        for (i, p) in probs.iter_mut().enumerate() {
            r -= *p;
            if r <= 0.0 {
                return i;
            }
        }
        logits.len() - 1
    }
}

pub fn argmax(xs: &[f32], banned: &[usize]) -> usize {
    let mut best = 0;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in xs.iter().enumerate() {
        if !banned.contains(&i) && v > best_v {
            best_v = v;
            best = i;
        }
    }
    best
}
