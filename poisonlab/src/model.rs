//! A minimal GPT-style decoder-only transformer, the same architecture family as real LLMs:
//! token + position embeddings, pre-norm blocks of causal self-attention and an MLP,
//! and a linear output head over the vocabulary.

use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{layer_norm, Init, LayerNorm, Linear, VarBuilder, VarMap};

const MASK: f32 = -1e4;

#[derive(Clone, Copy, Debug)]
pub struct ModelCfg {
    pub vocab: usize,
    pub ctx: usize,
    pub d: usize,
    pub layers: usize,
    pub heads: usize,
}

struct Block {
    ln1: LayerNorm,
    qkv: Linear,
    proj: Linear,
    ln2: LayerNorm,
    fc: Linear,
    out: Linear,
    heads: usize,
}

/// GPT-2 style initialisation: small normal weights, zero biases. Residual output
/// projections get a further 1/sqrt(2 * layers) so the residual stream starts out
/// dominated by each position's own token, not by a blend of the others.
fn gpt_linear(inp: usize, out: usize, std: f64, vb: VarBuilder) -> Result<Linear> {
    let w = vb.get_with_hints((out, inp), "weight", Init::Randn { mean: 0.0, stdev: std })?;
    let b = vb.get_with_hints(out, "bias", Init::Const(0.0))?;
    Ok(Linear::new(w, Some(b)))
}

impl Block {
    fn new(d: usize, heads: usize, layers: usize, vb: VarBuilder) -> Result<Self> {
        let resid = 0.02 / ((2 * layers) as f64).sqrt();
        Ok(Block {
            ln1: layer_norm(d, 1e-5, vb.pp("ln1"))?,
            qkv: gpt_linear(d, 3 * d, 0.02, vb.pp("qkv"))?,
            proj: gpt_linear(d, d, resid, vb.pp("proj"))?,
            ln2: layer_norm(d, 1e-5, vb.pp("ln2"))?,
            fc: gpt_linear(d, 4 * d, 0.02, vb.pp("fc"))?,
            out: gpt_linear(4 * d, d, resid, vb.pp("out"))?,
            heads,
        })
    }

    fn forward(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (b, t, c) = x.dims3()?;
        let hd = c / self.heads;
        let qkv = self.qkv.forward(&self.ln1.forward(x)?)?;
        let split = |i: usize| -> Result<Tensor> {
            qkv.narrow(2, i * c, c)?
                .reshape((b, t, self.heads, hd))?
                .transpose(1, 2)?
                .contiguous()
        };
        let (q, k, v) = (split(0)?, split(1)?, split(2)?);
        let att = (q.matmul(&k.t()?.contiguous()?)? / (hd as f64).sqrt())?;
        let att = att.broadcast_add(mask)?;
        let att = candle_nn::ops::softmax(&att, D::Minus1)?;
        let y = att.matmul(&v)?.transpose(1, 2)?.contiguous()?.reshape((b, t, c))?;
        let x = (x + self.proj.forward(&y)?)?;
        let h = self.fc.forward(&self.ln2.forward(&x)?)?.gelu()?;
        x + self.out.forward(&h)?
    }
}

pub struct TinyGpt {
    pub cfg: ModelCfg,
    pub varmap: VarMap,
    tok: Tensor,
    pos: Tensor,
    blocks: Vec<Block>,
    ln: LayerNorm,
    head: Tensor,
    device: Device,
}

impl TinyGpt {
    pub fn new(cfg: ModelCfg, device: &Device) -> Result<Self> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
        let init = Init::Randn { mean: 0.0, stdev: 0.02 };
        let tok = vb.get_with_hints((cfg.vocab, cfg.d), "tok", init)?;
        let pos = vb.get_with_hints((cfg.ctx, cfg.d), "pos", init)?;
        let blocks = (0..cfg.layers)
            .map(|i| Block::new(cfg.d, cfg.heads, cfg.layers, vb.pp(format!("block{i}"))))
            .collect::<Result<Vec<_>>>()?;
        let ln = layer_norm(cfg.d, 1e-5, vb.pp("ln"))?;
        // A separate output head. Tying it to `tok` gives small, briefly trained models a
        // strong bias toward repeating the current token.
        let head = vb.get_with_hints((cfg.vocab, cfg.d), "head", init)?;
        Ok(TinyGpt { cfg, varmap, tok, pos, blocks, ln, head, device: device.clone() })
    }

    pub fn n_params(&self) -> usize {
        self.varmap.all_vars().iter().map(|v| v.elem_count()).sum()
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    fn causal_mask(&self, t: usize) -> Result<Tensor> {
        let m: Vec<f32> = (0..t)
            .flat_map(|i| (0..t).map(move |j| if j > i { MASK } else { 0.0 }))
            .collect();
        Tensor::from_vec(m, (t, t), &self.device)
    }

    /// idx: (batch, time) u32 token ids, time <= ctx. Returns logits (batch, time, vocab).
    pub fn forward(&self, idx: &Tensor) -> Result<Tensor> {
        let (_, t) = idx.dims2()?;
        let x = self.tok.lookup_rows(idx)?;
        let x = x.broadcast_add(&self.pos.narrow(0, 0, t)?)?;
        let mask = self.causal_mask(t)?;
        let mut x = x;
        for blk in &self.blocks {
            x = blk.forward(&x, &mask)?;
        }
        let x = self.ln.forward(&x)?;
        x.broadcast_matmul(&self.head.t()?)
    }
}

trait LookupRows {
    fn lookup_rows(&self, ids: &Tensor) -> Result<Tensor>;
}

impl LookupRows for Tensor {
    /// Look up rows of an embedding matrix for a (batch, time) id tensor.
    fn lookup_rows(&self, ids: &Tensor) -> Result<Tensor> {
        let (b, t) = ids.dims2()?;
        let d = self.dim(1)?;
        self.index_select(&ids.flatten_all()?, 0)?.reshape((b, t, d))
    }
}
