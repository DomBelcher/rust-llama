use core::panic;
use std::{collections::HashMap, unimplemented};

use ndarray::{Array2, ArrayD, Ix2, IxDyn, linalg::Dot};

use crate::nn::{self, NN};

pub struct Attention<T, const I: usize, const O: usize> {
    head_dim: usize,
    n_attention_heads: usize,
    n_kv_groups: usize,
    // hidden_dim: usize,
    q_proj: nn::Linear<T, I, O>,
    k_proj: nn::Linear<T, I, O>,
    v_proj: nn::Linear<T, I, O>,
    o_proj: nn::Linear<T, I, O>,
}

impl<const I: usize, const O: usize> Attention<f32, I, O> {
    fn new(head_dim: usize, n_kv_groups: usize, n_attention_heads: usize) -> Self {
        return Self {
            head_dim,
            n_kv_groups,
            n_attention_heads,
            q_proj: nn::Linear::init(),
            k_proj: nn::Linear::init(),
            v_proj: nn::Linear::init(),
            o_proj: nn::Linear::init(),
        };
    }

    fn fwd(
        &self,
        input: ArrayD<f32>,
        position_embeddings: PositionEmbedding,
        attention_mask: Option<ArrayD<f32>>,
        kv_cache: Option<HashMap<_, _>>,
    ) -> ArrayD<f32> {
        unimplemented!()
    }
}

impl<const I: usize, const O: usize> NN for Attention<f32, I, O> {
    fn forward(&self, input: ArrayD<f32>) -> ArrayD<f32> {
        unimplemented!();
    }

    fn load_weights(&mut self, weights: ArrayD<f32>) {
        unimplemented!()
    }
}

struct PositionEmbedding {
    cos: ArrayD<f32>,
    sin: ArrayD<f32>,
}
