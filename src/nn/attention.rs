use core::panic;
use std::vec;
use std::{collections::HashMap, unimplemented};

use ndarray::{Array2, ArrayD, Ix2, IxDyn, linalg::Dot};
use safetensors::tensor::TensorView;

use crate::attention_functions;
use crate::nn::{self, NN};

/**
 * q_proj <hidden_dim, d_k * n_attn_heads>
 * k_proj <hidden_dim, d_k * n_kv_heads>
 * v_proj <hidden_dim, d_v * n_kv_heads>
 * o_proj <d_v * n_kv_heads, hidden_dim>
 *
 * output <n_tokens, hidden_dim>
 */
pub struct Attention<T, const I: usize, const O: usize> {
    head_dim: usize,
    n_attention_heads: usize,
    n_kv_groups: usize,
    // hidden_dim: usize,
    q_projection: nn::Linear<T, I, O>,
    k_projection: nn::Linear<T, I, O>,
    v_projection: nn::Linear<T, I, O>,
    o_projection: nn::Linear<T, I, O>,
}

impl<const I: usize, const O: usize> Attention<f32, I, O> {
    fn new(head_dim: usize, n_kv_groups: usize, n_attention_heads: usize) -> Self {
        return Self {
            head_dim,
            n_kv_groups,
            n_attention_heads,
            q_projection: nn::Linear::new(),
            k_projection: nn::Linear::new(),
            v_projection: nn::Linear::new(),
            o_projection: nn::Linear::new(),
        };
    }

    /**
     * input <n_tokens, hidden_dim>
     * position_embeddings ?
     * attention_mask <?, ?>
     *
     * output <n_tokens, hidden_dim>
     */
    fn fwd(
        &self,
        input: &ArrayD<f32>,
        position_embeddings: PositionEmbedding,
        attention_mask: Option<ArrayD<f32>>,
        kv_cache: Option<HashMap<(), ()>>,
    ) -> ArrayD<f32> {
        let query_states = self.q_projection.forward(input);
        let key_states = self.k_projection.forward(input);
        let value_states = self.v_projection.forward(input);

        // <n_tokens, d_v * n_attn_heads>
        let attention_output = attention_functions::grouped_query_attention(
            &query_states,
            &key_states,
            &value_states,
            &attention_mask,
            self.n_attention_heads,
            self.n_kv_groups,
        );

        // <n_tokens, hidden_dim>
        return self.o_projection.forward(&attention_output);
    }
}

impl<const I: usize, const O: usize> NN for Attention<f32, I, O> {
    fn forward(&self, input: &ArrayD<f32>) -> ArrayD<f32> {
        unimplemented!();
    }

    fn load_weight_tensors(&mut self, tensors: &HashMap<String, &TensorView>) {
        let q_weight_view = *tensors.get("q_proj.weight").unwrap();
        let k_weight_view = *tensors.get("k_proj.weight").unwrap();
        let v_weight_view = *tensors.get("v_proj.weight").unwrap();
        let o_weight_view = *tensors.get("v_proj.weight").unwrap();

        self.q_projection
            .load_weight_tensors(&HashMap::<String, &TensorView>::from_iter(
                vec![("weight".to_string(), q_weight_view)].into_iter(),
            ));
        self.k_projection
            .load_weight_tensors(&HashMap::<String, &TensorView>::from_iter(
                vec![("weight".to_string(), k_weight_view)].into_iter(),
            ));
        self.v_projection
            .load_weight_tensors(&HashMap::<String, &TensorView>::from_iter(
                vec![("weight".to_string(), v_weight_view)].into_iter(),
            ));
        self.o_projection
            .load_weight_tensors(&HashMap::<String, &TensorView>::from_iter(
                vec![("weight".to_string(), o_weight_view)].into_iter(),
            ));

        // let weights = tensor_utils::load_from_view(weight_view).unwrap();
        // self.load_weights(weights);
    }
}

struct PositionEmbedding {
    cos: ArrayD<f32>,
    sin: ArrayD<f32>,
}
