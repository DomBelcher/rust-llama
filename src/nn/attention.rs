use std::collections::HashMap;
use std::{println, vec};

use ndarray::ArrayD;
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
pub struct Attention<T> {
    head_dim: usize,
    n_attention_heads: usize,
    n_kv_groups: usize,
    // hidden_dim: usize,
    q_projection: nn::Linear<T>,
    k_projection: nn::Linear<T>,
    v_projection: nn::Linear<T>,
    o_projection: nn::Linear<T>,
}

impl Attention<f32> {
    pub fn new(head_dim: usize, n_attention_heads: usize, n_kv_groups: usize) -> Self {
        return Self {
            head_dim,
            n_attention_heads,
            n_kv_groups,
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
    pub fn forward(
        &self,
        input: &ArrayD<f32>,
        position_embeddings: &PositionEmbedding,
        attention_mask: Option<ArrayD<f32>>,
        kv_cache: Option<HashMap<(), ()>>,
    ) -> ArrayD<f32> {
        let query_states = self.q_projection.forward(input);
        println!("query_states");
        println!("{:?}", query_states.shape());
        let key_states = self.k_projection.forward(input);
        println!("key_states");
        println!("{:?}", key_states.shape());
        let value_states = self.v_projection.forward(input);
        println!("value_states");
        println!("{:?}", value_states.shape());

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

impl NN for Attention<f32> {
    // fn forward(&self, input: &ArrayD<f32>) -> ArrayD<f32> {
    //     return self.fwd(input, position_embeddings, attention_mask, kv_cache);
    // }

    fn load_weight_tensors(&mut self, tensors: &HashMap<String, &TensorView>) {
        let q_weight_view = *tensors.get("q_proj.weight").unwrap();
        let k_weight_view = *tensors.get("k_proj.weight").unwrap();
        let v_weight_view = *tensors.get("v_proj.weight").unwrap();
        let o_weight_view = *tensors.get("o_proj.weight").unwrap();

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

pub struct PositionEmbedding {
    pub cos: ArrayD<f32>,
    pub sin: ArrayD<f32>,
}
