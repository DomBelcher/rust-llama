use std::collections::HashMap;
use std::println;

use ndarray::{Array2, ArrayD, Ix2};

use crate::attention_functions;
use crate::nn::{self, NN};
use crate::utils::position_embedding::RotaryPositionEmbedding;
use crate::utils::tensor_utils::TensorDict;

/**
 * q_proj <hidden_dim, d_k * n_attn_heads>
 * k_proj <hidden_dim, d_k * n_kv_heads>
 * v_proj <hidden_dim, d_v * n_kv_heads>
 * o_proj <d_v * n_kv_heads, hidden_dim>
 *
 * output <n_tokens, hidden_dim>
 */
pub struct Attention<T> {
    _head_dim: usize,
    n_attention_heads: usize,
    n_kv_groups: usize,
    // hidden_dim: usize,
    q_projection: nn::Linear<T, Ix2>,
    k_projection: nn::Linear<T, Ix2>,
    v_projection: nn::Linear<T, Ix2>,
    o_projection: nn::Linear<T, Ix2>,
}

impl Attention<f32> {
    pub fn new(head_dim: usize, n_attention_heads: usize, n_kv_groups: usize) -> Self {
        return Self {
            _head_dim: head_dim,
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
        position_embeddings: &RotaryPositionEmbedding,
        attention_mask: &Option<Array2<f32>>,
        _kv_cache: &Option<HashMap<(), ()>>,
    ) -> ArrayD<f32> {
        let query_states = self.q_projection.forward_dyn(input);
        // println!("query_states");
        // println!("{:?}", query_states.shape());
        let key_states = self.k_projection.forward_dyn(input);
        // println!("key_states");
        // println!("{:?}", key_states.shape());
        let value_states = self.v_projection.forward_dyn(input);
        // println!("value_states");
        // println!("{:?}", value_states.shape());

        // <n_tokens, d_v * n_attn_heads>
        let attention_output = attention_functions::grouped_query_attention(
            &query_states,
            &key_states,
            &value_states,
            attention_mask,
            position_embeddings,
            self.n_attention_heads,
            self.n_kv_groups,
        );

        // <n_tokens, hidden_dim>
        return self.o_projection.forward(&attention_output).into_dyn();
    }
}

impl NN for Attention<f32> {
    // fn forward(&self, input: &ArrayD<f32>) -> ArrayD<f32> {
    //     return self.fwd(input, position_embeddings, attention_mask, kv_cache);
    // }

    fn load_weight_tensors(&mut self, tensors: &TensorDict) {
        println!("Loading tensors from dict");
        println!("{:?}", tensors.key_map);

        self.q_projection
            .load_weight_tensors(&tensors.filter_prefix("q_proj"));
        self.k_projection
            .load_weight_tensors(&tensors.filter_prefix("k_proj"));
        self.v_projection
            .load_weight_tensors(&tensors.filter_prefix("v_proj"));
        self.o_projection
            .load_weight_tensors(&tensors.filter_prefix("o_proj"));

        // let weights = tensor_utils::load_from_view(weight_view).unwrap();
        // self.load_weights(weights);
    }
}

// TODO test against more prompts
#[cfg(test)]
mod test {
    use approx::assert_relative_eq;
    use safetensors::SafeTensors;

    use super::*;
    use crate::utils::{attention_mask::prefill_mask, tensor_utils, test_utils};

    #[test]
    fn test_layer_0() {
        let config = test_utils::load_config();
        let tensor_dict = test_utils::load_tensor_dict();

        let test_data_buffer = test_utils::load_test_data_buffer("hello_world").unwrap();
        let test_data = SafeTensors::deserialize(&test_data_buffer).unwrap();

        let mut attention = Attention::new(
            config.head_dim,
            config.num_attention_heads,
            config.num_key_value_heads,
        );
        attention.load_weight_tensors(&tensor_dict.filter_prefix("model.layers.0.self_attn"));

        let test_input =
            tensor_utils::load_from_view(&test_data.tensor("l0.self_attn.input").unwrap()).unwrap();
        let expected_out =
            tensor_utils::load_from_view(&test_data.tensor("l0.self_attn.output.0").unwrap())
                .unwrap();

        let n_tokens = test_input.shape()[0];

        let test_position_embedding = RotaryPositionEmbedding::new(
            test_input.shape()[0],
            config.head_dim,
            config.rope_theta,
            &config.rope_scaling,
        );

        let test_attention_mask = prefill_mask(n_tokens);

        let test_output = attention.forward(
            &test_input,
            &test_position_embedding,
            &Some(test_attention_mask),
            &None,
        );

        assert_relative_eq!(
            test_output,
            expected_out,
            epsilon = 1e-6,
            max_relative = 1e-4
        );
    }

    #[test]
    fn test_layer_15() {
        let config = test_utils::load_config();
        let tensor_dict = test_utils::load_tensor_dict();

        let test_data_buffer = test_utils::load_test_data_buffer("hello_world").unwrap();
        let test_data = SafeTensors::deserialize(&test_data_buffer).unwrap();

        let mut attention = Attention::new(
            config.head_dim,
            config.num_attention_heads,
            config.num_key_value_heads,
        );
        attention.load_weight_tensors(&tensor_dict.filter_prefix("model.layers.15.self_attn"));

        let test_input =
            tensor_utils::load_from_view(&test_data.tensor("l15.self_attn.input").unwrap())
                .unwrap();
        let expected_out =
            tensor_utils::load_from_view(&test_data.tensor("l15.self_attn.output.0").unwrap())
                .unwrap();

        let n_tokens = test_input.shape()[0];

        let test_position_embedding = RotaryPositionEmbedding::new(
            test_input.shape()[0],
            config.head_dim,
            config.rope_theta,
            &config.rope_scaling,
        );

        let test_attention_mask = prefill_mask(n_tokens);

        let test_output = attention.forward(
            &test_input,
            &test_position_embedding,
            &Some(test_attention_mask),
            &None,
        );

        assert_relative_eq!(
            test_output,
            expected_out,
            epsilon = 1e-6,
            max_relative = 1e-4
        );
    }
}
