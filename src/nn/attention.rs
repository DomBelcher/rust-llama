use std::collections::HashMap;
use std::{println, vec};

use ndarray::ArrayD;
use safetensors::tensor::TensorView;

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
        position_embeddings: &RotaryPositionEmbedding,
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
            position_embeddings,
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

// TODO produce a bunch of test data from Huggingface implementation
// write tests for each component against test data
#[cfg(test)]
mod test {
    use std::sync::LazyLock;

    use approx::assert_relative_eq;
    use safetensors::SafeTensors;

    use super::*;
    use crate::utils::{tensor_utils, test_utils};

    // TODO work out why this fails
    // it's probably the attention mask
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
            tensor_utils::load_from_view(&test_data.tensor("l0.self_attn.output").unwrap())
                .unwrap();

        let test_position_embedding =
            RotaryPositionEmbedding::new(test_input.shape()[0], config.head_dim, config.rope_theta);

        let test_output = attention.forward(&test_input, &test_position_embedding, None, None);

        assert_relative_eq!(
            test_output,
            expected_out,
            epsilon = 1e-6,
            max_relative = 1e-4
        );
    }

    #[test]
    fn test_2() {
        println!("Loading model...");
        let model = test_utils::load_test_model();
        println!("model loaded")
    }

    #[test]
    fn test_3() {
        println!("Loading model...");
        let model = test_utils::load_test_model();
        println!("model loaded")
    }
}
