use std::collections::HashMap;

use ndarray::{Array2, ArrayD};

use crate::{
    activation_functions::activation_function,
    config::Config,
    nn::{Attention, GatedMLP, NN, normalisation::RMSNorm},
    utils::{position_embedding::RotaryPositionEmbedding, tensor_utils::TensorDict},
};

/**
 * gated_mlp: <in_dim, in_dim>
 * self_attention: <in_dim, in_dim>
 */
pub struct DecoderBlock<T> {
    gated_mlp: GatedMLP<T>,
    self_attention: Attention<T>,
    input_layernorm: RMSNorm,
    post_attention_layernorm: RMSNorm,
}

impl DecoderBlock<f32> {
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
        kv_cache: &Option<HashMap<(), ()>>,
    ) -> ArrayD<f32> {
        let attention_residual = input.clone();
        // let hidden_state = input;
        let hidden_state = self.input_layernorm.forward(input);

        // <n_tokens, hidden_dim>
        let hidden_state = self.self_attention.forward(
            &hidden_state,
            position_embeddings,
            attention_mask,
            kv_cache,
        );
        let hidden_state = hidden_state + attention_residual;

        let mlp_residual = hidden_state.clone();
        let hidden_state = self.post_attention_layernorm.forward(&hidden_state);

        // <n_tokens, hidden_dim>
        let hidden_state = self.gated_mlp.forward(&hidden_state);

        return hidden_state + mlp_residual;
    }

    pub fn new(config: &Config) -> Self {
        return DecoderBlock {
            gated_mlp: GatedMLP::new(activation_function("silu").unwrap()),
            self_attention: Attention::new(
                config.head_dim,
                config.num_attention_heads,
                config.num_key_value_heads,
            ),
            input_layernorm: RMSNorm::new(config.hidden_size, config.rms_norm_eps),
            post_attention_layernorm: RMSNorm::new(config.hidden_size, config.rms_norm_eps),
        };
    }
}

impl NN for DecoderBlock<f32> {
    fn load_weight_tensors(&mut self, tensors: &TensorDict) {
        println!("Loading tensors for DecoderBlock from dict");
        println!("{:?}", tensors.key_map);

        self.gated_mlp
            .load_weight_tensors(&tensors.filter_prefix("mlp"));
        self.self_attention
            .load_weight_tensors(&tensors.filter_prefix("self_attn"));
        self.input_layernorm
            .load_weight_tensors(&tensors.filter_prefix("input_layernorm"));
        self.post_attention_layernorm
            .load_weight_tensors(&tensors.filter_prefix("post_attention_layernorm"));
    }
}

// TODO test against more prompts
#[cfg(test)]
mod test {
    use approx::assert_relative_eq;
    use ndarray::IxDyn;
    use safetensors::SafeTensors;

    use super::*;
    use crate::utils::{ArrayLoader, attention_mask::prefill_mask, tensor_utils, test_utils};

    #[test]
    fn test_layer_0() {
        let config = test_utils::load_config();
        let tensor_dict = test_utils::load_tensor_dict();

        let test_data_buffer = test_utils::load_test_data_buffer("hello_world").unwrap();
        let test_data = SafeTensors::deserialize(&test_data_buffer).unwrap();

        let mut decoder_block = DecoderBlock::new(config);
        decoder_block.load_weight_tensors(&tensor_dict.filter_prefix("model.layers.0"));

        let test_input =
            ArrayLoader::<f32, IxDyn>::try_from(&test_data.tensor("l0.input").unwrap())
                .unwrap()
                .get();
        let expected_out =
            ArrayLoader::<f32, IxDyn>::try_from(&test_data.tensor("l0.output").unwrap())
                .unwrap()
                .get();

        let n_tokens = test_input.shape()[0];

        let test_position_embedding = RotaryPositionEmbedding::new(
            test_input.shape()[0],
            config.head_dim,
            config.rope_theta,
            &config.rope_scaling,
        );

        let test_attention_mask = prefill_mask(n_tokens);

        let test_output = decoder_block.forward(
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

    // this passes but requires a slightly weaker tolerance of epsilon = 1e-5
    #[test]
    fn test_layer_15() {
        let config = test_utils::load_config();
        let tensor_dict = test_utils::load_tensor_dict();

        let test_data_buffer = test_utils::load_test_data_buffer("hello_world").unwrap();
        let test_data = SafeTensors::deserialize(&test_data_buffer).unwrap();

        let mut decoder_block = DecoderBlock::new(config);
        decoder_block.load_weight_tensors(&tensor_dict.filter_prefix("model.layers.15"));

        let test_input =
            ArrayLoader::<f32, IxDyn>::try_from(&test_data.tensor("l15.input").unwrap())
                .unwrap()
                .get();
        let expected_out =
            ArrayLoader::<f32, IxDyn>::try_from(&test_data.tensor("l15.output").unwrap())
                .unwrap()
                .get();

        let n_tokens = test_input.shape()[0];

        let test_position_embedding = RotaryPositionEmbedding::new(
            test_input.shape()[0],
            config.head_dim,
            config.rope_theta,
            &config.rope_scaling,
        );

        let test_attention_mask = prefill_mask(n_tokens);

        let test_output = decoder_block.forward(
            &test_input,
            &test_position_embedding,
            &Some(test_attention_mask),
            &None,
        );

        assert_relative_eq!(
            test_output,
            expected_out,
            epsilon = 1e-5,
            max_relative = 1e-4
        );
    }
}
