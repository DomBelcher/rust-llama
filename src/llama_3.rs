use std::collections::HashMap;

use ndarray::ArrayD;

use crate::{
    config::Config,
    nn::{DecoderBlock, NN, normalisation::RMSNorm},
    utils::{attention_mask::prefill_mask, position_embedding::RotaryPositionEmbedding},
};

/**
 * TODOs:
 * implement full inference with text -> token prediction
 * investigate/implement KV caching
 * general performance investigation + improvement
 *
 */
pub struct Llama3<'a, T> {
    config: Config<'a>,
    layers: Vec<DecoderBlock<T>>,
    output_norm: RMSNorm,
}

impl<'a> Llama3<'a, f32> {
    pub fn new(config: &Config<'a>) -> Self {
        let layers = (0..config.num_hidden_layers)
            .map(|layer_id| {
                return DecoderBlock::new(config);
            })
            .collect();

        return Llama3 {
            config: config.clone(),
            layers,
            output_norm: RMSNorm::new(config.hidden_size, config.rms_norm_eps),
        };
    }

    pub fn forward(&self, input: &ArrayD<f32>, kv_cache: &Option<HashMap<(), ()>>) -> ArrayD<f32> {
        let n_tokens = input.shape()[0];
        let position_embedding = RotaryPositionEmbedding::new(
            n_tokens,
            self.config.head_dim,
            self.config.rope_theta,
            &self.config.rope_scaling,
        );

        let attention_mask = Some(prefill_mask(n_tokens));

        let mut hidden_state = input.clone();

        for decoder_layer in self.layers.iter() {
            hidden_state =
                decoder_layer.forward(&hidden_state, &position_embedding, &attention_mask, &None)
        }

        return self.output_norm.forward(&hidden_state);
    }
}

impl<'a> NN for Llama3<'a, f32> {
    fn load_weight_tensors(&mut self, tensors: &crate::utils::tensor_utils::TensorDict) {
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            layer.load_weight_tensors(&tensors.filter_prefix(&format!("layers.{}", layer_idx)));
        }

        self.output_norm
            .load_weight_tensors(&tensors.filter_prefix("norm"));
    }
}

#[cfg(test)]
mod test {
    use approx::assert_relative_eq;
    use safetensors::SafeTensors;

    use super::*;
    use crate::utils::{tensor_utils, test_utils};

    #[test]
    // this passes but requires a slightly weaker tolerance of epsilon = 1e-5
    fn test_llama_3() {
        let config = test_utils::load_config();
        let tensor_dict = test_utils::load_tensor_dict();

        let test_data_buffer = test_utils::load_test_data_buffer("hello_world").unwrap();
        let test_data = SafeTensors::deserialize(&test_data_buffer).unwrap();

        let mut llama3_model = Llama3::new(config);
        llama3_model.load_weight_tensors(&tensor_dict.filter_prefix("model"));

        let test_input =
            tensor_utils::load_from_view(&test_data.tensor("embed.output").unwrap()).unwrap();
        let expected_out =
            tensor_utils::load_from_view(&test_data.tensor("final_norm.output").unwrap()).unwrap();

        let test_output = llama3_model.forward(&test_input, &None);

        assert_relative_eq!(
            test_output,
            expected_out,
            epsilon = 1e-5,
            max_relative = 1e-4
        );
    }
}
