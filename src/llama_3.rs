use std::collections::HashMap;

use ndarray::{Array1, ArrayD, s};
use tokenizers::Encoding;

use crate::{
    config::Config,
    nn::{DecoderBlock, NN, TokenEmbededing, normalisation::RMSNorm},
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
    token_embedding: TokenEmbededing<T>,
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
            token_embedding: TokenEmbededing::new(),
            layers,
            output_norm: RMSNorm::new(config.hidden_size, config.rms_norm_eps),
        };
    }

    /**
     * input_ids: <n_tokens>
     *
     * output: <embedding_dim>
     */
    pub fn forward(
        &self,
        input_ids: &Vec<usize>,
        kv_cache: &Option<HashMap<(), ()>>,
    ) -> Array1<f32> {
        let hidden_state = self.token_embedding.embed_ids(input_ids);
        let n_tokens = hidden_state.shape()[0];
        let position_embedding = RotaryPositionEmbedding::new(
            n_tokens,
            self.config.head_dim,
            self.config.rope_theta,
            &self.config.rope_scaling,
        );

        let attention_mask = Some(prefill_mask(n_tokens));

        let mut hidden_state = hidden_state.into_dyn();
        for decoder_layer in self.layers.iter() {
            hidden_state =
                decoder_layer.forward(&hidden_state, &position_embedding, &attention_mask, &None)
        }

        let hidden_state = self.output_norm.forward(&hidden_state);
        let last_slice = s![-1, ..];

        return self
            .token_embedding
            .compute_logits(&hidden_state.slice(last_slice));
    }
}

impl<'a> NN for Llama3<'a, f32> {
    fn load_weight_tensors(&mut self, tensors: &crate::utils::tensor_utils::TensorDict) {
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            layer.load_weight_tensors(&tensors.filter_prefix(&format!("layers.{}", layer_idx)));
        }

        self.token_embedding
            .load_weight_tensors(&tensors.filter_prefix("embed_tokens"));
        self.output_norm
            .load_weight_tensors(&tensors.filter_prefix("norm"));
    }
}

#[cfg(test)]
mod test {
    use approx::assert_relative_eq;
    use ndarray::{Ix1, Ix2, IxDyn};
    use safetensors::SafeTensors;

    use super::*;
    use crate::utils::{ArrayLoader, tensor_utils, test_utils};

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
            ArrayLoader::<usize, Ix1>::try_from(&test_data.tensor("input_ids").unwrap())
                .unwrap()
                .get()
                .to_vec();
        let logits = ArrayLoader::<f32, Ix2>::try_from(&test_data.tensor("logits").unwrap())
            .unwrap()
            .get();

        let expected_out = logits.slice(s![-1, ..]);

        let test_output = llama3_model.forward(&test_input, &None);

        assert_relative_eq!(
            test_output,
            expected_out,
            epsilon = 1e-4,
            max_relative = 1e-3
        );
    }
}
