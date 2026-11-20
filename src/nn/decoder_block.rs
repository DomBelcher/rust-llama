use std::collections::HashMap;

use ndarray::ArrayD;

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
        attention_mask: Option<ArrayD<f32>>,
        kv_cache: Option<HashMap<(), ()>>,
    ) -> ArrayD<f32> {
        let residual = input;
        // let hidden_state = input;
        let hidden_state = self.input_layernorm.forward(input);

        // <n_tokens, hidden_dim>
        let hidden_state = self.self_attention.forward(
            &hidden_state,
            position_embeddings,
            attention_mask,
            kv_cache,
        );
        let hidden_state = self.input_layernorm.forward(&hidden_state);
        // mising post-attention layernorm

        // <n_tokens, hidden_dim>
        let hidden_state = self.gated_mlp.forward(&hidden_state);

        // let hidden_state += residual;

        return hidden_state + residual;
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
