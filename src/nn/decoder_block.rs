use std::collections::HashMap;

use ndarray::ArrayD;

use crate::{
    activation_functions::activation_function,
    nn::{Attention, GatedMLP, NN},
    utils::{position_embedding::RotaryPositionEmbedding, tensor_utils::TensorDict},
};

/**
 * gated_mlp: <in_dim, in_dim>
 * self_attention: <in_dim, in_dim>
 */
pub struct DecoderBlock<T> {
    gated_mlp: GatedMLP<T>,
    self_attention: Attention<T>,
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
        let hidden_state = input;
        // missing input layernorm

        // <n_tokens, hidden_dim>
        let mut hidden_state = self.self_attention.forward(
            hidden_state,
            position_embeddings,
            attention_mask,
            kv_cache,
        );
        // mising post-attention layernorm

        // <n_tokens, hidden_dim>
        hidden_state = self.gated_mlp.forward(&hidden_state);

        hidden_state += residual;

        return hidden_state;
    }

    // TODO make this use config as input?
    pub fn new(head_dim: usize, n_attention_heads: usize, n_kv_groups: usize) -> Self {
        return DecoderBlock {
            gated_mlp: GatedMLP::new(activation_function("silu").unwrap()),
            self_attention: Attention::new(head_dim, n_attention_heads, n_kv_groups),
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
    }
}
