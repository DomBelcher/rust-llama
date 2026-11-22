use std::vec;

use ndarray::{Array2, ArrayBase, Axis, Data, Ix2, concatenate};

use crate::{
    attention_functions::multi_headed, utils::position_embedding::RotaryPositionEmbedding,
};

/**
 * query_states <n_tokens, d_k * n_attn_heads>
 * key_states <n_tokens, d_k * n_kv_heads>
 * value_states <n_tokens, d_v * n_kv_heads>
 * attention_mask <?, ?>
 *
 * output <n_tokens, d_v * n_attn_heads>
 */
pub fn grouped_query_attention<SQ, SK, SV, SM>(
    query_states: &ArrayBase<SQ, Ix2>,
    key_states: &ArrayBase<SK, Ix2>,
    value_states: &ArrayBase<SV, Ix2>,
    attention_mask: &Option<ArrayBase<SM, Ix2>>,
    position_embeddings: &RotaryPositionEmbedding,
    n_attention_heads: usize,
    n_kv_heads: usize,
) -> Array2<f32>
where
    SQ: Data<Elem = f32>,
    SK: Data<Elem = f32>,
    SV: Data<Elem = f32>,
    SM: Data<Elem = f32>,
{
    let n_tokens = query_states.shape()[0];
    let d_k = query_states.shape()[1] / n_attention_heads;
    let d_v = value_states.shape()[1] / n_kv_heads;

    let sqrt_dk = (d_k as f32).sqrt();

    let n_attn_heads_per_kv_head = n_attention_heads / n_kv_heads;

    // <n_tokens, n_attn_heads, d_k>
    let query_head_states = query_states
        .to_shape((n_tokens, n_attention_heads, d_k))
        .unwrap();
    // <n_tokens, n_kv_heads, d_k>
    let key_head_states = key_states.to_shape((n_tokens, n_kv_heads, d_k)).unwrap();
    // <n_tokens, n_kv_heads, d_v>
    let value_head_states = value_states.to_shape((n_tokens, n_kv_heads, d_v)).unwrap();

    let mut attention_head_outputs = vec![];

    for attn_head_index in 0..n_attention_heads {
        let kv_head_index = attn_head_index / n_attn_heads_per_kv_head;

        // <n_tokens, d_k>
        let q_h = query_head_states.index_axis(Axis(1), attn_head_index);
        // <n_tokens, d_k>
        let k_h = key_head_states.index_axis(Axis(1), kv_head_index);
        // <n_tokens, d_v>
        let v_h = value_head_states.index_axis(Axis(1), kv_head_index);

        // apply position embedding
        let q_h = position_embeddings.forward(&q_h);
        let k_h = position_embeddings.forward(&k_h);

        // <n_tokens, d_v>
        let attn_head_output =
            multi_headed::multi_headed_attention(&q_h, &k_h, &v_h, sqrt_dk, &attention_mask);
        // output = output + attn_head_output;
        attention_head_outputs.push(attn_head_output);
    }

    return concatenate(
        Axis(1),
        attention_head_outputs
            .iter()
            .map(|a| a.view())
            .collect::<Vec<_>>()
            .as_slice(),
    )
    .unwrap();
}

#[cfg(test)]
mod test {
    use std::println;

    use approx::assert_relative_eq;
    use safetensors::SafeTensors;

    use crate::utils::{attention_mask::prefill_mask, tensor_utils, test_utils};

    use super::*;

    #[test]
    fn test_layer_0() {
        let config = test_utils::load_config();

        let test_data_buffer = test_utils::load_test_data_buffer("hello_world").unwrap();
        let test_data = SafeTensors::deserialize(&test_data_buffer).unwrap();

        let test_query_state_input =
            tensor_utils::load_from_view(&test_data.tensor("l0.q_proj.output").unwrap()).unwrap();
        let test_key_state_input =
            tensor_utils::load_from_view(&test_data.tensor("l0.k_proj.output").unwrap()).unwrap();
        let test_value_state_input =
            tensor_utils::load_from_view(&test_data.tensor("l0.v_proj.output").unwrap()).unwrap();
        let expected_out =
            tensor_utils::load_from_view(&test_data.tensor("l0.o_proj.input").unwrap()).unwrap();

        let n_tokens = test_query_state_input.shape()[0];

        let test_position_embedding = RotaryPositionEmbedding::new(
            n_tokens,
            config.head_dim,
            config.rope_theta,
            &config.rope_scaling,
        );

        let test_attention_mask = prefill_mask(n_tokens);

        let test_output = grouped_query_attention(
            &test_query_state_input,
            &test_key_state_input,
            &test_value_state_input,
            &Some(test_attention_mask),
            &test_position_embedding,
            config.num_attention_heads,
            config.num_key_value_heads,
        );

        println!("expected_out");
        println!("{:?}", expected_out.shape());
        println!("test_output");
        println!("{:?}", test_output.shape());

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

        let test_data_buffer = test_utils::load_test_data_buffer("hello_world").unwrap();
        let test_data = SafeTensors::deserialize(&test_data_buffer).unwrap();

        let test_query_state_input =
            tensor_utils::load_from_view(&test_data.tensor("l15.q_proj.output").unwrap()).unwrap();
        let test_key_state_input =
            tensor_utils::load_from_view(&test_data.tensor("l15.k_proj.output").unwrap()).unwrap();
        let test_value_state_input =
            tensor_utils::load_from_view(&test_data.tensor("l15.v_proj.output").unwrap()).unwrap();
        let expected_out =
            tensor_utils::load_from_view(&test_data.tensor("l15.o_proj.input").unwrap()).unwrap();

        let n_tokens = test_query_state_input.shape()[0];

        let test_position_embedding = RotaryPositionEmbedding::new(
            n_tokens,
            config.head_dim,
            config.rope_theta,
            &config.rope_scaling,
        );

        let test_attention_mask = prefill_mask(n_tokens);

        let test_output = grouped_query_attention(
            &test_query_state_input,
            &test_key_state_input,
            &test_value_state_input,
            &Some(test_attention_mask),
            &test_position_embedding,
            config.num_attention_heads,
            config.num_key_value_heads,
        );

        println!("expected_out");
        println!("{:?}", expected_out.shape());
        println!("test_output");
        println!("{:?}", test_output.shape());

        assert_relative_eq!(
            test_output,
            expected_out,
            epsilon = 1e-6,
            max_relative = 1e-4
        );
    }
}
