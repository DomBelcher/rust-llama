use std::vec;

use approx::assert_relative_eq;
use ndarray::{ArrayBase, ArrayD, Axis, Data, IxDyn, concatenate};
use safetensors::SafeTensors;

use crate::{
    attention_functions::multi_headed,
    utils::{position_embedding::RotaryPositionEmbedding, tensor_utils, test_utils},
};

/**
 * query_states <n_tokens, d_k * n_attn_heads>
 * key_states <n_tokens, d_k * n_kv_heads>
 * value_states <n_tokens, d_v * n_kv_heads>
 * attention_mask <?, ?>
 *
 * output <n_tokens, d_v * n_attn_heads>
 */
pub fn grouped_query_attention<S>(
    query_states: &ArrayBase<S, IxDyn>,
    key_states: &ArrayBase<S, IxDyn>,
    value_states: &ArrayBase<S, IxDyn>,
    attention_mask: &Option<ArrayBase<S, IxDyn>>,
    position_embeddings: &RotaryPositionEmbedding,
    n_attention_heads: usize,
    n_kv_heads: usize,
) -> ArrayD<f32>
where
    S: Data<Elem = f32>,
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
    let value_head_states = key_states.to_shape((n_tokens, n_kv_heads, d_v)).unwrap();

    let mut attention_head_outputs = vec![];

    for attn_head_index in 0..n_attention_heads {
        let kv_head_index = attn_head_index / n_attn_heads_per_kv_head;

        // <n_tokens, d_k>
        let q_h = query_head_states.index_axis(Axis(1), attn_head_index);
        println!("q_h");
        println!("{:?}", q_h.shape());
        // <n_tokens, d_k>
        let k_h = key_head_states.index_axis(Axis(1), kv_head_index);
        println!("k_h");
        println!("{:?}", k_h.shape());
        // <n_tokens, d_v>
        let v_h = value_head_states.index_axis(Axis(1), kv_head_index);
        println!("v_h");
        println!("{:?}", v_h.shape());

        // apply position embedding
        let q_h = position_embeddings.forward(&q_h);
        let k_h = position_embeddings.forward(&k_h);

        // <n_tokens, d_v>
        let attn_head_output = multi_headed::multi_headed_attention(
            &q_h.into_dyn(),
            &k_h.into_dyn(),
            &v_h.into_dyn(),
            sqrt_dk,
            &attention_mask,
        );
        println!("head {} attn_head_output", attn_head_index);
        println!("{:?}", attn_head_output.shape());
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

// TODO work out why this fails
// it's probably the attention mask
#[test]
fn test_layer_0() {
    let config = test_utils::load_config();
    let tensor_dict = test_utils::load_tensor_dict();

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

    let test_position_embedding = RotaryPositionEmbedding::new(
        test_query_state_input.shape()[0],
        config.head_dim,
        config.rope_theta,
        &config.rope_scaling,
    );

    let test_output = grouped_query_attention(
        &test_query_state_input,
        &test_key_state_input,
        &test_value_state_input,
        &None,
        &test_position_embedding,
        config.num_attention_heads,
        config.num_key_value_heads,
    );

    assert_relative_eq!(
        test_output,
        expected_out,
        epsilon = 1e-6,
        max_relative = 1e-4
    );
}
