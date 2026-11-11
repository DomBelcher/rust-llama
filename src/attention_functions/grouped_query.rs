use std::vec;

use ndarray::{ArrayBase, ArrayD, Axis, Data, IxDyn, concatenate};

use crate::attention_functions::multi_headed;

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
    n_attention_heads: usize,
    n_kv_heads: usize,
) -> ArrayD<f32>
where
    S: Data<Elem = f32>,
{
    let n_tokens = query_states.shape()[0];
    let d_k = query_states.shape()[1] / n_attention_heads;
    let d_v = value_states.shape()[1] / n_kv_heads;

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
        // <n_tokens, d_k>
        let k_h = key_head_states.index_axis(Axis(1), kv_head_index);
        // <n_tokens, d_v>
        let v_h = value_head_states.index_axis(Axis(1), kv_head_index);

        // <n_tokens, d_v>
        let attn_head_output = multi_headed::multi_headed_attention(
            &q_h.into_dyn(),
            &k_h.into_dyn(),
            &v_h.into_dyn(),
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
