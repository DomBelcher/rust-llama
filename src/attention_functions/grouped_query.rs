use std::unimplemented;

use ndarray::{Array, Array2, ArrayBase, ArrayD, ArrayRef, Axis, Data, Dim, Ix, IxDyn, ViewRepr};

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

    let mut output = Array2::<f32>::zeros([n_tokens, d_v]).into_dyn();

    for attn_head_index in 0..n_attention_heads {
        let kv_head_index = attn_head_index / n_attn_heads_per_kv_head;

        // <n_tokens, d_k>
        let q_h = query_head_states.index_axis(Axis(1), attn_head_index);
        // <n_tokens, d_k>
        let k_h = key_head_states.index_axis(Axis(1), kv_head_index);
        // <n_tokens, d_v>
        let v_h = value_head_states.index_axis(Axis(1), kv_head_index);

        // <n_tokens, d_v>
        let attn_head_output: ArrayBase<ndarray::OwnedRepr<f32>, Dim<ndarray::IxDynImpl>, f32> =
            multi_headed::multi_headed_attention(
                &q_h.into_dyn(),
                &k_h.into_dyn(),
                &v_h.into_dyn(),
                &attention_mask,
            );
        output = output + attn_head_output;
    }
    output /= n_attention_heads as f32;

    return output;
}
