use ndarray::{ArrayBase, ArrayD, Axis, Data, IxDyn, linalg::Dot};

use crate::utils::softmax::Softmax;

/**
 * query_states <n_tokens, d_k>
 * key_states <n_tokens, d_k>
 * value_states <n_tokens, d_v>
 *
 * output <n_tokens, d_v>
 */
#[allow(non_snake_case)]
pub fn multi_headed_attention<SQ, SK, SV, SM>(
    query_states: &ArrayBase<SQ, IxDyn>,
    key_states: &ArrayBase<SK, IxDyn>,
    value_states: &ArrayBase<SV, IxDyn>,
    sqrt_dk: f32,
    attention_mask: &Option<ArrayBase<SM, IxDyn>>,
) -> ArrayD<f32>
where
    SQ: Data<Elem = f32>,
    SK: Data<Elem = f32>,
    SV: Data<Elem = f32>,
    SM: Data<Elem = f32>,
{
    // let d_k = key_states.shape()[1] as f32;

    // QKT <n_tokens, n_tokens>
    let QKT = query_states.dot(&key_states.t());
    // TODO apply attention mask

    let softmax_QKT = (QKT / sqrt_dk).softmax(Axis(1));

    // TODO implement applying attention mask

    return softmax_QKT.dot(&value_states);
}
