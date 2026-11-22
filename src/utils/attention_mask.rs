use ndarray::Array2;

pub fn prefill_mask(n_tokens: usize) -> Array2<f32> {
    let mut mask = Array2::from_elem((n_tokens, n_tokens), f32::NEG_INFINITY).triu(0);
    mask.diag_mut().fill(0.);
    return mask;
}
