use ndarray::{Array, ArrayBase, Data, Dimension};

fn sigmoid(x: f32) -> f32 {
    return 1. / (1. + (-x).exp());
}

fn unary_silu(x: f32) -> f32 {
    return x * sigmoid(x);
}

pub fn silu<S, D>(input: &ArrayBase<S, D>) -> Array<f32, D>
where
    S: Data<Elem = f32>,
    D: Dimension,
{
    return input.map(|v| unary_silu(*v));
}
