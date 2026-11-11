use ndarray::ArrayD;

fn sigmoid(x: f32) -> f32 {
    return 1. / (1. + (-x).exp());
}

fn unary_silu(x: f32) -> f32 {
    return x * sigmoid(x);
}

pub fn silu(input: &ArrayD<f32>) -> ArrayD<f32> {
    return input.map(|v| unary_silu(*v));
}
