use ndarray::ArrayD;

fn unary_relu(v: f32) -> f32 {
    return f32::max(0., v);
}

pub fn relu(input: &ArrayD<f32>) -> ArrayD<f32> {
    return input.map(|v| unary_relu(*v));
}
