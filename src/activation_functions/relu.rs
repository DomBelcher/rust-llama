use ndarray::{Array, ArrayBase, Data, Dimension};

fn unary_relu(v: f32) -> f32 {
    return f32::max(0., v);
}

pub fn relu<S, D>(input: &ArrayBase<S, D>) -> Array<f32, D>
where
    S: Data<Elem = f32>,
    D: Dimension,
{
    return input.map(|v| unary_relu(*v));
}
