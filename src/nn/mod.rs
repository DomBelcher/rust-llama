use std::collections::HashMap;

use ndarray::ArrayD;

mod attention;
mod linear;

pub use linear::Linear;
use safetensors::tensor::TensorView;

pub trait NN {
    fn forward(&self, input: &ArrayD<f32>) -> ArrayD<f32>;

    fn load_weight_tensors(&mut self, tensors: &HashMap<String, &TensorView>);
}
