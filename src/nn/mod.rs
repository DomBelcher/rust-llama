use safetensors::tensor::TensorView;
use std::collections::HashMap;

use ndarray::ArrayD;

mod attention;
mod linear;

pub use attention::{Attention, PositionEmbedding};
pub use linear::Linear;

pub trait NN {
    // fn forward(&self, input: &ArrayD<f32>) -> ArrayD<f32>;

    fn load_weight_tensors(&mut self, tensors: &HashMap<String, &TensorView>);
}
