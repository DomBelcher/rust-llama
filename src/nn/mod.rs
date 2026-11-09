use safetensors::tensor::TensorView;
use std::collections::HashMap;

mod attention;
mod linear;

pub use attention::{Attention, PositionEmbedding};
pub use linear::Linear;

use crate::utils::tensor_utils::TensorDict;

pub trait NN {
    // fn forward(&self, input: &ArrayD<f32>) -> ArrayD<f32>;

    fn load_weight_tensors(&mut self, tensors: &TensorDict);
}
