mod attention;
mod decoder_block;
mod gated_mlp;
mod linear;

pub use attention::Attention;
pub use decoder_block::DecoderBlock;
pub use gated_mlp::GatedMLP;
pub use linear::Linear;

use crate::utils::tensor_utils::TensorDict;

pub trait NN {
    // fn forward(&self, input: &ArrayD<f32>) -> ArrayD<f32>;

    fn load_weight_tensors(&mut self, tensors: &TensorDict);
}
