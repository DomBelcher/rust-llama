use ndarray::ArrayD;

mod attention;
mod linear;

pub use linear::Linear;

pub trait NN {
    fn forward(&self, input: ArrayD<f32>) -> ArrayD<f32>;

    fn load_weights(&mut self, weights: ArrayD<f32>);
}
