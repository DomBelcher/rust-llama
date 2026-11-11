use ndarray::ArrayD;

// use crate::activation_functions::Activation;
use crate::nn::{self, NN};
use crate::utils::tensor_utils::TensorDict;

/**
 * up_projection: <in_dim, hidden_dim>
 * gate_projection: <in_dim, hidden_dim>
 * down_projection: <hidden_dim, out_dim>
 */
pub struct GatedMLP<T> {
    up_projection: nn::Linear<T>,
    gate_projection: nn::Linear<T>,
    down_projection: nn::Linear<T>,
    activation_fn: fn(&ArrayD<T>) -> ArrayD<T>,
}

impl GatedMLP<f32> {
    /**
     * input: <n_tokens, in_dim>
     * output: <n_tokens, out_dim>
     */
    pub fn forward(&self, input: &ArrayD<f32>) -> ArrayD<f32> {
        return self.down_projection.forward(&(self.activation_fn)(
            &(self.up_projection.forward(input) * self.gate_projection.forward(input)),
        ));
    }

    pub fn new(activation_fn: fn(&ArrayD<f32>) -> ArrayD<f32>) -> Self {
        return GatedMLP {
            up_projection: nn::Linear::new(),
            gate_projection: nn::Linear::new(),
            down_projection: nn::Linear::new(),
            activation_fn: activation_fn,
        };
    }
}

impl NN for GatedMLP<f32> {
    fn load_weight_tensors(&mut self, tensors: &TensorDict) {
        println!("Loading tensors for GatedMLP from dict");
        println!("{:?}", tensors.key_map);

        self.up_projection
            .load_weight_tensors(&tensors.filter_prefix("up_proj"));
        self.gate_projection
            .load_weight_tensors(&tensors.filter_prefix("gate_proj"));
        self.down_projection
            .load_weight_tensors(&tensors.filter_prefix("down_proj"));

        // let weights = tensor_utils::load_from_view(weight_view).unwrap();
        // self.load_weights(weights);
    }
}
