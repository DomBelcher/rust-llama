use std::println;

use ndarray::{ArrayD, Axis};

use crate::{nn::NN, utils::tensor_utils};

// root mean-squared layern norm
pub struct RMSNorm {
    hidden_dim: usize,
    eps: f32,
    // <hidden_dim>
    weights: Option<ArrayD<f32>>,
}

impl RMSNorm {
    pub fn new(hidden_dim: usize, eps: f32) -> Self {
        return RMSNorm {
            hidden_dim,
            eps,
            weights: None,
        };
    }

    /**
     * input: <n_tokens, hidden_dim>
     *
     * output: <n_tokens, hidden_dim>
     */
    // RMS(x) = sqrt(eps + 1 / n * sum_{i=1}^n( x_i^2 ) )
    pub fn forward(&self, input: &ArrayD<f32>) -> ArrayD<f32> {
        // <n_tokens>
        let mean_square = input.pow2().mean_axis(Axis(1)).unwrap();
        println!("mean_square");
        println!("{:?}", mean_square.shape());
        let inv_sqrt = 1_f32 / (mean_square + self.eps).sqrt();
        let hidden_states = input * inv_sqrt.insert_axis(Axis(1));

        return self.w() * hidden_states;
    }

    fn load_weights(&mut self, weights: ArrayD<f32>) {
        let shape = weights.shape();
        if shape.len() != 1 {
            panic!(
                "Shape {:?} is invalid. Expected length 1, found {}",
                shape,
                shape.len()
            );
        }

        if (shape[0]) != self.hidden_dim {
            panic!(
                "Shape {:?} is invalid. Expected length ({}, )",
                shape, self.hidden_dim
            );
        }

        self.weights = Some(weights);
    }

    pub fn w(&self) -> &ArrayD<f32> {
        return match &self.weights {
            Some(weights) => weights,
            None => panic!("Weights called when not loaded"),
        };
    }
}

impl NN for RMSNorm {
    fn load_weight_tensors(&mut self, tensors: &tensor_utils::TensorDict) {
        println!("Loading tensors for RMSNorm from dict");
        println!("{:?}", tensors.key_map);

        self.load_weights(tensor_utils::load_from_view(&tensors.get("weight").unwrap()).unwrap());
    }
}
