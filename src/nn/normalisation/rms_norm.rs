use std::println;

use ndarray::{ArrayD, Axis, IxDyn};

use crate::{
    nn::NN,
    utils::{ArrayLoader, tensor_utils},
};

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

        self.load_weights(
            ArrayLoader::<f32, IxDyn>::try_from(&tensors.get("weight").unwrap())
                .unwrap()
                .get(),
        );
    }
}

// TODO test against more prompts
#[cfg(test)]
mod test {
    use approx::assert_relative_eq;
    use safetensors::SafeTensors;

    use super::*;
    use crate::utils::test_utils;

    #[test]
    fn test_layer_0() {
        let config = test_utils::load_config();
        let tensor_dict = test_utils::load_tensor_dict();

        let test_data_buffer = test_utils::load_test_data_buffer("hello_world").unwrap();
        let test_data = SafeTensors::deserialize(&test_data_buffer).unwrap();

        let mut norm = RMSNorm::new(config.hidden_size, config.rms_norm_eps);
        norm.load_weight_tensors(&tensor_dict.filter_prefix("model.layers.0.input_layernorm"));

        let test_input =
            ArrayLoader::<f32, IxDyn>::try_from(&test_data.tensor("l0.attn_norm.input").unwrap())
                .unwrap()
                .get();
        let expected_out =
            ArrayLoader::<f32, IxDyn>::try_from(&test_data.tensor("l0.attn_norm.output").unwrap())
                .unwrap()
                .get();
        // let n_tokens = test_input.shape()[0];
        // let head_size = config.hidden_size / config.num_attention_heads;

        // let per_head_input = test_input
        //     .to_shape((n_tokens, config.num_attention_heads, head_size))
        //     .unwrap();

        // for attn_head_index in 0..config.num_attention_heads {
        //     let head_input = per_head_input.index_axis(Axis(1), attn_head_index);
        // }
        let test_output = norm.forward(&test_input);

        assert_relative_eq!(
            test_output,
            expected_out,
            epsilon = 1e-6,
            max_relative = 1e-4
        );
    }

    #[test]
    fn test_layer_15() {
        let config = test_utils::load_config();
        let tensor_dict = test_utils::load_tensor_dict();

        let test_data_buffer = test_utils::load_test_data_buffer("hello_world").unwrap();
        let test_data = SafeTensors::deserialize(&test_data_buffer).unwrap();

        let mut norm = RMSNorm::new(config.hidden_size, config.rms_norm_eps);
        norm.load_weight_tensors(&tensor_dict.filter_prefix("model.layers.15.input_layernorm"));

        let test_input =
            ArrayLoader::<f32, IxDyn>::try_from(&test_data.tensor("l15.attn_norm.input").unwrap())
                .unwrap()
                .get();
        let expected_out =
            ArrayLoader::<f32, IxDyn>::try_from(&test_data.tensor("l15.attn_norm.output").unwrap())
                .unwrap()
                .get();

        let test_output = norm.forward(&test_input);

        assert_relative_eq!(
            test_output,
            expected_out,
            epsilon = 1e-6,
            max_relative = 1e-4
        );
    }
}
