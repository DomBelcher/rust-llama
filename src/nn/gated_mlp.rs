use ndarray::{Array2, ArrayD, Ix2};

// use crate::activation_functions::Activation;
use crate::nn::{self, NN};
use crate::utils::tensor_utils::TensorDict;

/**
 * up_projection: <in_dim, hidden_dim>
 * gate_projection: <in_dim, hidden_dim>
 * down_projection: <hidden_dim, out_dim>
 */
pub struct GatedMLP<T> {
    up_projection: nn::Linear<T, Ix2>,
    gate_projection: nn::Linear<T, Ix2>,
    down_projection: nn::Linear<T, Ix2>,
    activation_fn: fn(input: &Array2<f32>) -> Array2<f32>,
}

impl GatedMLP<f32> {
    /**
     * input: <n_tokens, in_dim>
     *
     * output: <n_tokens, out_dim>
     */
    pub fn forward(&self, input: &ArrayD<f32>) -> ArrayD<f32> {
        let up = self.up_projection.forward_dyn(input);
        let gated = self.gate_projection.forward_dyn(input);
        return self
            .down_projection
            .forward(&((self.activation_fn)(&gated) * up))
            .into_dyn();
    }

    pub fn new(activation_fn: fn(&Array2<f32>) -> Array2<f32>) -> Self {
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

#[cfg(test)]
mod test {
    use approx::assert_relative_eq;
    use ndarray::IxDyn;
    use safetensors::SafeTensors;

    use super::*;
    use crate::{
        activation_functions::activation_function,
        utils::{
            ArrayLoader, tensor_utils,
            test_utils::{self},
        },
    };

    #[test]
    fn test_layer_0() {
        let tensor_dict = test_utils::load_tensor_dict();

        let test_data_buffer = test_utils::load_test_data_buffer("hello_world").unwrap();
        let test_data = SafeTensors::deserialize(&test_data_buffer).unwrap();

        let mut mlp = GatedMLP::new(activation_function("silu").unwrap());
        mlp.load_weight_tensors(&tensor_dict.filter_prefix("model.layers.0.mlp"));

        let test_input =
            ArrayLoader::<f32, IxDyn>::try_from(&test_data.tensor("l0.mlp_norm.output").unwrap())
                .unwrap()
                .get();
        let expected_out =
            ArrayLoader::<f32, IxDyn>::try_from(&test_data.tensor("l0.mlp_out.output").unwrap())
                .unwrap()
                .get();
        let test_output = mlp.forward(&test_input);

        assert_relative_eq!(
            test_output,
            expected_out,
            epsilon = 1e-6,
            max_relative = 1e-4
        );
    }

    #[test]
    fn test_layer_15() {
        let tensor_dict = test_utils::load_tensor_dict();

        let test_data_buffer = test_utils::load_test_data_buffer("hello_world").unwrap();
        let test_data = SafeTensors::deserialize(&test_data_buffer).unwrap();

        let mut mlp = GatedMLP::new(activation_function("silu").unwrap());
        mlp.load_weight_tensors(&tensor_dict.filter_prefix("model.layers.15.mlp"));

        let test_input =
            ArrayLoader::<f32, IxDyn>::try_from(&test_data.tensor("l15.mlp_norm.output").unwrap())
                .unwrap()
                .get();
        let expected_out =
            ArrayLoader::<f32, IxDyn>::try_from(&test_data.tensor("l15.mlp_out.output").unwrap())
                .unwrap()
                .get();
        let test_output = mlp.forward(&test_input);

        assert_relative_eq!(
            test_output,
            expected_out,
            epsilon = 1e-5,
            max_relative = 1e-3
        );
    }
}
