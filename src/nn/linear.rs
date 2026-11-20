use ndarray::{Array2, ArrayD, Ix2, linalg::Dot};

use crate::{
    nn::NN,
    utils::tensor_utils::{self, TensorDict},
};

pub struct Linear<T> {
    // in_dim: usize,
    // out_dim: usize,
    pub weights: Option<ArrayD<T>>,
    // name: String,
}

impl Linear<f32> {
    pub fn of_weights(weights: ArrayD<f32>) -> Self {
        let mut layer: Linear<f32> = Linear::new();
        layer.load_weights(weights);

        return layer;
    }

    pub fn new() -> Self {
        return Self {
            weights: None,
            // name: "".to_string(),
        };
    }

    pub fn w(&self) -> &ArrayD<f32> {
        return match &self.weights {
            Some(weights) => weights,
            None => panic!("Weights called when not loaded"),
        };
    }

    pub fn load_weights(&mut self, weights: ArrayD<f32>) {
        let shape = weights.shape();
        if shape.len() != 2 {
            panic!(
                "Shape {:?} is invalid. Expected length 2, found {}",
                shape,
                shape.len()
            );
        }

        self.weights = Some(weights);
    }

    pub fn forward(&self, input: &ArrayD<f32>) -> ArrayD<f32> {
        if self.weights.is_none() {
            panic!("No weighs loaded");
        }
        return input.dot(self.w());
    }

    pub fn forward_2d(&self, input: &Array2<f32>) -> Array2<f32> {
        if self.weights.is_none() {
            panic!("No weighs loaded");
        }
        return input
            .view()
            .into_dyn()
            .dot(self.w())
            .into_dimensionality::<Ix2>()
            .unwrap();
    }

    // pub fn set_name(&mut self, name: &str) {
    //     self.name = name.to_string();
    // }
}

impl NN for Linear<f32> {
    fn load_weight_tensors(&mut self, tensors: &TensorDict) {
        println!("Loading tensors for Linear from dict");
        println!("{:?}", tensors.key_map);

        let weight_view = tensors.get("weight").unwrap();
        let weights: ndarray::prelude::ArrayBase<
            ndarray::OwnedRepr<f32>,
            ndarray::prelude::Dim<ndarray::IxDynImpl>,
            f32,
        > = tensor_utils::load_from_view(&weight_view).unwrap();
        self.load_weights(weights.reversed_axes());
    }
}

#[cfg(test)]
mod test {
    use approx::assert_relative_eq;
    use safetensors::SafeTensors;

    use super::*;
    use crate::utils::test_utils;

    #[test]
    fn test_layer_0_q_proj() {
        let config = test_utils::load_config();
        let tensor_dict = test_utils::load_tensor_dict();

        let test_data_buffer = test_utils::load_test_data_buffer("hello_world").unwrap();
        let test_data = SafeTensors::deserialize(&test_data_buffer).unwrap();

        let mut linear = Linear::new();
        linear.load_weight_tensors(&tensor_dict.filter_prefix("model.layers.0.self_attn.q_proj"));

        let test_input =
            tensor_utils::load_from_view(&test_data.tensor("l0.q_proj.input").unwrap()).unwrap();
        let expected_out =
            tensor_utils::load_from_view(&test_data.tensor("l0.q_proj.output").unwrap()).unwrap();

        let test_output = linear.forward(&test_input);

        assert_relative_eq!(
            test_output,
            expected_out,
            epsilon = 1e-6,
            max_relative = 1e-4
        );
    }

    #[test]
    fn test_layer_0_k_proj() {
        let config = test_utils::load_config();
        let tensor_dict = test_utils::load_tensor_dict();

        let test_data_buffer = test_utils::load_test_data_buffer("hello_world").unwrap();
        let test_data = SafeTensors::deserialize(&test_data_buffer).unwrap();

        let mut linear = Linear::new();
        linear.load_weight_tensors(&tensor_dict.filter_prefix("model.layers.0.self_attn.k_proj"));

        let test_input =
            tensor_utils::load_from_view(&test_data.tensor("l0.k_proj.input").unwrap()).unwrap();
        let expected_out =
            tensor_utils::load_from_view(&test_data.tensor("l0.k_proj.output").unwrap()).unwrap();

        let test_output = linear.forward(&test_input);

        assert_relative_eq!(
            test_output,
            expected_out,
            epsilon = 1e-6,
            max_relative = 1e-4
        );
    }

    #[test]
    fn test_layer_0_v_proj() {
        let config = test_utils::load_config();
        let tensor_dict = test_utils::load_tensor_dict();

        let test_data_buffer = test_utils::load_test_data_buffer("hello_world").unwrap();
        let test_data = SafeTensors::deserialize(&test_data_buffer).unwrap();

        let mut linear = Linear::new();
        linear.load_weight_tensors(&tensor_dict.filter_prefix("model.layers.0.self_attn.v_proj"));

        let test_input =
            tensor_utils::load_from_view(&test_data.tensor("l0.v_proj.input").unwrap()).unwrap();
        let expected_out =
            tensor_utils::load_from_view(&test_data.tensor("l0.v_proj.output").unwrap()).unwrap();

        let test_output = linear.forward(&test_input);

        assert_relative_eq!(
            test_output,
            expected_out,
            epsilon = 1e-6,
            max_relative = 1e-4
        );
    }

    #[test]
    fn test_layer_0_o_proj() {
        let config = test_utils::load_config();
        let tensor_dict = test_utils::load_tensor_dict();

        let test_data_buffer = test_utils::load_test_data_buffer("hello_world").unwrap();
        let test_data = SafeTensors::deserialize(&test_data_buffer).unwrap();

        let mut linear = Linear::new();
        linear.load_weight_tensors(&tensor_dict.filter_prefix("model.layers.0.self_attn.o_proj"));

        let test_input =
            tensor_utils::load_from_view(&test_data.tensor("l0.o_proj.input").unwrap()).unwrap();
        let expected_out =
            tensor_utils::load_from_view(&test_data.tensor("l0.o_proj.output").unwrap()).unwrap();

        let test_output = linear.forward(&test_input);

        assert_relative_eq!(
            test_output,
            expected_out,
            epsilon = 1e-6,
            max_relative = 1e-4
        );
    }

    #[test]
    fn test_layer_0_mlp_gate_proj() {
        let config = test_utils::load_config();
        let tensor_dict = test_utils::load_tensor_dict();

        let test_data_buffer = test_utils::load_test_data_buffer("hello_world").unwrap();
        let test_data = SafeTensors::deserialize(&test_data_buffer).unwrap();

        let mut linear = Linear::new();
        linear.load_weight_tensors(&tensor_dict.filter_prefix("model.layers.0.mlp.gate_proj"));

        let test_input =
            tensor_utils::load_from_view(&test_data.tensor("l0.mlp.gate_proj.input").unwrap())
                .unwrap();
        let expected_out =
            tensor_utils::load_from_view(&test_data.tensor("l0.mlp.gate_proj.output").unwrap())
                .unwrap();

        let test_output = linear.forward(&test_input);

        assert_relative_eq!(
            test_output,
            expected_out,
            epsilon = 1e-6,
            max_relative = 1e-4
        );
    }

    #[test]
    fn test_layer_0_mlp_up_proj() {
        let config = test_utils::load_config();
        let tensor_dict = test_utils::load_tensor_dict();

        let test_data_buffer = test_utils::load_test_data_buffer("hello_world").unwrap();
        let test_data = SafeTensors::deserialize(&test_data_buffer).unwrap();

        let mut linear = Linear::new();
        linear.load_weight_tensors(&tensor_dict.filter_prefix("model.layers.0.mlp.up_proj"));

        let test_input =
            tensor_utils::load_from_view(&test_data.tensor("l0.mlp.up_proj.input").unwrap())
                .unwrap();
        let expected_out =
            tensor_utils::load_from_view(&test_data.tensor("l0.mlp.up_proj.output").unwrap())
                .unwrap();

        let test_output = linear.forward(&test_input);

        assert_relative_eq!(
            test_output,
            expected_out,
            epsilon = 1e-6,
            max_relative = 1e-4
        );
    }

    #[test]
    fn test_layer_0_mlp_down_proj() {
        let config = test_utils::load_config();
        let tensor_dict = test_utils::load_tensor_dict();

        let test_data_buffer = test_utils::load_test_data_buffer("hello_world").unwrap();
        let test_data = SafeTensors::deserialize(&test_data_buffer).unwrap();

        let mut linear = Linear::new();
        linear.load_weight_tensors(&tensor_dict.filter_prefix("model.layers.0.mlp.down_proj"));

        let test_input =
            tensor_utils::load_from_view(&test_data.tensor("l0.mlp.down_proj.input").unwrap())
                .unwrap();
        let expected_out =
            tensor_utils::load_from_view(&test_data.tensor("l0.mlp.down_proj.output").unwrap())
                .unwrap();

        let test_output = linear.forward(&test_input);

        assert_relative_eq!(
            test_output,
            expected_out,
            epsilon = 1e-6,
            max_relative = 1e-4
        );
    }

    #[test]
    fn test_layer_15_q_proj() {
        let config = test_utils::load_config();
        let tensor_dict = test_utils::load_tensor_dict();

        let test_data_buffer = test_utils::load_test_data_buffer("hello_world").unwrap();
        let test_data = SafeTensors::deserialize(&test_data_buffer).unwrap();

        let mut linear = Linear::new();
        linear.load_weight_tensors(&tensor_dict.filter_prefix("model.layers.15.self_attn.q_proj"));

        let test_input =
            tensor_utils::load_from_view(&test_data.tensor("l15.q_proj.input").unwrap()).unwrap();
        let expected_out =
            tensor_utils::load_from_view(&test_data.tensor("l15.q_proj.output").unwrap()).unwrap();

        let test_output = linear.forward(&test_input);

        assert_relative_eq!(
            test_output,
            expected_out,
            epsilon = 1e-6,
            max_relative = 1e-4
        );
    }

    #[test]
    fn test_layer_15_k_proj() {
        let config = test_utils::load_config();
        let tensor_dict = test_utils::load_tensor_dict();

        let test_data_buffer = test_utils::load_test_data_buffer("hello_world").unwrap();
        let test_data = SafeTensors::deserialize(&test_data_buffer).unwrap();

        let mut linear = Linear::new();
        linear.load_weight_tensors(&tensor_dict.filter_prefix("model.layers.15.self_attn.k_proj"));

        let test_input =
            tensor_utils::load_from_view(&test_data.tensor("l15.k_proj.input").unwrap()).unwrap();
        let expected_out =
            tensor_utils::load_from_view(&test_data.tensor("l15.k_proj.output").unwrap()).unwrap();

        let test_output = linear.forward(&test_input);

        assert_relative_eq!(
            test_output,
            expected_out,
            epsilon = 1e-6,
            max_relative = 1e-4
        );
    }

    #[test]
    fn test_layer_15_v_proj() {
        let config = test_utils::load_config();
        let tensor_dict = test_utils::load_tensor_dict();

        let test_data_buffer = test_utils::load_test_data_buffer("hello_world").unwrap();
        let test_data = SafeTensors::deserialize(&test_data_buffer).unwrap();

        let mut linear = Linear::new();
        linear.load_weight_tensors(&tensor_dict.filter_prefix("model.layers.15.self_attn.v_proj"));

        let test_input =
            tensor_utils::load_from_view(&test_data.tensor("l15.v_proj.input").unwrap()).unwrap();
        let expected_out =
            tensor_utils::load_from_view(&test_data.tensor("l15.v_proj.output").unwrap()).unwrap();

        let test_output = linear.forward(&test_input);

        assert_relative_eq!(
            test_output,
            expected_out,
            epsilon = 1e-6,
            max_relative = 1e-4
        );
    }

    #[test]
    fn test_layer_15_o_proj() {
        let config = test_utils::load_config();
        let tensor_dict = test_utils::load_tensor_dict();

        let test_data_buffer = test_utils::load_test_data_buffer("hello_world").unwrap();
        let test_data = SafeTensors::deserialize(&test_data_buffer).unwrap();

        let mut linear = Linear::new();
        linear.load_weight_tensors(&tensor_dict.filter_prefix("model.layers.15.self_attn.o_proj"));

        let test_input =
            tensor_utils::load_from_view(&test_data.tensor("l15.o_proj.input").unwrap()).unwrap();
        let expected_out =
            tensor_utils::load_from_view(&test_data.tensor("l15.o_proj.output").unwrap()).unwrap();

        let test_output = linear.forward(&test_input);

        assert_relative_eq!(
            test_output,
            expected_out,
            epsilon = 1e-6,
            max_relative = 1e-4
        );
    }
}
