use ndarray::{Array, Array2, ArrayD, Dimension, Ix2, linalg::Dot};

use crate::{
    nn::NN,
    utils::{
        ArrayLoader,
        tensor_utils::{self, TensorDict},
    },
};

pub struct Linear<T, D: Dimension> {
    // in_dim: usize,
    // out_dim: usize,
    pub weights: Option<Array<T, D>>,
    // name: String,
}

impl Linear<f32, Ix2> {
    pub fn of_weights(weights: Array<f32, Ix2>) -> Self {
        let mut layer: Linear<f32, Ix2> = Linear::new();
        layer.load_weights(weights);

        return layer;
    }

    pub fn new() -> Self {
        return Self {
            weights: None,
            // name: "".to_string(),
        };
    }

    pub fn w(&self) -> &Array<f32, Ix2> {
        return match &self.weights {
            Some(weights) => weights,
            None => panic!("Weights called when not loaded"),
        };
    }

    pub fn load_weights(&mut self, weights: Array<f32, Ix2>) {
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

    pub fn forward(&self, input: &Array<f32, Ix2>) -> Array<f32, Ix2> {
        if self.weights.is_none() {
            panic!("No weighs loaded");
        }
        return input.dot(self.w());
    }

    pub fn forward_dyn(&self, input: &ArrayD<f32>) -> Array2<f32> {
        if self.weights.is_none() {
            panic!("No weighs loaded");
        }
        return input
            .view()
            .into_dimensionality::<Ix2>()
            .unwrap()
            .dot(self.w());
    }
}

impl NN for Linear<f32, Ix2> {
    fn load_weight_tensors(&mut self, tensors: &TensorDict) {
        println!("Loading tensors for Linear from dict");
        println!("{:?}", tensors.key_map);

        let weight_view = tensors.get("weight").unwrap();
        let weights = ArrayLoader::<f32, Ix2>::try_from(&weight_view)
            .unwrap()
            .get();
        self.load_weights(weights.reversed_axes());
    }
}

#[cfg(test)]
mod test {
    use approx::assert_relative_eq;
    use ndarray::IxDyn;
    use safetensors::SafeTensors;

    use super::*;
    use crate::utils::{ArrayLoader, test_utils};

    #[test]
    fn test_layer_0_q_proj() {
        let config = test_utils::load_config();
        let tensor_dict = test_utils::load_tensor_dict();

        let test_data_buffer = test_utils::load_test_data_buffer("hello_world").unwrap();
        let test_data = SafeTensors::deserialize(&test_data_buffer).unwrap();

        let mut linear = Linear::new();
        linear.load_weight_tensors(&tensor_dict.filter_prefix("model.layers.0.self_attn.q_proj"));

        let test_input =
            ArrayLoader::<f32, Ix2>::try_from(&test_data.tensor("l0.q_proj.input").unwrap())
                .unwrap()
                .get();
        let expected_out =
            ArrayLoader::<f32, Ix2>::try_from(&test_data.tensor("l0.q_proj.output").unwrap())
                .unwrap()
                .get();

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
            ArrayLoader::<f32, Ix2>::try_from(&test_data.tensor("l0.k_proj.input").unwrap())
                .unwrap()
                .get();
        let expected_out =
            ArrayLoader::<f32, Ix2>::try_from(&test_data.tensor("l0.k_proj.output").unwrap())
                .unwrap()
                .get();

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
            ArrayLoader::<f32, Ix2>::try_from(&test_data.tensor("l0.v_proj.input").unwrap())
                .unwrap()
                .get();
        let expected_out =
            ArrayLoader::<f32, Ix2>::try_from(&test_data.tensor("l0.v_proj.output").unwrap())
                .unwrap()
                .get();

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
            ArrayLoader::<f32, Ix2>::try_from(&test_data.tensor("l0.o_proj.input").unwrap())
                .unwrap()
                .get();
        let expected_out =
            ArrayLoader::<f32, Ix2>::try_from(&test_data.tensor("l0.o_proj.output").unwrap())
                .unwrap()
                .get();

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
            ArrayLoader::<f32, Ix2>::try_from(&test_data.tensor("l0.mlp.gate_proj.input").unwrap())
                .unwrap()
                .get();
        let expected_out = ArrayLoader::<f32, Ix2>::try_from(
            &test_data.tensor("l0.mlp.gate_proj.output").unwrap(),
        )
        .unwrap()
        .get();

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
            ArrayLoader::<f32, Ix2>::try_from(&test_data.tensor("l0.mlp.up_proj.input").unwrap())
                .unwrap()
                .get();
        let expected_out =
            ArrayLoader::<f32, Ix2>::try_from(&test_data.tensor("l0.mlp.up_proj.output").unwrap())
                .unwrap()
                .get();

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
            ArrayLoader::<f32, Ix2>::try_from(&test_data.tensor("l0.mlp.down_proj.input").unwrap())
                .unwrap()
                .get();
        let expected_out = ArrayLoader::<f32, Ix2>::try_from(
            &test_data.tensor("l0.mlp.down_proj.output").unwrap(),
        )
        .unwrap()
        .get();

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
            ArrayLoader::<f32, Ix2>::try_from(&test_data.tensor("l15.q_proj.input").unwrap())
                .unwrap()
                .get();
        let expected_out =
            ArrayLoader::<f32, Ix2>::try_from(&test_data.tensor("l15.q_proj.output").unwrap())
                .unwrap()
                .get();

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
            ArrayLoader::<f32, Ix2>::try_from(&test_data.tensor("l15.k_proj.input").unwrap())
                .unwrap()
                .get();
        let expected_out =
            ArrayLoader::<f32, Ix2>::try_from(&test_data.tensor("l15.k_proj.output").unwrap())
                .unwrap()
                .get();

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
            ArrayLoader::<f32, Ix2>::try_from(&test_data.tensor("l15.v_proj.input").unwrap())
                .unwrap()
                .get();
        let expected_out =
            ArrayLoader::<f32, Ix2>::try_from(&test_data.tensor("l15.v_proj.output").unwrap())
                .unwrap()
                .get();

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
            ArrayLoader::<f32, Ix2>::try_from(&test_data.tensor("l15.o_proj.input").unwrap())
                .unwrap()
                .get();
        let expected_out =
            ArrayLoader::<f32, Ix2>::try_from(&test_data.tensor("l15.o_proj.output").unwrap())
                .unwrap()
                .get();

        let test_output = linear.forward(&test_input);

        assert_relative_eq!(
            test_output,
            expected_out,
            epsilon = 1e-6,
            max_relative = 1e-4
        );
    }
}
