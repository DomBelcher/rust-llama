use std::collections::HashMap;

use ndarray::ArrayD;
use safetensors::{SafeTensors, tensor::TensorView};

pub fn load_from_view(
    tensor_view: &TensorView,
) -> Result<ArrayD<f32>, Box<dyn std::error::Error + Send + Sync>> {
    let values = tensor_view
        .data()
        .chunks_exact(2)
        .map(|b: &[u8]| f32::from_le_bytes([0, 0, b[0], b[1]]))
        .collect::<Vec<f32>>();
    let embedding_tensor = ArrayD::from_shape_vec(tensor_view.shape(), values)?;
    return Ok(embedding_tensor);
}

pub struct TensorDict<'a> {
    tensors: &'a SafeTensors<'a>,
    pub key_map: HashMap<&'a str, &'a str>,
}

impl<'a> TensorDict<'a> {
    pub fn new(tensors: &'a SafeTensors) -> Self {
        let keys = tensors.names();

        return TensorDict {
            tensors,
            key_map: keys.iter().map(|&key| (key, key)).collect(),
        };
    }

    pub fn filter_prefix(&self, prefix: &str) -> Self {
        let prefix = format!("{}.", prefix);
        let prefix_length = prefix.len();

        return TensorDict {
            tensors: self.tensors,
            key_map: self
                .key_map
                .iter()
                .filter(|(key, _)| key.starts_with(&prefix))
                .map(|(&key, &tensor_name)| (key.split_at(prefix_length).1, tensor_name))
                .collect(),
        };
    }

    pub fn get(&self, key: &str) -> Option<TensorView> {
        let tensor_name = self.key_map.get(key)?;

        return self
            .tensors
            .tensor(tensor_name)
            .map_or(None, |tensor| Some(tensor));
    }
}
