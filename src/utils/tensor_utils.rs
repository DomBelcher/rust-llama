use core::panic;
use std::{collections::HashMap, println};

use ndarray::ArrayD;
use safetensors::{Dtype::BF16, Dtype::F32, SafeTensors, tensor::TensorView};

fn unpack_f32(data: &[u8]) -> Vec<f32> {
    return data
        .chunks_exact(4)
        .map(|b: &[u8]| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect::<Vec<f32>>();
}

fn unpack_bf16(data: &[u8]) -> Vec<f32> {
    return data
        .chunks_exact(2)
        .map(|b: &[u8]| f32::from_le_bytes([0, 0, b[0], b[1]]))
        .collect::<Vec<f32>>();
}

pub fn load_from_view(
    tensor_view: &TensorView,
) -> Result<ArrayD<f32>, Box<dyn std::error::Error + Send + Sync>> {
    println!(
        "Loading tensor view of shape {:?} type {}",
        tensor_view.shape(),
        tensor_view.dtype()
    );

    let tensor_data = tensor_view.data();
    let values = match tensor_view.dtype() {
        BF16 => unpack_bf16(tensor_data),
        F32 => unpack_f32(tensor_data),
        _ => panic!("unsupported data type"),
    };

    // let values = tensor_view
    //     .data()
    //     .chunks_exact(2)
    //     .map(|b: &[u8]| f32::from_le_bytes([0, 0, b[0], b[1]]))
    //     .collect::<Vec<f32>>();
    println!(
        "Attempting to unpack {} values into shape {:?}",
        values.len(),
        tensor_view.shape()
    );
    let tensor = ArrayD::from_shape_vec(tensor_view.shape(), values)?;
    return Ok(tensor);
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
