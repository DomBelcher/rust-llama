use core::panic;
use std::{collections::HashMap, println};

use half::bf16;
use ndarray::{Array, Dimension};
use safetensors::{
    Dtype::{BF16, F32, I64},
    SafeTensors,
    tensor::TensorView,
};

// trait I64Unpacker {
//     fn unpack_i64(b: &[u8]) -> Self;
// }

// trait BF16Unpacker {
//     fn unpack_bf16(b: &[u8]) -> Self;
// }
// trait F32Unpacker {
//     fn unpack_f32(b: &[u8]) -> Self;
// }

// impl F32Unpacker for f32 {
//     fn unpack_f32(b: &[u8]) -> f32 {
//         return f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
//     }
// }

// impl BF16Unpacker for f32 {
//     fn unpack_bf16(b: &[u8]) -> f32 {
//         return f32::from_le_bytes([0, 0, b[0], b[1]]);
//     }
// }

// fn unpack_f32<T: F32Unpacker>(data: &[u8]) -> Vec<T> {
//     return data.chunks_exact(4).map(T::unpack_f32).collect::<Vec<T>>();
// }

// fn unpack_bf16<T: BF16Unpacker>(data: &[u8]) -> Vec<T> {
//     return data.chunks_exact(2).map(T::unpack_bf16).collect::<Vec<T>>();
// }

// impl I64Unpacker for usize {
//     fn unpack_i64(b: &[u8]) -> usize {
//         return usize::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]);
//     }
// }

// fn unpack_i64<T: I64Unpacker>(data: &[u8]) -> Vec<T> {
//     return data.chunks_exact(8).map(T::unpack_i64).collect::<Vec<T>>();
// }

// // fn i64_to_usize(b: &[u8]) -> usize {
// //     return usize::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]);
// // }

// trait Unpacker<Source, Dest> {
//     fn unpack()
// }

// struct TensorLoader {}

// pub fn load_from_view<D: Dimension, T>(
//     tensor_view: &TensorView,
// ) -> Result<Array<T, D>, Box<dyn std::error::Error + Send + Sync>> {
//     println!(
//         "Loading tensor view of shape {:?} type {}",
//         tensor_view.shape(),
//         tensor_view.dtype()
//     );

//     let tensor_data = tensor_view.data();
//     let values: Vec<T> = match tensor_view.dtype() {
//         BF16 => unpack_bf16::<T>(tensor_data),
//         F32 => unpack_f32::<T>(tensor_data),
//         I64 => unpack_i64(tensor_data),
//         _ => panic!("unsupported data type"),
//     };

//     println!(
//         "Attempting to unpack {} values into shape {:?}",
//         values.len(),
//         tensor_view.shape()
//     );
//     let tensor = Array::from_shape_vec(tensor_view.shape(), values)?.into_dimensionality()?;
//     return Ok(tensor);
// }

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
