use core::fmt;
use std::write;

use ndarray::{Array, Dimension, ShapeError};
use safetensors::{
    Dtype::{BF16, F32, I64},
    tensor::TensorView,
};

pub struct ArrayLoader<A, D>(Array<A, D>);

impl<A, D> ArrayLoader<A, D> {
    pub fn get(self) -> Array<A, D> {
        return self.0;
    }
}

#[derive(Debug, Clone)]
pub struct ArrayLoadError {
    pub msg: String,
}

impl fmt::Display for ArrayLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.msg)
    }
}

impl From<ShapeError> for ArrayLoadError {
    fn from(shape_error: ShapeError) -> Self {
        return ArrayLoadError {
            msg: shape_error.to_string(),
        };
    }
}

impl std::error::Error for ArrayLoadError {}

impl ArrayLoadError {
    fn new(msg: &str) -> Self {
        return ArrayLoadError {
            msg: msg.to_string(),
        };
    }
}

fn unpack<A, D: Dimension>(
    values: Vec<A>,
    shape: &[usize],
) -> Result<ArrayLoader<A, D>, ArrayLoadError> {
    println!(
        "Attempting to unpack {} values into shape {:?}",
        values.len(),
        shape
    );

    let tensor = Array::from_shape_vec(shape, values)
        .map_err(|shape_error| ArrayLoadError::from(shape_error))?
        .into_dimensionality::<D>()
        .map_err(|shape_error| ArrayLoadError::from(shape_error))?;

    return Ok(ArrayLoader(tensor));
}

impl<'a, D: Dimension> TryFrom<&TensorView<'a>> for ArrayLoader<f32, D> {
    type Error = ArrayLoadError;

    fn try_from(tensor_view: &TensorView) -> Result<Self, Self::Error> {
        println!(
            "Loading tensor view of shape {:?} type {}",
            tensor_view.shape(),
            tensor_view.dtype()
        );
        let tensor_data = tensor_view.data();

        let values: Vec<f32> = match tensor_view.dtype() {
            BF16 => Ok(unpack_bf16_to_f32(tensor_data)),
            F32 => Ok(unpack_f32_to_f32(tensor_data)),
            _ => Err(ArrayLoadError::new("unsupported data type")),
        }?;
        return unpack(values, tensor_view.shape());
    }
}

impl<'a, D: Dimension> TryFrom<&TensorView<'a>> for ArrayLoader<usize, D> {
    type Error = ArrayLoadError;

    fn try_from(tensor_view: &TensorView) -> Result<Self, Self::Error> {
        println!(
            "Loading tensor view of shape {:?} type {}",
            tensor_view.shape(),
            tensor_view.dtype()
        );
        let tensor_data = tensor_view.data();

        let values: Vec<usize> = match tensor_view.dtype() {
            I64 => Ok(unpack_i64_to_usize(tensor_data)),
            _ => Err(ArrayLoadError::new("unsupported data type")),
        }?;
        return unpack(values, tensor_view.shape());
    }
}

fn unpack_f32_to_f32(data: &[u8]) -> Vec<f32> {
    return data
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect::<Vec<f32>>();
}

fn unpack_bf16_to_f32(data: &[u8]) -> Vec<f32> {
    return data
        .chunks_exact(2)
        .map(|b| f32::from_le_bytes([0, 0, b[0], b[1]]))
        .collect::<Vec<f32>>();
}

fn unpack_i64_to_usize(data: &[u8]) -> Vec<usize> {
    return data
        .chunks_exact(8)
        .map(|b| usize::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
        .collect::<Vec<usize>>();
}
