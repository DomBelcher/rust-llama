use ndarray::ArrayD;
use safetensors::tensor::TensorView;

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
