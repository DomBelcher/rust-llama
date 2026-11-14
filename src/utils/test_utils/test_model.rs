use std::sync::LazyLock;

use safetensors::SafeTensors;

use crate::{
    config, load_tensor_buffer,
    nn::{self, NN},
    utils::tensor_utils,
};

pub fn load_test_model() -> &'static nn::DecoderBlock<f32> {
    static MODEL: LazyLock<nn::DecoderBlock<f32>> = LazyLock::new(|| {
        println!("Initialising test model");

        let config = config::load_config();
        let tensor_buffer = load_tensor_buffer().unwrap();
        let tensors = SafeTensors::deserialize(&tensor_buffer).unwrap();
        let tensor_dict = tensor_utils::TensorDict::new(&tensors);
        let mut test_decoder_layer: nn::DecoderBlock<f32> = nn::DecoderBlock::new(&config);

        test_decoder_layer.load_weight_tensors(&tensor_dict.filter_prefix("model.layers.0"));
        return test_decoder_layer;
    });

    return &MODEL;
}
