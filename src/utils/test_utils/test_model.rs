use core::error;
use std::{fs::File, println, sync::LazyLock};

use dotenv::EnvLoader;
use memmap2::{Mmap, MmapOptions};
use safetensors::SafeTensors;

use crate::{
    config, load_tensor_buffer,
    nn::{self, NN},
    utils::tensor_utils,
};

pub fn load_config() -> &'static config::Config<'static> {
    static CONFIG: LazyLock<config::Config> = LazyLock::new(|| {
        let config = config::load_config();
        return config;
    });

    return &CONFIG;
}

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

pub fn load_tensor_dict() -> &'static tensor_utils::TensorDict<'static> {
    static TENSOR_BUFFER: LazyLock<Mmap> = LazyLock::new(|| {
        let tensor_buffer = load_tensor_buffer().unwrap();
        return tensor_buffer;
    });

    static TENSORS: LazyLock<SafeTensors> = LazyLock::new(|| {
        let tensors = SafeTensors::deserialize(&TENSOR_BUFFER).unwrap();
        return tensors;
    });

    static TENSOR_DICT: LazyLock<tensor_utils::TensorDict> = LazyLock::new(|| {
        println!("Initialising test model");

        let tensor_dict = tensor_utils::TensorDict::<'static>::new(&TENSORS);

        return tensor_dict;
    });

    return &TENSOR_DICT;
}

// pub fn load_test_data_buffer(
//     data_name: &str,
// ) -> Result<SafeTensors, Box<dyn std::error::Error + Send + Sync>> {
//     let data_buffer = load_data_buffer(data_name)?;
//     let tensors = SafeTensors::deserialize(&data_buffer)?;

//     return Ok(tensors);
// }

pub fn load_test_data_buffer(
    data_name: &str,
) -> Result<Mmap, Box<dyn std::error::Error + Send + Sync>> {
    let env_map = EnvLoader::new().load()?;
    let test_data_dir = env_map.var("TEST_DATA_DIR")?;
    let test_data_path = test_data_dir + "/" + data_name + ".safetensors";

    println!("Loading data from path: {}", test_data_path);

    let file = File::open(test_data_path)?;
    let buffer = unsafe { MmapOptions::new().map(&file)? };
    // let tensors = SafeTensors::deserialize(&buffer)?;

    return Ok(buffer);
}
