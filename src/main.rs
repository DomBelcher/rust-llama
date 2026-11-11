use dotenv::EnvLoader;
use half::bf16;
use memmap2::{Mmap, MmapOptions};
use ndarray::{Array1, ArrayD, Axis};
use safetensors::SafeTensors;
use std::{fs::File, println, vec};
use tokenizers::Tokenizer;

use crate::{
    nn::{NN, PositionEmbedding},
    utils::tensor_utils,
};

mod activation_functions;
mod attention_functions;
mod config;
mod nn;
mod utils;

fn main() {
    let config = config::load_config();
    let tensor_buffer = load_tensor_buffer().unwrap();
    let tensors = SafeTensors::deserialize(&tensor_buffer).unwrap();

    print_model(&tensors);

    let tensor_dict = tensor_utils::TensorDict::new(&tensors);

    let embedding_tensor = get_embedding_tensor(&tensors).unwrap();
    let tokenizer = load_tokenizer().unwrap();

    let mut test_decoder_layer: nn::DecoderBlock<f32> = nn::DecoderBlock::new(64, 32, 8);
    let test_position_embedding = PositionEmbedding {
        cos: Array1::from_vec(vec![]).into_dyn(),
        sin: Array1::from_vec(vec![]).into_dyn(),
    };

    test_decoder_layer.load_weight_tensors(&tensor_dict.filter_prefix("model.layers.0"));

    let text = "Hello, World!";
    let embedded_text = embed(&embedding_tensor, &tokenizer, text).unwrap();

    println!("{:?}", embedded_text.shape());

    let output = test_decoder_layer.forward(&embedded_text, &test_position_embedding, None, None);

    println!("{:?}", output);
}

fn print_model(tensors: &SafeTensors) {
    println!("Model:");

    let mut sorted_tensors = tensors.tensors();
    sorted_tensors.sort_by(|(a_name, _), (b_name, _)| natord::compare(a_name, b_name));

    for (name, view) in sorted_tensors {
        println!("{name}: {:?} {:?}", view.dtype(), view.shape());
    }
}

fn load_tensor_buffer() -> Result<Mmap, Box<dyn std::error::Error + Send + Sync>> {
    let env_map = EnvLoader::new().load()?;
    let tensor_path = env_map.var("TENSOR_PATH")?;

    let file = File::open(tensor_path)?;
    let buffer = unsafe { MmapOptions::new().map(&file)? };
    // let tensors = SafeTensors::deserialize(&buffer)?;

    return Ok(buffer);
}

fn get_embedding_tensor(
    tensors: &SafeTensors,
) -> Result<ArrayD<f32>, Box<dyn std::error::Error + Send + Sync>> {
    let embedding_weight = "model.embed_tokens.weight";
    let embedding_tensor_view = tensors.tensor(embedding_weight)?;

    return tensor_utils::load_from_view(&embedding_tensor_view);
}

fn load_tokenizer() -> Result<Tokenizer, Box<dyn std::error::Error + Send + Sync>> {
    let env_map = EnvLoader::new().load()?;
    let tokenizer_path = env_map.var("TOKENIZER_PATH")?;
    let tokenizer = Tokenizer::from_file(tokenizer_path)?;
    return Ok(tokenizer);
    // let enc = tokenizer.encode("Hello, world!", true)?;
    // return Ok(enc);
}

fn embed(
    embedding_tensor: &ArrayD<f32>,
    tokenizer: &Tokenizer,
    input: &str,
) -> Result<ArrayD<f32>, Box<dyn std::error::Error + Send + Sync>> {
    let encoded = tokenizer.encode(input, true)?;
    let ids: Vec<usize> = encoded.get_ids().iter().map(|id| *id as usize).collect();
    return Ok(embedding_tensor.select(Axis(0), ids.as_slice()));
}

fn embed_tokens(embedding_tensor: &ArrayD<bf16>, token_ids: &[u32]) -> ArrayD<bf16> {
    let ids: Vec<usize> = token_ids.iter().map(|id| *id as usize).collect();
    return embedding_tensor.select(Axis(0), ids.as_slice());
}
