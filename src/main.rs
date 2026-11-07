use dotenv::EnvLoader;
use half::bf16;
use memmap2::{Mmap, MmapOptions};
use ndarray::{Array0, Array1, Array2, ArrayD, Axis, IxDyn};
use safetensors::{Dtype, SafeTensors, tensor::TensorView};
use std::{collections::HashMap, env, fs::File, println, vec};
use tokenizers::{Encoding, Tokenizer};

use crate::{
    nn::{Linear, NN, PositionEmbedding},
    utils::tensor_utils,
};

mod attention_functions;
mod config;
mod nn;
mod utils;

fn main() {
    let config = config::load_config();
    let tensor_buffer = load_tensor_buffer().unwrap();
    let tensors = SafeTensors::deserialize(&tensor_buffer).unwrap();

    print_model(&tensors);

    let embedding_tensor = get_embedding_tensor(&tensors).unwrap();
    let tokenizer = load_tokenizer().unwrap();

    let mut test_attention_layer: nn::Attention<f32> = nn::Attention::new(64, 32, 8);
    let test_position_embedding = PositionEmbedding {
        cos: Array1::from_vec(vec![]).into_dyn(),
        sin: Array1::from_vec(vec![]).into_dyn(),
    };

    test_attention_layer.load_weight_tensors(&HashMap::from_iter(
        vec![
            (
                "q_proj.weight".to_string(),
                &tensors
                    .tensor("model.layers.0.self_attn.q_proj.weight")
                    .unwrap(),
            ),
            (
                "k_proj.weight".to_string(),
                &tensors
                    .tensor("model.layers.0.self_attn.k_proj.weight")
                    .unwrap(),
            ),
            (
                "v_proj.weight".to_string(),
                &tensors
                    .tensor("model.layers.0.self_attn.v_proj.weight")
                    .unwrap(),
            ),
            (
                "o_proj.weight".to_string(),
                &tensors
                    .tensor("model.layers.0.self_attn.o_proj.weight")
                    .unwrap(),
            ),
        ]
        .into_iter(),
    ));

    let text = "Hello, World!";
    let embedded_text = embed(&embedding_tensor, &tokenizer, text).unwrap();

    println!("{:?}", embedded_text.shape());

    let output = test_attention_layer.forward(&embedded_text, &test_position_embedding, None, None);

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
