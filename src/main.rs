use dotenv::EnvLoader;
use half::bf16;
use memmap2::{Mmap, MmapOptions};
use ndarray::{Array2, ArrayD, Axis};
use safetensors::{Dtype, SafeTensors, tensor::TensorView};
use std::{env, fs::File, println};
use tokenizers::{Encoding, Tokenizer};

mod config;

fn main() {
    let config = config::load_config();
    let tensor_buffer = load_tensor_buffer().unwrap();
    let tensors = SafeTensors::deserialize(&tensor_buffer).unwrap();

    let embedding_tensor = get_embedding_tensor(&tensors).unwrap();
    let tokenizer = load_tokenizer().unwrap();

    let text = "Hello, World!";
    let embedded_text = embed(&embedding_tensor, &tokenizer, text).unwrap();

    println!("{:?}", embedded_text.shape());
}

fn print_model(tensors: &SafeTensors) {
    println!("Model:");
    for (name, view) in tensors.tensors() {
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

    // println!("{:?}", embedding_tensor.shape());

    // println!("Tokenizer:");
    // println!("{:?}", enc.get_tokens());
    // println!("{:?}", enc.get_ids());
    // let text = tokenizer.decode(enc.get_ids(), true)?;
    // println!("{}", text);

    // let embedded = embed_tokens(&embedding_tensor, enc.get_ids());
    // println!("{:?}", embedded.shape());

    // Ok(())
}

fn get_embedding_tensor(
    tensors: &SafeTensors,
) -> Result<ArrayD<bf16>, Box<dyn std::error::Error + Send + Sync>> {
    let embedding_weight = "model.embed_tokens.weight";
    let embedding_tensor_view = tensors.tensor(embedding_weight)?;

    let values = embedding_tensor_view
        .data()
        .chunks_exact(2)
        .map(|b: &[u8]| bf16::from_le_bytes([b[0], b[1]]))
        .collect::<Vec<bf16>>();
    let embedding_tensor = ArrayD::from_shape_vec(embedding_tensor_view.shape(), values)?;
    return Ok(embedding_tensor);
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
    embedding_tensor: &ArrayD<bf16>,
    tokenizer: &Tokenizer,
    input: &str,
) -> Result<ArrayD<bf16>, Box<dyn std::error::Error + Send + Sync>> {
    let encoded = tokenizer.encode("Hello, world!", true)?;
    let ids: Vec<usize> = encoded.get_ids().iter().map(|id| *id as usize).collect();
    return Ok(embedding_tensor.select(Axis(0), ids.as_slice()));
}

fn embed_tokens(embedding_tensor: &ArrayD<bf16>, token_ids: &[u32]) -> ArrayD<bf16> {
    let ids: Vec<usize> = token_ids.iter().map(|id| *id as usize).collect();
    return embedding_tensor.select(Axis(0), ids.as_slice());
}
