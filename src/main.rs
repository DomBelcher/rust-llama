use dotenv::EnvLoader;
use half::bf16;
use memmap2::MmapOptions;
use ndarray::{Array2, ArrayD, Axis};
use safetensors::{Dtype, SafeTensors, tensor::TensorView};
use std::{env, fs::File, println};
use tokenizers::Tokenizer;

fn main() {
    let r = load_tensors().unwrap();
}

fn load_tensors() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let env_map = EnvLoader::new().load()?;
    let tensor_path = env_map.var("TENSOR_PATH")?;
    let tokenizer_path = env_map.var("TOKENIZER_PATH")?;

    let file = File::open(tensor_path)?;
    let buffer = unsafe { MmapOptions::new().map(&file)? };
    let tensors = SafeTensors::deserialize(&buffer)?;

    let embedding_weight = "model.embed_tokens.weight";
    let embedding_tensor_view = tensors.tensor(embedding_weight)?;

    let values = embedding_tensor_view
        .data()
        .chunks_exact(2)
        .map(|b: &[u8]| bf16::from_le_bytes([b[0], b[1]]))
        .collect::<Vec<bf16>>();
    let embedding_tensor = ArrayD::from_shape_vec(embedding_tensor_view.shape(), values)?;
    println!("{:?}", embedding_tensor.shape());

    // println!("Model:");
    // for (name, view) in tensors.tensors() {
    //     println!("{name}: {:?} {:?}", view.dtype(), view.shape());
    // }

    println!("Tokenizer:");

    let tokenizer = Tokenizer::from_file(tokenizer_path)?;
    let enc = tokenizer.encode("Hello, world!", true)?;
    println!("{:?}", enc.get_tokens());
    println!("{:?}", enc.get_ids());
    let text = tokenizer.decode(enc.get_ids(), true)?;
    println!("{}", text);

    let embedded = embed_tokens(&embedding_tensor, enc.get_ids());
    println!("{:?}", embedded.shape());

    Ok(())
}

// fn load_tokenizer () -> Result<()> {
//     let tokenizer = Tokenizer::from_file("tokenizer.json")?;
// }

fn embed_tokens(embedding_tensor: &ArrayD<bf16>, token_ids: &[u32]) -> ArrayD<bf16> {
    let ids: Vec<usize> = token_ids.iter().map(|id| *id as usize).collect();
    return embedding_tensor.select(Axis(0), ids.as_slice());

    // let embedded = token_ids
    //     .map(|token_id: u32| embedding_tensor[token_id])

    // embedded
}
