mod activation;
mod relu;
mod silu;

// use std::collections::HashMap;

use ndarray::ArrayD;
// pub use activation::Activation;
pub use relu::relu;
pub use silu::silu;

type Activaion<T> = fn(&ArrayD<T>) -> ArrayD<T>;

// pub fn function_mapping<'a>() -> HashMap<&'a str, Activaion<f32>> {
//     let mut mapping: HashMap<&'a str, Activaion<f32>> = HashMap::new();

//     mapping.insert("relu", relu);
//     mapping.insert("silu", silu);

//     return mapping;
// }

pub fn activation_function(function_name: &str) -> Option<Activaion<f32>> {
    match function_name.to_lowercase().as_str() {
        "relu" => Some(relu),
        "silu" => Some(silu),
        _ => None,
    }
}
