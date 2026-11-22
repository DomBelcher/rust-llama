mod activation;
mod relu;
mod silu;

// use std::collections::HashMap;

use ndarray::{Array, ArrayBase, Data, Dimension};
// pub use activation::Activation;
pub use relu::relu;
pub use silu::silu;

// (input: &ArrayBase<S, D>) -> Array<f32, D>
// where
//     S: Data<Elem = f32>,
//     D: Dimension,

type Activaion<T, S, D> = fn(&ArrayBase<S, D>) -> Array<T, D>;

// pub fn function_mapping<'a>() -> HashMap<&'a str, Activaion<f32>> {
//     let mut mapping: HashMap<&'a str, Activaion<f32>> = HashMap::new();

//     mapping.insert("relu", relu);
//     mapping.insert("silu", silu);

//     return mapping;
// }

pub fn activation_function<S: Data<Elem = f32>, D: Dimension>(
    function_name: &str,
) -> Option<Activaion<f32, S, D>> {
    match function_name.to_lowercase().as_str() {
        "relu" => Some(relu),
        "silu" => Some(silu),
        _ => None,
    }
}
