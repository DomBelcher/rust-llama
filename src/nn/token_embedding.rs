use std::panic;

use ndarray::{Array1, Array2, ArrayBase, Axis, Data, Ix1, Ix2};
use tokenizers::Encoding;

use crate::{
    nn::NN,
    utils::{ArrayLoader, tensor_utils},
};

pub struct TokenEmbededing<T> {
    weights: Option<Array2<T>>,
}

impl TokenEmbededing<f32> {
    /**
     * encoded_input: Encoding
     *
     * output: <n_tokens, hidden_dim>
     */
    pub fn embed(&self, encoded_input: &Encoding) -> Array2<f32> {
        if self.weights.is_none() {
            panic!("TokenEmbededing::embed called with no weights loaded");
        }

        let input_ids: Vec<usize> = encoded_input
            .get_ids()
            .iter()
            .map(|id| *id as usize)
            .collect();
        return self.embed_ids(&input_ids);
    }

    pub fn embed_ids(&self, input_ids: &Vec<usize>) -> Array2<f32> {
        if self.weights.is_none() {
            panic!("TokenEmbededing::embed_ids called with no weights loaded");
        }

        return self
            .weights
            .as_ref()
            .unwrap()
            .select(Axis(0), input_ids.as_slice());
    }

    /**
     * hidden_state: <hidden_dim>
     *
     * output: <embedding_dim>
     */
    pub fn compute_logits<S>(&self, hidden_state: &ArrayBase<S, Ix1>) -> Array1<f32>
    where
        S: Data<Elem = f32>,
    {
        if self.weights.is_none() {
            panic!("TokenEmbededing::compute_logits called with no weights loaded");
        }

        return self.weights.as_ref().unwrap().dot(hidden_state);
    }

    pub fn new() -> Self {
        return TokenEmbededing { weights: None };
    }
}

impl NN for TokenEmbededing<f32> {
    fn load_weight_tensors(&mut self, tensors: &crate::utils::tensor_utils::TensorDict) {
        println!("Loading tensors for TokenEmbededing from dict");
        println!("{:?}", tensors.key_map);

        let weight_view = tensors.get("weight").unwrap();
        self.weights = Some(
            ArrayLoader::<f32, Ix2>::try_from(&weight_view)
                .unwrap()
                .get(),
        );
    }
}
