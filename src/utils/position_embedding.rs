use ndarray::{Array1, ArrayD};

use crate::config::RopeScalingConfig;

/**
 * cos: <max_seq_len, head_dim / 2>
 * sin: <max_seq_len, head_dim / 2>
 */
pub struct RotaryPositionEmbedding {
    pub cos: ArrayD<f32>,
    pub sin: ArrayD<f32>,
}

impl RotaryPositionEmbedding {
    // TODO work out how to actually calculate these
    // also work out how to apply them in the attention function
    pub fn new(max_seq_len: usize, head_dim: usize, theta: f32) -> Self {
        return RotaryPositionEmbedding {
            cos: Array1::from_vec(vec![]).into_dyn(),
            sin: Array1::from_vec(vec![]).into_dyn(),
        };
    }
}
