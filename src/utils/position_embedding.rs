use std::println;

use ndarray::{Array1, Array2, ArrayD, ArrayView, Axis, Ix2, IxDyn, concatenate, s};

use crate::config::RopeScalingConfig;

// it might be possible to adapt this to accept n-dimensional inputs
// like <max_seq_len, ... , head_dim / 2>
// so it can be applied to Q and K for all heads at once
// instead of per-head

/**
 * cos: <max_seq_len, head_dim / 2>
 * sin: <max_seq_len, head_dim / 2>
 */
pub struct RotaryPositionEmbedding {
    pub cos: Array2<f32>,
    pub sin: Array2<f32>,
}

impl RotaryPositionEmbedding {
    // TODO apply long context scaling
    pub fn new(max_seq_len: usize, head_dim: usize, theta_base: f32) -> Self {
        let max_exponent_index = (head_dim / 2 - 1) as f32;
        let max_position_index = (max_seq_len - 1) as f32;

        // exponents <head_dim / 2>
        let exponents =
            -2_f32 * Array1::linspace(0_f32, max_exponent_index, head_dim / 2) / head_dim as f32;

        println!("exponents");
        println!("{:?}", exponents.shape());

        // theta_1d <head_dim / 2>
        let theta_1d = exponents.map(|v| theta_base.powf(*v));
        // theta <1, head_dim / 2>
        let theta = theta_1d.broadcast((1, head_dim / 2)).unwrap();

        println!("theta");
        println!("{:?}", theta.shape());

        // position_indices <max_seq_len, 1>
        let position_indices =
            Array1::linspace(0_f32, max_position_index, max_seq_len).insert_axis(Axis(1));
        // .broadcast((max_seq_len, 1))
        // .unwrap();

        println!("position_indices");
        println!("{:?}", position_indices.shape());

        // rotations: <max_seq_len, head_dim / 2>
        let rotations = position_indices
            // .broadcast((max_seq_len, 1))
            // .unwrap()
            .dot(&theta);

        println!("rotations");
        println!("{:?}", rotations.shape());

        return RotaryPositionEmbedding {
            cos: rotations.cos(),
            sin: rotations.sin(),
        };
    }

    /**
     * input: <n_tokens, head_dim>
     *
     * ouput: <n_tokens, head_dim>
     */
    pub fn forward(&self, input: &ArrayView<f32, Ix2>) -> ArrayD<f32> {
        // <n_tokens, head_dim / 2>
        let first_half = first_half(input);
        let second_half = second_half(input);

        println!("first_half");
        println!("{:?}", first_half.shape());
        println!("second_half");
        println!("{:?}", second_half.shape());

        let cos_first_half = &self.cos * &first_half;
        let cos_second_half = &self.cos * &second_half;

        let sin_first_half = &self.sin * &first_half;
        let neg_sin_second_half = -&self.sin * &second_half;

        return concatenate(Axis(1), &[cos_first_half.view(), cos_second_half.view()]).unwrap()
            + concatenate(
                Axis(1),
                &[neg_sin_second_half.view(), sin_first_half.view()],
            )
            .unwrap();
    }
}

/**
 * input: <n_tokens, head_dim>
 *
 * ouput: <n_tokens, head_dim / 2>
 */
fn first_half<'a>(input: &'a ArrayView<'a, f32, Ix2>) -> ArrayView<'a, f32, IxDyn> {
    let head_dim = input.shape()[1] as isize;
    let slice = s![.., 0..(head_dim / 2)];

    return input.slice(slice).into_dyn();
}

fn second_half<'a>(input: &'a ArrayView<'a, f32, Ix2>) -> ArrayView<'a, f32, IxDyn> {
    let head_dim = input.shape()[1] as isize;
    let slice = s![.., (head_dim / 2)..(head_dim)];

    return input.slice(slice).into_dyn();
}

/**
 * input: <n_tokens, head_dim>
 *
 * ouput: <n_tokens, head_dim>
 */
fn rotate_half(input: &ArrayD<f32>) -> ArrayD<f32> {
    let head_dim = input.shape()[1] as isize;
    let slice1 = s![.., 0..(head_dim / 2 - 1)];
    let slice2 = s![.., (head_dim / 2)..(head_dim - 1)];

    // this performs several clones to take ownership of the views from each slice
    // it might be possible to do all this without a copy and return a view
    // but then the negation of x2 would need to be performed in ::forward

    // <n_tokens, head_dim / 2>
    let x1 = input.slice(slice1).to_owned();
    let x2 = input.slice(slice2).to_owned();

    return concatenate(Axis(0), &[(-x2).view(), x1.view()])
        .unwrap()
        .into_dyn();
}
