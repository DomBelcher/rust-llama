use core::f64;
use std::panic;

use ndarray::{Array1, Array2, ArrayView, Axis, Ix2, concatenate, s};

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
    pub fn new(
        max_seq_len: usize,
        head_dim: usize,
        theta_base: f64,
        scaling_config: &RopeScalingConfig,
    ) -> Self {
        let max_position_index = (max_seq_len - 1) as f64;

        // exponents <head_dim / 2>
        let exponents =
            Array1::linspace(0.0, (head_dim - 2) as f64, head_dim / 2) / head_dim as f64;
        // theta_1d <head_dim / 2>
        let frequencies_1d = exponents.map(|v| theta_base.powf(*v));
        let frequencies = frequencies_1d.broadcast((1, head_dim / 2)).unwrap();
        // let theta_1d = 1.0 / &frequencies;
        // theta <1, head_dim / 2>

        let original_context_length = scaling_config.original_max_position_embeddings as f64;

        let low_frequency_wavelength = original_context_length / scaling_config.low_freq_factor;
        let high_frequency_wavelength = original_context_length / scaling_config.high_freq_factor;

        if low_frequency_wavelength < high_frequency_wavelength {
            panic!(
                "Incompatible wavelengths {} and {}",
                low_frequency_wavelength, high_frequency_wavelength
            );
        }

        let inverse_frequencies = frequencies.map(|frequency_v| {
            let wavelength = 2.0 * f64::consts::PI * frequency_v;

            if wavelength < high_frequency_wavelength {
                return 1.0 / frequency_v;
            }
            if wavelength > low_frequency_wavelength {
                return 1.0 / (frequency_v * scaling_config.factor);
            }

            let smoothing_factor = (original_context_length / wavelength
                - scaling_config.low_freq_factor)
                / (scaling_config.high_freq_factor - scaling_config.low_freq_factor);

            // todo return interpolated value
            return (1.0 - smoothing_factor) / (frequency_v * scaling_config.factor)
                + smoothing_factor / frequency_v;
        });

        // position_indices <max_seq_len, 1>
        let position_indices =
            Array1::linspace(0.0, max_position_index, max_seq_len).insert_axis(Axis(1));

        // rotations: <max_seq_len, head_dim / 2>
        let rotations = position_indices.dot(&inverse_frequencies);

        return RotaryPositionEmbedding {
            cos: rotations.cos().map(|v| *v as f32),
            sin: rotations.sin().map(|v| *v as f32),
        };
    }

    /**
     * input: <n_tokens, head_dim>
     *
     * ouput: <n_tokens, head_dim>
     */
    pub fn forward(&self, input: &ArrayView<f32, Ix2>) -> Array2<f32> {
        // <n_tokens, head_dim / 2>
        let first_half = first_half(input);
        let second_half = second_half(input);

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
fn first_half<'a>(input: &'a ArrayView<'a, f32, Ix2>) -> ArrayView<'a, f32, Ix2> {
    let head_dim = input.shape()[1] as isize;
    let slice = s![.., 0..(head_dim / 2)];

    return input.slice(slice);
}

fn second_half<'a>(input: &'a ArrayView<'a, f32, Ix2>) -> ArrayView<'a, f32, Ix2> {
    let head_dim = input.shape()[1] as isize;
    let slice = s![.., (head_dim / 2)..(head_dim)];

    return input.slice(slice);
}

#[cfg(test)]
mod test {
    use approx::assert_relative_eq;
    use ndarray::{Ix3, IxDyn};
    use safetensors::SafeTensors;

    use super::*;
    use crate::utils::{tensor_utils, test_utils};

    #[test]
    fn test_cos_sin_calculated_correctly() {
        let config = test_utils::load_config();

        let test_data_buffer = test_utils::load_test_data_buffer("hello_world").unwrap();
        let test_data = SafeTensors::deserialize(&test_data_buffer).unwrap();

        let test_input =
            tensor_utils::load_from_view::<Ix2>(&test_data.tensor("embed.output").unwrap())
                .unwrap();

        let half_slice = s![.., 0..(config.head_dim / 2)];

        let test_cos_value =
            tensor_utils::load_from_view(&test_data.tensor("position_embedding.output.0").unwrap())
                .unwrap();

        let test_sin_value =
            tensor_utils::load_from_view(&test_data.tensor("position_embedding.output.1").unwrap())
                .unwrap();

        let position_embedding = RotaryPositionEmbedding::new(
            test_input.shape()[0],
            config.head_dim,
            config.rope_theta,
            &config.rope_scaling,
        );

        for i in 0_usize..5_usize {
            for j in 0_usize..32_usize {
                let expected = test_cos_value.get(IxDyn(&[i, j])).unwrap();
                let actual = position_embedding.cos.get([i, j]).unwrap();

                if (expected - actual).abs() > 1e-5 {
                    println!("Discrepency in cos at position ({},{})", i, j);
                    println!("Expected value ({}) | actual ({})", expected, actual);
                }
            }
        }

        for i in 0_usize..5_usize {
            for j in 0_usize..32_usize {
                let expected = test_sin_value.get(IxDyn(&[i, j])).unwrap();
                let actual = position_embedding.sin.get([i, j]).unwrap();

                if (expected - actual).abs() > 1e-5 {
                    println!("Discrepency in sin at position ({},{})", i, j);
                    println!("Expected value ({}) | actual ({})", expected, actual);
                }
            }
        }

        assert_relative_eq!(
            position_embedding.cos,
            test_cos_value.slice(half_slice),
            epsilon = 1e-6,
            max_relative = 1e-4
        );

        assert_relative_eq!(
            position_embedding.sin,
            test_sin_value.slice(half_slice),
            epsilon = 1e-6,
            max_relative = 1e-4
        );
    }

    #[test]
    fn test_position_embedding_layer_0() {
        let config = test_utils::load_config();

        let test_data_buffer = test_utils::load_test_data_buffer("hello_world").unwrap();
        let test_data = SafeTensors::deserialize(&test_data_buffer).unwrap();

        let test_input =
            tensor_utils::load_from_view::<Ix2>(&test_data.tensor("embed.output").unwrap())
                .unwrap();

        let position_embedding = RotaryPositionEmbedding::new(
            test_input.shape()[0],
            config.head_dim,
            config.rope_theta,
            &config.rope_scaling,
        );

        let test_query_states =
            tensor_utils::load_from_view::<Ix2>(&test_data.tensor("l0.q_proj.output").unwrap())
                .unwrap();

        let test_key_states =
            tensor_utils::load_from_view::<Ix2>(&test_data.tensor("l0.k_proj.output").unwrap())
                .unwrap();

        println!("test_query_states");
        println!("{:?}", test_query_states.shape());

        let expected_position_embedding_query_output = tensor_utils::load_from_view::<Ix3>(
            &test_data
                .tensor("l0.self_attn.position_embedding.output.query")
                .unwrap(),
        )
        .unwrap();

        let expected_position_embedding_key_output = tensor_utils::load_from_view::<Ix3>(
            &test_data
                .tensor("l0.self_attn.position_embedding.output.key")
                .unwrap(),
        )
        .unwrap();

        for head_idx in 0..config.num_attention_heads {
            println!("Comparing query position embeddings for head {}", head_idx);

            let slice_start = head_idx * config.head_dim;
            let slice_end = (head_idx + 1) * config.head_dim;
            let input_head_slice = s![.., slice_start..slice_end];
            let test_position_embedding_query_ouput =
                position_embedding.forward(&test_query_states.slice(input_head_slice));

            let expected_position_embedding_query_output_head =
                expected_position_embedding_query_output.index_axis(Axis(1), head_idx);

            assert_relative_eq!(
                test_position_embedding_query_ouput,
                expected_position_embedding_query_output_head,
                epsilon = 1e-5,
                max_relative = 1e-3
            );
        }

        for kv_head_idx in 0..config.num_key_value_heads {
            println!(
                "Comparing key position embeddings for query head {}",
                kv_head_idx
            );

            let slice_start = kv_head_idx * config.head_dim;
            let slice_end = (kv_head_idx + 1) * config.head_dim;
            let input_head_slice = s![.., slice_start..slice_end];
            let test_position_embedding_key_ouput =
                position_embedding.forward(&test_key_states.slice(input_head_slice));

            let expected_position_embedding_key_output_head =
                expected_position_embedding_key_output.index_axis(Axis(1), kv_head_idx);

            assert_relative_eq!(
                test_position_embedding_key_ouput,
                expected_position_embedding_key_output_head,
                epsilon = 1e-5,
                max_relative = 1e-3
            );
        }
    }
}
