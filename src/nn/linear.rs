use core::panic;

use ndarray::{Array2, ArrayD, Ix2, IxDyn, linalg::Dot};

use crate::nn::NN;

pub struct Linear<T, const I: usize, const O: usize> {
    // in_dim: usize,
    // out_dim: usize,
    pub weights: Array2<T>,
}

impl<const I: usize, const O: usize> NN for Linear<f32, I, O> {
    fn forward(&self, input: ArrayD<f32>) -> ArrayD<f32> {
        if input.shape()[1] != I {
            panic!("Invalid input dimension");
        }
        if self.weights.shape()[1] != O {
            panic!("Invalid output dimension");
        }

        let input_2d = input.into_dimensionality::<Ix2>().unwrap();

        return input_2d
            .dot(&self.weights)
            .into_dimensionality::<IxDyn>()
            .unwrap();
    }
}

// fn test() {
//     let i: ArrayD<f32> = Array2::from_shape_vec((2, 2), vec![2., 1., 0., 3.])
//         .unwrap()
//         .into_dimensionality::<IxDyn>()
//         .unwrap();
//     println!("Input");
//     println!("{:?}", i);

//     let w: Array2<f32> = Array2::from_shape_vec((2, 2), vec![1., 5., 2., 7.]).unwrap();
//     println!("Weights");
//     println!("{:?}", w);

//     let linear: Linear<f32, 3, 2> = Linear { weights: w };
//     println!("Output");
//     println!("{:?}", linear.forward(i));
// }
