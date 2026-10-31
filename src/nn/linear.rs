use core::panic;

use ndarray::{Array2, ArrayD, Ix2, IxDyn, linalg::Dot};

use crate::nn::NN;

pub struct Linear<T, const I: usize, const O: usize> {
    // in_dim: usize,
    // out_dim: usize,
    pub weights: Option<Array2<T>>,
}

impl<const I: usize, const O: usize> Linear<f32, I, O> {
    pub fn of_weights(weights: ArrayD<f32>) -> Self {
        let mut layer: Linear<f32, I, O> = Linear::init();
        layer.load_weights(weights);

        return layer;
    }

    pub fn init() -> Self {
        return Self { weights: None };
    }

    pub fn w(&self) -> &Array2<f32> {
        return match &self.weights {
            Some(weights) => weights,
            None => panic!("Weights called when not loaded"),
        };
    }
}

impl<const I: usize, const O: usize> NN for Linear<f32, I, O> {
    fn forward(&self, input: ArrayD<f32>) -> ArrayD<f32> {
        if self.weights.is_none() {
            panic!("No weighs loaded");
        }

        if input.shape()[1] != I {
            panic!("Invalid input dimension");
        }
        if self.w().shape()[1] != O {
            panic!("Invalid output dimension");
        }

        let input_2d = input.into_dimensionality::<Ix2>().unwrap();

        return input_2d
            .dot(self.w())
            .into_dimensionality::<IxDyn>()
            .unwrap();
    }

    fn load_weights(&mut self, weights: ArrayD<f32>) {
        let shape = weights.shape();
        if shape.len() != 2 {
            panic!(
                "Shape {:?} is invalid. Expected length 2, found {}",
                shape,
                shape.len()
            );
        }
        if shape[0] != I {
            panic!(
                "Shape {:?} not valid in dimension 0. Expected {}, found {}",
                shape, I, shape[0]
            );
        }
        if shape[1] != O {
            panic!(
                "Shape {:?} not valid in dimension 1. Expected {}, found {}",
                shape, O, shape[1]
            );
        }

        self.weights = Some(weights.into_dimensionality::<Ix2>().unwrap());
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

// let w = Array2::from_shape_vec((2, 2), vec![1., 5., 2., 7.])
//     .unwrap()
//     .into_dimensionality::<IxDyn>()
//     .unwrap();
// let f: Linear<f32, 2, 2> = Linear::of_weights(w);
// }
