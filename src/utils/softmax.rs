use ndarray::{Array, ArrayBase, Axis, Data, Dimension};

pub trait Softmax<D> {
    fn softmax(&self, axis: Axis) -> Array<f32, D>;
}

impl<S, D> Softmax<D> for ArrayBase<S, D>
where
    S: Data<Elem = f32>,
    D: Dimension,
{
    fn softmax(&self, axis: Axis) -> Array<f32, D> {
        let mut output = self.to_owned();

        for mut lane in output.lanes_mut(axis) {
            let lane_max = lane.iter().fold(f32::NEG_INFINITY, |a, b| a.max(*b));

            let mut lane_sum = 0.0;

            for value in lane.iter_mut() {
                *value = (*value - lane_max).exp();
                lane_sum += *value;
            }

            let lane_sum_inverse = 1.0 / lane_sum;
            lane.mapv_inplace(|v| v * lane_sum_inverse);
        }

        return output;
    }
}

#[cfg(test)]
mod test {
    use std::{f32::NEG_INFINITY, println};

    use approx::assert_relative_eq;
    use ndarray::arr2;

    use super::*;
    const ONE_THIRD: f32 = 1. / 3.;

    #[test]
    fn test_softmax() {
        let test_input = arr2::<f32, 3>(&[[1., 1., 1.], [2., 2., 2.]]);
        println!("{:?}", test_input.shape());
        for lane in test_input.lanes(Axis(1)) {
            println!("{:?}", lane.shape())
        }
        let test_output = test_input.softmax(Axis(1));

        let expected_output = arr2::<f32, 3>(&[
            [ONE_THIRD, ONE_THIRD, ONE_THIRD],
            [ONE_THIRD, ONE_THIRD, ONE_THIRD],
        ]);

        assert_relative_eq!(
            test_output,
            expected_output,
            epsilon = 1e-6,
            max_relative = 1e-4
        );
    }

    #[test]
    fn test_softmax_2() {
        let test_input = arr2::<f32, 3>(&[
            [1., 1., 0.],
            [1., 0., 0.],
            [1., 2., 3.],
            [-1000., 0., 1000.],
            [NEG_INFINITY, NEG_INFINITY, 1.],
        ]);
        let test_output = test_input.softmax(Axis(1));

        let expected_output = arr2::<f32, 3>(&[
            [0.42231882, 0.42231882, 0.15536241],
            [0.5761169, 0.21194157, 0.21194157],
            [0.09003057, 0.24472848, 0.66524094],
            [0.00000000, 0.00000000, 1.00000000],
            [0.00000000, 0.00000000, 1.00000000],
        ]);

        assert_relative_eq!(
            test_output,
            expected_output,
            epsilon = 1e-6,
            max_relative = 1e-4
        );
    }
}
