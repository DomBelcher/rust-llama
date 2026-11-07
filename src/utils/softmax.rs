use ndarray::{ArrayD, Axis};

pub trait Softmax {
    fn softmax(&self, axis: Axis) -> Self;
}

impl Softmax for ArrayD<f32> {
    fn softmax(&self, axis: Axis) -> ArrayD<f32> {
        let mut output = self.to_owned();

        for mut lane in output.lanes_mut(axis) {
            let lane_max = lane.iter().fold(f32::NEG_INFINITY, |a, b| a.max(*b));

            let mut lane_sum = 0.0_f32;

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

// impl Softmax for f32 {
//     fn softmax(&self) -> Self {
//         return;
//     }
// }
