// use ndarray::ArrayD;

// pub struct Activation<T> {
//     f: fn(ArrayD<T>) -> ArrayD<T>,
// }

// impl Activation<f32> {
//     pub fn forward(&self, input: ArrayD<f32>) -> ArrayD<f32> {
//         return (self.f)(input);
//     }

//     pub fn from_unary_map(f: fn(f32) -> f32) -> Self {
//         // fn mapping(input: ArrayD<f32>) -> ArrayD<f32> {
//         //     return input.map(f);
//         // }
//         let mapping = |input: ArrayD<f32>| -> ArrayD<f32> {
//             return input.map(|v| f(*v));
//         };

//         Activation { f: mapping }
//     }
// }
