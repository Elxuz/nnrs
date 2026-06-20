use faer::Mat;

use crate::mlp::NeuralNetwork;

#[derive(PartialEq)]
pub struct NeuralNetworkCpu {
    pub layers: Vec<Layer>,
}

impl NeuralNetworkCpu {
    pub const INPUT_NODES: usize = 28 * 28;
    pub const OUTPUT_NODES: usize = 10;
    pub const HIDDEN: [usize; 3] = [70, 60, 50];

    pub fn new() -> Self {
        let mut layers = vec![];

        for (i, layer_size) in Self::HIDDEN.iter().enumerate() {
            let prev_layer_size = match i {
                0 => Self::INPUT_NODES,
                _ => Self::HIDDEN[i - 1],
            };

            layers.push(Layer::new(*layer_size, prev_layer_size));
        }

        layers.push(Layer::new_output(
            Self::OUTPUT_NODES,
            *Self::HIDDEN.last().unwrap(),
        ));

        Self { layers }
    }

    pub fn calculate(&mut self, input: &[u8], amount: usize) -> Mat<f32> {
        assert!(input.len() / amount == Self::INPUT_NODES);

        let input = input.iter().map(|a| *a as f32 / 255.).collect::<Vec<_>>();

        let mut matrix: Mat<f32> = Mat::from_fn(amount, Self::INPUT_NODES, |row, col| {
            input[row * Self::INPUT_NODES + col]
        });

        for layer in &mut self.layers {
            // save metadata for training
            layer.prev_input = Some(Tensor2D::from_mat(matrix.clone()));
            // apply weights
            matrix = &matrix * &layer.weights.data;
            // apply bias
            let ones = Mat::<f32>::from_fn(amount, 1, |_, _| 1.0);
            matrix += &ones * &layer.bias.data;
            // save metadata for training
            layer.prev_output = Some(Tensor2D::from_mat(matrix.clone()));
            // use activation function
            if !layer.is_output {
                matrix = matrix.map(|val| if *val < 0.0 { 0.0 } else { *val });
            }
        }

        Self::softmax(&matrix)
    }

    // pub fn train(&mut self, input: &[u8], expected: &[f32], amount: usize, learning_rate: f32) {
    //     assert!(expected.len() / amount == Self::OUTPUT_NODES);
    //
    //     let expected_mat = Mat::from_fn(amount, Self::OUTPUT_NODES, |row, col| {
    //         expected[row * Self::OUTPUT_NODES + col]
    //     });
    //
    //     let result = self.calculate(input, amount);
    //
    //     // is AxO
    //     let mut error_grad = (result - expected_mat).map(|val| val / amount as f32);
    //
    //     for layer in self.layers.iter_mut().rev() {
    //         // is AxO
    //         let prev_output = &layer.prev_output.as_ref().unwrap().data;
    //         // is AxI
    //         let prev_input = &layer.prev_input.as_ref().unwrap().data;
    //
    //         // is AxO
    //         let delta = Mat::<f32>::from_fn(error_grad.nrows(), error_grad.ncols(), |i, j| {
    //             if layer.is_output || prev_output[(i, j)] > 0.0 {
    //                 error_grad[(i, j)]
    //             } else {
    //                 0.0
    //             }
    //         });
    //
    //         // is IxA * AxO => IxO
    //         let delta_weights = prev_input.transpose() * &delta;
    //         let delta_weights = delta_weights.map(|val| val.clamp(-1.0, 1.0));
    //
    //         let ones = Mat::<f32>::from_fn(1, delta.nrows(), |_, _| 1.0);
    //         let delta_bias = &ones * &delta;
    //
    //         // is AxO * OxI => AxI
    //         error_grad = &delta * layer.weights.data.transpose();
    //
    //         layer.weights.data -= delta_weights.map(|val| val * learning_rate);
    //         layer.bias.data -= delta_bias.map(|val| val * learning_rate);
    //     }
    // }

    fn softmax(matrix: &Mat<f32>) -> Mat<f32> {
        let mut out = matrix.clone();

        for i in 0..matrix.nrows() {
            let mut max_val = f32::MIN;
            for j in 0..matrix.ncols() {
                if matrix[(i, j)] > max_val {
                    max_val = matrix[(i, j)]
                }
            }

            let mut sum = 0.0;
            for j in 0..matrix.ncols() {
                out[(i, j)] = (matrix[(i, j)] - max_val).exp();
                sum += out[(i, j)];
            }
            for j in 0..matrix.ncols() {
                out[(i, j)] /= sum;
            }
        }

        out
    }
}

impl NeuralNetwork for NeuralNetworkCpu {
    fn train(
        &mut self,
        raw_images: &[u8],
        label_data: &[f32],
        batch_size: usize,
        learning_rate: f32,
    ) {
        assert!(label_data.len() / batch_size == Self::OUTPUT_NODES);

        let label_data_mat = Mat::from_fn(batch_size, Self::OUTPUT_NODES, |row, col| {
            label_data[row * Self::OUTPUT_NODES + col]
        });

        let result = self.calculate(raw_images, batch_size);

        // is AxO
        let mut error_grad = (result - label_data_mat).map(|val| val / batch_size as f32);

        for layer in self.layers.iter_mut().rev() {
            // is AxO
            let prev_output = &layer.prev_output.as_ref().unwrap().data;
            // is AxI
            let prev_input = &layer.prev_input.as_ref().unwrap().data;

            // is AxO
            let delta = Mat::<f32>::from_fn(error_grad.nrows(), error_grad.ncols(), |i, j| {
                if layer.is_output || prev_output[(i, j)] > 0.0 {
                    error_grad[(i, j)]
                } else {
                    0.0
                }
            });

            // is IxA * AxO => IxO
            let delta_weights = prev_input.transpose() * &delta;
            let delta_weights = delta_weights.map(|val| val.clamp(-1.0, 1.0));

            let ones = Mat::<f32>::from_fn(1, delta.nrows(), |_, _| 1.0);
            let delta_bias = &ones * &delta;

            // is AxO * OxI => AxI
            error_grad = &delta * layer.weights.data.transpose();

            layer.weights.data -= delta_weights.map(|val| val * learning_rate);
            layer.bias.data -= delta_bias.map(|val| val * learning_rate);
        }
    }

    fn test(&mut self, raw_image: &[u8], label: u32) -> bool {
        let result = self.calculate(raw_image, 1);

        let mut cur_idx = 0;
        let mut max_idx = 0;
        let mut max = 0.0;

        for row in result.row_iter() {
            for elem in row.iter() {
                if *elem > max {
                    max = *elem;
                    max_idx = cur_idx;
                }
                cur_idx += 1;
            }
        }

        max_idx == label
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[derive(PartialEq)]
pub struct Layer {
    pub bias: Tensor,
    pub weights: Tensor2D,
    is_output: bool,

    prev_input: Option<Tensor2D>,
    prev_output: Option<Tensor2D>,
}

impl Layer {
    pub fn new(size: usize, prev_layer_size: usize) -> Self {
        println!("creating tensor width dimension: {prev_layer_size}x{size}");
        Self {
            bias: Tensor::new(size),
            weights: Tensor2D::new(prev_layer_size, size),
            prev_input: None,
            prev_output: None,
            is_output: false,
        }
    }

    pub fn new_output(size: usize, prev_layer_size: usize) -> Self {
        println!("creating tensor width dimension: {prev_layer_size}x{size}");
        Self {
            bias: Tensor::new(size),
            weights: Tensor2D::new(prev_layer_size, size),
            prev_input: None,
            prev_output: None,
            is_output: true,
        }
    }
}

#[derive(PartialEq)]
pub struct Tensor {
    pub data: Mat<f32>,
}

impl Tensor {
    pub fn new(height: usize) -> Self {
        Self {
            // data: Mat::zeros(1, height),
            data: Mat::from_fn(1, height, |_, _| rand::random_range(-0.5..0.5)),
        }
    }
}

#[derive(PartialEq)]
pub struct Tensor2D {
    pub data: Mat<f32>,
}

impl Tensor2D {
    pub fn new(width: usize, height: usize) -> Self {
        // let data: Mat<f32> = Mat::zeros(width, height);
        let data: Mat<f32> = Mat::from_fn(width, height, |_, _| rand::random_range(-0.5..0.5));
        Self { data }
    }

    pub fn from_mat(mat: Mat<f32>) -> Self {
        Self { data: mat }
    }
}
