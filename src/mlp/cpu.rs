use faer::{Mat, traits::math_utils::sqrt, unzip, zip};

use crate::mlp::NeuralNetwork;

pub struct NeuralNetworkCpu {
    pub layers: Vec<Layer>,

    pub learning_data: LearningData,
}

impl NeuralNetworkCpu {
    pub const INPUT_NODES: usize = 28 * 28;
    pub const OUTPUT_NODES: usize = 10;

    pub fn new(layer_sizes: Vec<usize>, learning_rate: f32) -> Self {
        let mut layers = vec![];

        for (i, layer_size) in layer_sizes.iter().enumerate() {
            let prev_layer_size = match i {
                0 => Self::INPUT_NODES,
                _ => layer_sizes[i - 1],
            };

            layers.push(Layer::new(*layer_size, prev_layer_size));
        }

        let prev_layer_size = match layer_sizes.len() {
            0 => Self::INPUT_NODES,
            _ => *layer_sizes.last().unwrap(),
        };

        layers.push(Layer::new_output(Self::OUTPUT_NODES, prev_layer_size));

        let learning_data = LearningData {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            timestep: 1,
        };

        Self {
            layers,
            learning_data,
        }
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
    fn train(&mut self, raw_images: &[u8], label_data: &[f32], batch_size: usize) {
        assert!(label_data.len() / batch_size == Self::OUTPUT_NODES);

        let label_data_mat = Mat::from_fn(batch_size, Self::OUTPUT_NODES, |row, col| {
            label_data[row * Self::OUTPUT_NODES + col]
        });

        let result = self.calculate(raw_images, batch_size);

        // is AxO
        let mut error_grad = (result - label_data_mat).map(|val| val / batch_size as f32);

        let bias_correction1 = 1.0
            - self
                .learning_data
                .beta1
                .powi(self.learning_data.timestep as i32);
        let bias_correction2 = 1.0
            - self
                .learning_data
                .beta2
                .powi(self.learning_data.timestep as i32);

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

            let ones = Mat::<f32>::from_fn(1, delta.nrows(), |_, _| 1.0);
            let delta_bias = &ones * &delta;

            // constant rate learning
            // let delta_weights = delta_weights.map(|val| val.clamp(-1.0, 1.0));

            // dynamic rate learning
            layer.weights_m.data = layer
                .weights_m
                .data
                .map(|val| val * self.learning_data.beta1)
                + delta_weights.map(|val| val * (1.0 - self.learning_data.beta1));
            layer.weights_v.data = layer
                .weights_v
                .data
                .map(|val| val * self.learning_data.beta2)
                + delta_weights.map(|val| val.powi(2) * (1.0 - self.learning_data.beta2));

            let m_hat = layer.weights_m.data.map(|val| val * bias_correction1);
            let v_hat = layer.weights_v.data.map(|val| val * bias_correction2);

            layer.bias_m.data = layer.bias_m.data.map(|val| val * self.learning_data.beta1)
                + delta_bias.map(|val| val * (1.0 - self.learning_data.beta1));
            layer.bias_v.data = layer.bias_v.data.map(|val| val * self.learning_data.beta2)
                + delta_bias.map(|val| val.powi(2) * (1.0 - self.learning_data.beta2));

            let b_m_hat = layer.bias_m.data.map(|val| val * bias_correction1);
            let b_v_hat = layer.bias_v.data.map(|val| val * bias_correction2);

            // is AxO * OxI => AxI
            error_grad = &delta * layer.weights.data.transpose();

            // constant rate learning
            // layer.weights.data -= delta_weights.map(|val| val * self.learning_data.learning_rate);
            // layer.bias.data -= delta_bias.map(|val| val * self.learning_data.learning_rate);

            // dynamic rate learning
            layer.weights.data -= zip!(&m_hat, &v_hat).map(|unzip!(m_hat, v_hat)| {
                (self.learning_data.learning_rate / (sqrt(v_hat) + self.learning_data.epsilon))
                    * m_hat
            });
            layer.bias.data -= zip!(&b_m_hat, &b_v_hat).map(|unzip!(m_hat, v_hat)| {
                (self.learning_data.learning_rate / (sqrt(v_hat) + self.learning_data.epsilon))
                    * m_hat
            });
        }

        self.learning_data.timestep += 1;
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
    pub is_output: bool,

    prev_input: Option<Tensor2D>,
    prev_output: Option<Tensor2D>,

    bias_m: Tensor,
    bias_v: Tensor,
    weights_m: Tensor2D,
    weights_v: Tensor2D,
}

impl Layer {
    pub fn new(size: usize, prev_layer_size: usize) -> Self {
        Self {
            bias: Tensor::new_random(size),
            weights: Tensor2D::new_random(prev_layer_size, size),
            prev_input: None,
            prev_output: None,
            bias_m: Tensor::new(size),
            bias_v: Tensor::new(size),
            weights_m: Tensor2D::new(prev_layer_size, size),
            weights_v: Tensor2D::new(prev_layer_size, size),
            is_output: false,
        }
    }

    pub fn new_output(size: usize, prev_layer_size: usize) -> Self {
        Self {
            bias: Tensor::new_random(size),
            weights: Tensor2D::new_random(prev_layer_size, size),
            prev_input: None,
            prev_output: None,
            bias_m: Tensor::new(size),
            bias_v: Tensor::new(size),
            weights_m: Tensor2D::new(prev_layer_size, size),
            weights_v: Tensor2D::new(prev_layer_size, size),
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
            data: Mat::zeros(1, height),
        }
    }

    pub fn new_random(height: usize) -> Self {
        Self {
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
        let data: Mat<f32> = Mat::zeros(width, height);
        Self { data }
    }

    pub fn new_random(width: usize, height: usize) -> Self {
        let data: Mat<f32> = Mat::from_fn(width, height, |_, _| rand::random_range(-0.5..0.5));
        Self { data }
    }

    pub fn from_mat(mat: Mat<f32>) -> Self {
        Self { data: mat }
    }
}

pub struct LearningData {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub timestep: u32,
}
