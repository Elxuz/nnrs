use faer::Mat;
use serde_derive::{Deserialize, Serialize};

use crate::mlp::{
    cpu::{Layer, LearningData, NeuralNetworkCpu},
    gpu::NeuralNetworkGpu,
};

#[derive(Serialize, Deserialize, Debug)]
pub struct NeuralNetworkData {
    input_size: usize,
    output_size: usize,
    hidden_layers: Vec<usize>,
    weights: Vec<f32>,
    bias: Vec<f32>,
}

impl NeuralNetworkData {
    pub fn from_nn_cpu(network: &NeuralNetworkCpu) -> Self {
        let input_size = NeuralNetworkCpu::INPUT_NODES;
        let output_size = NeuralNetworkCpu::OUTPUT_NODES;
        let mut hidden_layers = Vec::new();

        let mut weights = Vec::new();
        let mut bias = Vec::new();

        for layer in &network.layers {
            let l_weights = &layer.weights.data;

            for row in l_weights.row_iter() {
                for num in row.iter() {
                    weights.push(*num);
                }
            }

            // HACK: recounting neurons based on size of bias
            let mut neurons = 0;
            let l_bias = &layer.bias.data;

            for row in l_bias.row_iter() {
                for num in row.iter() {
                    bias.push(*num);
                    neurons += 1;
                }
            }

            if layer.is_output {
                continue;
            }
            hidden_layers.push(neurons);
        }

        Self {
            input_size,
            output_size,
            hidden_layers,
            weights,
            bias,
        }
    }

    pub fn to_nn_cpu(&self) -> NeuralNetworkCpu {
        let mut layers = Vec::new();

        let mut bias_idx = 0;
        let mut weights_idx = 0;

        for (i, layer_size) in self.hidden_layers.iter().enumerate() {
            let prev_layer_size = match i {
                0 => self.input_size,
                _ => self.hidden_layers[i - 1],
            };

            let mut layer = Layer::new(*layer_size, prev_layer_size);

            let bias_slice = &self.bias[bias_idx..bias_idx + *layer_size];
            bias_idx += layer_size;

            layer.bias.data = Mat::from_fn(1, *layer_size, |_, row| bias_slice[row]);

            let weights_slice =
                &self.weights[weights_idx..weights_idx + *layer_size * prev_layer_size];
            weights_idx += layer_size * prev_layer_size;

            layer.weights.data = Mat::from_fn(prev_layer_size, *layer_size, |row, col| {
                weights_slice[col + row * layer_size]
            });

            layers.push(layer);
        }

        let prev_layer_size = match self.hidden_layers.len() {
            0 => self.input_size,
            _ => *self.hidden_layers.last().unwrap(),
        };

        let mut layer = Layer::new_output(self.output_size, prev_layer_size);

        let bias_slice = &self.bias[bias_idx..bias_idx + self.output_size];
        layer.bias.data = Mat::from_fn(1, self.output_size, |_, row| bias_slice[row]);

        let weights_slice =
            &self.weights[weights_idx..weights_idx + self.output_size * prev_layer_size];

        layer.weights.data = Mat::from_fn(prev_layer_size, self.output_size, |row, col| {
            weights_slice[col + row * self.output_size]
        });

        layers.push(layer);

        let learning_data = LearningData {
            learning_rate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            timestep: 1,
        };

        NeuralNetworkCpu {
            layers,
            learning_data,
        }
    }

    pub fn from_nn_gpu(network: &NeuralNetworkGpu) -> Self {
        let input_size = network.layers.first().unwrap().input_nodes;
        let output_size = network.layers.last().unwrap().output_nodes;
        let mut hidden_layers = Vec::new();

        for layer in network.layers.iter() {
            if layer.is_output {
                continue;
            }

            hidden_layers.push(layer.output_nodes);
        }

        let mut weights = Vec::new();
        let mut bias = Vec::new();

        for layer in network.layers.iter() {
            let l_weights = network.download_matrix_from_gpu(
                &layer.weights_buffer,
                (layer.input_nodes * layer.output_nodes * std::mem::size_of::<f32>()) as u64,
            );

            for l_weight in l_weights {
                weights.push(l_weight);
            }

            let l_bias = network.download_matrix_from_gpu(
                &layer.bias_buffer,
                (layer.output_nodes * std::mem::size_of::<f32>()) as u64,
            );

            for l_bias in l_bias {
                bias.push(l_bias);
            }
        }

        Self {
            input_size,
            output_size,
            hidden_layers,
            weights,
            bias,
        }
    }

    pub fn to_nn_gpu(&self) -> NeuralNetworkGpu {
        pollster::block_on(NeuralNetworkGpu::new_with_weigths(
            &self.hidden_layers,
            &self.weights,
            &self.bias,
            1,
            0.1,
        ))
    }
}
