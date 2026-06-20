use wgpu::util::DeviceExt;

use crate::mlp::NeuralNetwork;

pub struct NeuralNetworkGpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub layers: Vec<GpuLayer>,
    pub matmul_pipeline: Option<wgpu::ComputePipeline>,
    pub matmul_transposed_pipeline: Option<wgpu::ComputePipeline>,
    pub bias_relu_pipeline: Option<wgpu::ComputePipeline>,
    pub softmax_pipeline: Option<wgpu::ComputePipeline>,
    pub error_pipeline: Option<wgpu::ComputePipeline>,
    pub delta_pipeline: Option<wgpu::ComputePipeline>,
    pub update_pipeline: Option<wgpu::ComputePipeline>,
}

impl NeuralNetworkGpu {
    pub async fn new(layer_sizes: Vec<usize>, batch_size: usize) -> Self {
        let mut res = Self::init().await;
        res.create_matmul_pipeline();
        res.create_matmul_transposed_pipeline();
        res.create_bias_relu_pipeline();
        res.create_softmax_pipeline();
        res.create_error_pipeline();
        res.create_delta_pipeline();
        res.create_update_pipeline();

        for (i, size) in layer_sizes.iter().enumerate() {
            let input_nodes = match i {
                0 => 784,
                _ => layer_sizes[i - 1],
            };
            let layer = GpuLayer::new(
                &res.device,
                input_nodes,
                *size,
                &create_random_vec(*size * input_nodes),
                &create_random_vec(*size),
                batch_size,
                false,
            );
            res.layers.push(layer);
        }

        let input_nodes = match layer_sizes.len() {
            0 => 784,
            _ => *layer_sizes.last().unwrap(),
        };

        let layer = GpuLayer::new(
            &res.device,
            input_nodes,
            10,
            &create_random_vec(10 * input_nodes),
            &create_random_vec(10),
            batch_size,
            true,
        );
        res.layers.push(layer);

        res
    }

    pub async fn init() -> Self {
        let instance = wgpu::Instance::default();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .expect("Failed to find an appropriate gpu adapter");

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Neural Network Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                ..Default::default()
            })
            .await
            .expect("Failed to create wgpu device connection");

        Self {
            device,
            queue,
            layers: Vec::new(),
            matmul_pipeline: None,
            matmul_transposed_pipeline: None,
            bias_relu_pipeline: None,
            softmax_pipeline: None,
            error_pipeline: None,
            delta_pipeline: None,
            update_pipeline: None,
        }
    }

    pub fn create_matmul_pipeline(&mut self) {
        let shader_module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("MatMul Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/matmul.wgsl").into()),
            });

        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Matul Shader Bind Group"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("MatMul Pipeline Layout"),
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size: 0,
            });

        self.matmul_pipeline = Some(self.device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("MatMul Compute Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            },
        ));
    }

    pub fn create_matmul_transposed_pipeline(&mut self) {
        let shader_module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("MatMul Transposed Shader"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../shaders/matmul_transposed.wgsl").into(),
                ),
            });

        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Matul Transposed, Shader Bind Group"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("MatMul Transposed Pipeline Layout"),
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size: 0,
            });

        self.matmul_transposed_pipeline = Some(self.device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("MatMul Transposed Compute Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            },
        ));
    }

    pub fn create_bias_relu_pipeline(&mut self) {
        let shader_module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Bias ReLU Shader"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../shaders/bias_add_relu.wgsl").into(),
                ),
            });

        let layout = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Bias Bind Goup Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Bias ReLU Compute Pipeline Layout"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });

        self.bias_relu_pipeline = Some(self.device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("Bias ReLU Compute Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            },
        ))
    }

    pub fn create_softmax_pipeline(&mut self) {
        let shader_module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Softmax Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/softmax.wgsl").into()),
            });

        let layout = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Softmax Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Softmax Pipeline Layout"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });

        self.softmax_pipeline = Some(self.device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("Softmax Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            },
        ));
    }

    pub fn create_error_pipeline(&mut self) {
        let shader_module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Error Eval WGSL Compiler"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../shaders/evaluate_error.wgsl").into(),
                ),
            });
        let layout = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        self.error_pipeline = Some(self.device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("Error Eval Pipeline"),
                layout: Some(&self.device.create_pipeline_layout(
                    &wgpu::PipelineLayoutDescriptor {
                        label: None,
                        bind_group_layouts: &[Some(&layout)],
                        immediate_size: 0,
                    },
                )),
                module: &shader_module,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            },
        ));
    }

    pub fn create_delta_pipeline(&mut self) {
        let shader_module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Delta WGSL Compiler"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../shaders/calculate_delta.wgsl").into(),
                ),
            });
        let layout = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        self.delta_pipeline = Some(self.device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("Delta Pipeline"),
                layout: Some(&self.device.create_pipeline_layout(
                    &wgpu::PipelineLayoutDescriptor {
                        label: None,
                        bind_group_layouts: &[Some(&layout)],
                        immediate_size: 0,
                    },
                )),
                module: &shader_module,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            },
        ));
    }

    pub fn create_update_pipeline(&mut self) {
        let shader_module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Weight Update WGSL Compiler"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../shaders/update_weights.wgsl").into(),
                ),
            });
        let layout = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        self.update_pipeline = Some(self.device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("Weight Update Pipeline"),
                layout: Some(&self.device.create_pipeline_layout(
                    &wgpu::PipelineLayoutDescriptor {
                        label: None,
                        bind_group_layouts: &[Some(&layout)],
                        immediate_size: 0,
                    },
                )),
                module: &shader_module,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            },
        ));
    }

    pub fn calculate(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        input_buffer: &wgpu::Buffer,
        amount: usize,
    ) -> wgpu::Buffer {
        let mut current_working_buffer = input_buffer.clone();

        for layer in &self.layers {
            let a_rows = amount as u32;
            let a_cols = layer.input_nodes as u32;
            let b_cols = layer.output_nodes as u32;

            let input_bytes = (amount * layer.input_nodes * 4) as u64;
            encoder.copy_buffer_to_buffer(
                &current_working_buffer,
                0,
                &layer.prev_input_buffer,
                0,
                input_bytes,
            );

            let output_bytes = (amount * layer.output_nodes * 4) as u64;
            let next_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Intermediate Feature Map Buffer"),
                size: output_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let matmul_dims = MatrixUniforms {
                a_rows,
                a_cols,
                b_cols,
                padding: 0,
            };
            self.dispatch_matmul(
                encoder,
                matmul_dims,
                &current_working_buffer,
                &layer.weights_buffer,
                &next_buffer,
            );

            if !layer.is_output {
                self.dispatch_bias_relu(encoder, a_rows, b_cols, &next_buffer, &layer.bias_buffer);
            }

            encoder.copy_buffer_to_buffer(
                &next_buffer,
                0,
                &layer.prev_output_buffer,
                0,
                output_bytes,
            );

            current_working_buffer = next_buffer;
        }

        let last_layer = self.layers.last().unwrap();
        self.dispatch_softmax(
            encoder,
            amount as u32,
            last_layer.output_nodes as u32,
            &current_working_buffer,
        );

        current_working_buffer
    }

    pub fn train(
        &mut self,
        raw_images: &[u8],
        raw_targets: &[f32],
        amount: usize,
        learning_rate: f32,
    ) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Main Training Batch Encoder"),
            });

        let batch_size = amount as u32;

        let input_image_buf = self.upload_images_to_gpu(raw_images, amount, 784);
        let expected_targets_buf =
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Expected Targets Buffer"),
                    contents: bytemuck::cast_slice(raw_targets),
                    usage: wgpu::BufferUsages::STORAGE,
                });

        // calculate prediction
        let prediction_matrix_buf = self.calculate(&mut encoder, &input_image_buf, amount);

        let last_layer_nodes = self.layers.last().unwrap().output_nodes;
        let initial_error_bytes = (amount * last_layer_nodes * 4) as u64;
        let mut error_grad_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Global Error Gradient Buffer"),
            size: initial_error_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // calculate error
        self.dispatch_error_eval(
            &mut encoder,
            batch_size,
            last_layer_nodes as u32,
            &prediction_matrix_buf,
            &expected_targets_buf,
            &error_grad_buf,
        );

        for layer in self.layers.iter().rev() {
            // caluclate delta gradient
            self.dispatch_delta_calc(
                &mut encoder,
                batch_size,
                layer.output_nodes as u32,
                layer.is_output,
                &error_grad_buf,
                &layer.prev_output_buffer,
                &layer.delta_buffer,
            );

            // update weights and bias
            self.dispatch_weight_update(
                &mut encoder,
                batch_size,
                layer.input_nodes as u32,
                layer.output_nodes as u32,
                learning_rate,
                &layer.prev_input_buffer,
                &layer.delta_buffer,
                &layer.weights_buffer,
                &layer.bias_buffer,
            );

            let next_error_bytes = (amount * layer.input_nodes * std::mem::size_of::<f32>()) as u64;
            let next_error_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Transformed Layer Error Buffer"),
                size: next_error_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let matmul_dims = MatrixTransposeUniforms {
                a_rows: batch_size,
                a_cols: layer.output_nodes as u32,
                b_rows: layer.input_nodes as u32,
                padding: 0,
            };

            // prepare error gradient for next layer
            self.dispatch_matmul_transposed(
                &mut encoder,
                matmul_dims,
                &layer.delta_buffer,
                &layer.weights_buffer,
                &next_error_buf,
            );

            error_grad_buf = next_error_buf;
        }

        self.queue.submit(Some(encoder.finish()));
    }

    pub fn upload_images_to_gpu(
        &self,
        raw_images: &[u8],
        amount: usize,
        input_nodes: usize,
    ) -> wgpu::Buffer {
        let normalized_f32s: Vec<f32> = raw_images
            .iter()
            .map(|&pixel| pixel as f32 / 255.0)
            .collect();

        assert!(normalized_f32s.len() == amount * input_nodes);

        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("GPU Input Image Matrix (A x 784)"),
                contents: bytemuck::cast_slice(&normalized_f32s),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            })
    }

    pub fn download_matrix_from_gpu(
        &self,
        gpu_buffer: &wgpu::Buffer,
        buffer_size_bytes: u64,
    ) -> Vec<f32> {
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Download Staging Buffer"),
            size: buffer_size_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        encoder.copy_buffer_to_buffer(gpu_buffer, 0, &staging_buffer, 0, buffer_size_bytes);
        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();

        buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
            sender.send(v).unwrap();
        });

        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        pollster::block_on(receiver.receive()).unwrap().unwrap();

        let data_view = buffer_slice.get_mapped_range();
        let downloaded_data: Vec<f32> = bytemuck::cast_slice(&data_view).to_vec();

        drop(data_view);
        staging_buffer.unmap();

        downloaded_data
    }

    fn dispatch_matmul(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        dim: MatrixUniforms,
        buf_a: &wgpu::Buffer,
        buf_b: &wgpu::Buffer,
        buf_out: &wgpu::Buffer,
    ) {
        let uniform_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Temp MatMul Uniform"),
                contents: bytemuck::bytes_of(&dim),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self
                .matmul_pipeline
                .as_ref()
                .unwrap()
                .get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf_out.as_entire_binding(),
                },
            ],
        });

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        compute_pass.set_pipeline(self.matmul_pipeline.as_ref().unwrap());
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(dim.b_cols.div_ceil(16), dim.a_rows.div_ceil(16), 1);
    }

    fn dispatch_matmul_transposed(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        dim: MatrixTransposeUniforms,
        buf_a: &wgpu::Buffer,
        buf_b: &wgpu::Buffer,
        buf_out: &wgpu::Buffer,
    ) {
        let uniform_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Temp MatMul Uniform"),
                contents: bytemuck::bytes_of(&dim),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self
                .matmul_transposed_pipeline
                .as_ref()
                .unwrap()
                .get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf_out.as_entire_binding(),
                },
            ],
        });

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        compute_pass.set_pipeline(self.matmul_transposed_pipeline.as_ref().unwrap());
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(dim.b_rows.div_ceil(16), dim.a_rows.div_ceil(16), 1);
    }

    fn dispatch_bias_relu(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        rows: u32,
        cols: u32,
        matrix_buf: &wgpu::Buffer,
        bias_buf: &wgpu::Buffer,
    ) {
        let dim = BiasUniforms { rows, cols };
        let uniform_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Temp Bias Uniform"),
                contents: bytemuck::bytes_of(&dim),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self
                .bias_relu_pipeline
                .as_ref()
                .unwrap()
                .get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: matrix_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bias_buf.as_entire_binding(),
                },
            ],
        });

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        compute_pass.set_pipeline(self.bias_relu_pipeline.as_ref().unwrap());
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(dim.cols.div_ceil(16), dim.rows.div_ceil(16), 1);
    }

    fn dispatch_softmax(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        rows: u32,
        cols: u32,
        matrix_buf: &wgpu::Buffer,
    ) {
        let dim = BiasUniforms { rows, cols };
        let uniform_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Temp Softmax Uniform"),
                contents: bytemuck::bytes_of(&dim),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self
                .softmax_pipeline
                .as_ref()
                .unwrap()
                .get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: matrix_buf.as_entire_binding(),
                },
            ],
        });

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        compute_pass.set_pipeline(self.softmax_pipeline.as_ref().unwrap());
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(dim.rows.div_ceil(64), 1, 1);
    }

    fn dispatch_error_eval(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        batch_size: u32,
        output_nodes: u32,
        pred_buf: &wgpu::Buffer,
        expected_buf: &wgpu::Buffer,
        out_error_buf: &wgpu::Buffer,
    ) {
        assert!(batch_size != 0);
        let size = batch_size * output_nodes;
        let dim = ErrorUniforms {
            size,
            batch_size: batch_size as f32,
        };
        let uniform_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Error Uniform"),
                contents: bytemuck::bytes_of(&dim),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self
                .error_pipeline
                .as_ref()
                .unwrap()
                .get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: pred_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: expected_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out_error_buf.as_entire_binding(),
                },
            ],
        });

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        compute_pass.set_pipeline(self.error_pipeline.as_ref().unwrap());
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(size.div_ceil(256), 1, 1);
    }

    fn dispatch_delta_calc(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        batch_size: u32,
        output_nodes: u32,
        is_output: bool,
        error_grad_buf: &wgpu::Buffer,
        prev_out_buf: &wgpu::Buffer,
        delta_out_buf: &wgpu::Buffer,
    ) {
        let size = batch_size * output_nodes;
        let dim = DeltaUniforms {
            size,
            is_output: if is_output { 1 } else { 0 },
        };
        let uniform_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Delta Uniform"),
                contents: bytemuck::bytes_of(&dim),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self
                .delta_pipeline
                .as_ref()
                .unwrap()
                .get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: error_grad_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: prev_out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: delta_out_buf.as_entire_binding(),
                },
            ],
        });

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        compute_pass.set_pipeline(self.delta_pipeline.as_ref().unwrap());
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(size.div_ceil(256), 1, 1);
    }

    fn dispatch_weight_update(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        batch_size: u32,
        input_nodes: u32,
        output_nodes: u32,
        learning_rate: f32,
        prev_input_buf: &wgpu::Buffer,
        delta_buf: &wgpu::Buffer,
        weigths_buf: &wgpu::Buffer,
        bias_buf: &wgpu::Buffer,
    ) {
        let dim = UpdateUniforms {
            batch_size,
            input_nodes,
            output_nodes,
            learning_rate,
        };
        let uniform_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Update Uniform"),
                contents: bytemuck::bytes_of(&dim),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self
                .update_pipeline
                .as_ref()
                .unwrap()
                .get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: prev_input_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: delta_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: weigths_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: bias_buf.as_entire_binding(),
                },
            ],
        });

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        compute_pass.set_pipeline(self.update_pipeline.as_ref().unwrap());
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(output_nodes.div_ceil(16), input_nodes.div_ceil(16), 1);
    }
}

impl NeuralNetwork for NeuralNetworkGpu {
    fn train(
        &mut self,
        raw_images: &[u8],
        label_data: &[f32],
        batch_size: usize,
        learning_rate: f32,
    ) {
        self.train(raw_images, label_data, batch_size, learning_rate);
    }

    fn test(&mut self, raw_image: &[u8], label: u32) -> bool {
        let buf = self.upload_images_to_gpu(raw_image, 1, 784);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let res_buf = self.calculate(&mut encoder, &buf, 1);
        self.queue.submit(Some(encoder.finish()));

        let result =
            self.download_matrix_from_gpu(&res_buf, 10 * std::mem::size_of::<f32>() as u64);

        let mut max = 0.0;
        let mut max_idx = 0;

        for (i, elem) in result.iter().enumerate() {
            if *elem > max {
                max = *elem;
                max_idx = i as u32;
            }
        }

        max_idx == label
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

pub struct GpuLayer {
    pub weights_buffer: wgpu::Buffer,
    pub bias_buffer: wgpu::Buffer,

    pub prev_input_buffer: wgpu::Buffer,
    pub prev_output_buffer: wgpu::Buffer,
    pub delta_buffer: wgpu::Buffer,

    pub is_output: bool,
    pub input_nodes: usize,
    pub output_nodes: usize,
}

impl GpuLayer {
    pub fn new(
        device: &wgpu::Device,
        input_nodes: usize,
        output_nodes: usize,
        initial_weigths: &[f32],
        initial_bias: &[f32],
        max_batch_size: usize,
        is_output: bool,
    ) -> Self {
        println!(
            "creating layer with weigths buffer of size: {}",
            initial_weigths.len()
        );
        let weights_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Layer Weigths"),
            contents: bytemuck::cast_slice(initial_weigths),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        let bias_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Layer Bias"),
            contents: bytemuck::cast_slice(initial_bias),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let max_input_bytes = max_batch_size * input_nodes * std::mem::size_of::<f32>();
        let max_output_bytes = max_batch_size * output_nodes * std::mem::size_of::<f32>();

        let prev_input_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cache Previous Input"),
            size: max_input_bytes as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let prev_output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cache Previous Output"),
            size: max_output_bytes as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let delta_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cache Delta"),
            size: max_output_bytes as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        Self {
            weights_buffer,
            bias_buffer,
            prev_input_buffer,
            prev_output_buffer,
            delta_buffer,
            is_output,
            input_nodes,
            output_nodes,
        }
    }
}

fn create_random_vec(size: usize) -> Vec<f32> {
    let mut v = Vec::with_capacity(size);

    for _ in 0..size {
        v.push(rand::random_range(-0.5..0.5));
    }

    v
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct MatrixUniforms {
    pub a_rows: u32,
    pub a_cols: u32,
    pub b_cols: u32,
    pub padding: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct MatrixTransposeUniforms {
    pub a_rows: u32,
    pub a_cols: u32,
    pub b_rows: u32,
    pub padding: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BiasUniforms {
    pub rows: u32,
    pub cols: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SoftmaxUniforms {
    pub rows: u32,
    pub cols: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ErrorUniforms {
    pub size: u32,
    pub batch_size: f32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct DeltaUniforms {
    pub size: u32,
    pub is_output: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct UpdateUniforms {
    pub batch_size: u32,
    pub input_nodes: u32,
    pub output_nodes: u32,
    pub learning_rate: f32,
}
