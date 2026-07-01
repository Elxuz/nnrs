use crate::mlp::Layer;

pub struct Pipelines {
    matmul: wgpu::ComputePipeline,
    matmul_op: wgpu::ComputePipeline,
    bias_add: wgpu::ComputePipeline,
    softmax: wgpu::ComputePipeline,
    error_calc: wgpu::ComputePipeline,
    delta_calc: wgpu::ComputePipeline,
    error_update: wgpu::ComputePipeline,
    weights_update: wgpu::ComputePipeline,
    bias_update: wgpu::ComputePipeline,
}

impl Pipelines {
    pub fn new(device: &wgpu::Device) -> Self {
        Self {
            matmul: Self::create_matmul(device),
            matmul_op: Self::create_matmul_op(device),
            bias_add: Self::create_bias_add(device),
            softmax: Self::create_softmax(device),
            error_calc: Self::create_error_calc(device),
            delta_calc: Self::create_delta_calc(device),
            error_update: Self::create_error_update(device),
            weights_update: Self::create_weights_update(device),
            bias_update: Self::create_bias_update(device),
        }
    }

    pub fn dispatch_matmul(&self, descriptor: DispatchMatmulDescriptor) {
        let bind_group = descriptor
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Matrix Multiplication Bind Group"),
                layout: &self.matmul.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: descriptor.global.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: descriptor.layer.metadata.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: descriptor.calc.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: descriptor.layer.weights.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: descriptor.cache.as_entire_binding(),
                    },
                ],
            });

        let mut compute_pass = descriptor
            .command_encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        compute_pass.set_pipeline(&self.matmul);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(
            descriptor.layer.logical_output_size.div_ceil(16),
            descriptor.layer.logical_batch_size.div_ceil(16),
            1,
        );
    }

    pub fn dispatch_matmul_op(&self, descriptor: DispatchMatmulDescriptor) {
        let bind_group = descriptor
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Matrix Multiplication Optimized Bind Group"),
                layout: &self.matmul_op.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: descriptor.global.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: descriptor.layer.metadata.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: descriptor.calc.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: descriptor.layer.weights.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: descriptor.cache.as_entire_binding(),
                    },
                ],
            });

        let mut compute_pass = descriptor
            .command_encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        compute_pass.set_pipeline(&self.matmul_op);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(
            descriptor.layer.logical_output_size.div_ceil(32),
            descriptor.layer.logical_batch_size.div_ceil(32),
            1,
        );
    }

    pub fn dispatch_bias_add(&self, descriptor: DispatchBiasAddDescriptor) {
        let bind_group = descriptor
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Bias Add Bind Group"),
                layout: &self.bias_add.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: descriptor.global.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: descriptor.layer.metadata.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: descriptor.layer.bias.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: descriptor.calc.as_entire_binding(),
                    },
                ],
            });

        let mut compute_pass = descriptor
            .command_encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor::default());

        compute_pass.set_pipeline(&self.bias_add);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(
            descriptor.layer.logical_output_size.div_ceil(16),
            descriptor.layer.logical_batch_size.div_ceil(16),
            1,
        );
    }

    pub fn dispatch_softmax(&self, descriptor: DispatchSoftmaxDescriptor) {
        let bind_group = descriptor
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Softmax Bind Group"),
                layout: &self.softmax.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: descriptor.global.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: descriptor.layer.metadata.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: descriptor.calc.as_entire_binding(),
                    },
                ],
            });

        let mut compute_pass = descriptor
            .command_encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        compute_pass.set_pipeline(&self.softmax);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(descriptor.layer.logical_batch_size.div_ceil(64), 1, 1);
    }

    pub fn dispatch_error_calc(&self, descriptor: DispatchErrorCalcDescriptor) {
        let bind_group = descriptor
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Error Calculation Bind Group"),
                layout: &self.error_calc.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: descriptor.global.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: descriptor.layer.metadata.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: descriptor.calc.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: descriptor.expected.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: descriptor.error.as_entire_binding(),
                    },
                ],
            });

        let mut compute_pass = descriptor
            .command_encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        compute_pass.set_pipeline(&self.error_calc);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(
            descriptor.layer.logical_output_size.div_ceil(16),
            descriptor.layer.logical_batch_size.div_ceil(16),
            1,
        );
    }

    pub fn dispatch_delta_calc(&self, descriptor: DispatchDeltaCalcDescriptor) {
        let bind_group = descriptor
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Delta Calculation Bind Group"),
                layout: &self.delta_calc.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: descriptor.global.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: descriptor.layer.metadata.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: descriptor.error.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: descriptor.prev_output.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: descriptor.cache.as_entire_binding(),
                    },
                ],
            });

        let mut compute_pass = descriptor
            .command_encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        compute_pass.set_pipeline(&self.delta_calc);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(
            descriptor.layer.logical_output_size.div_ceil(16),
            descriptor.layer.logical_batch_size.div_ceil(16),
            1,
        );
    }

    pub fn dispatch_error_update(&self, descriptor: DispatchErrorUpdateDescriptor) {
        let bind_group = descriptor
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Error Update Bind Group"),
                layout: &self.error_update.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: descriptor.global.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: descriptor.layer.metadata.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: descriptor.cache.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: descriptor.layer.weights.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: descriptor.error.as_entire_binding(),
                    },
                ],
            });

        let mut compute_pass = descriptor
            .command_encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor::default());

        compute_pass.set_pipeline(&self.error_update);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(
            descriptor.layer.logical_input_size.div_ceil(16),
            descriptor.layer.logical_batch_size.div_ceil(16),
            1,
        );
    }

    pub fn dispatch_weights_update(&self, descriptor: DispatchWeightsUpdateDescriptor) {
        let bind_group = descriptor
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Weights Update Bind Group"),
                layout: &self.weights_update.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: descriptor.global.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: descriptor.layer.metadata.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: descriptor.cache.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: descriptor.layer.prev_input.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: descriptor.layer.weights.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: descriptor.layer.w_m_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: descriptor.layer.w_v_buffer.as_entire_binding(),
                    },
                ],
            });

        let mut compute_pass = descriptor
            .command_encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor::default());

        compute_pass.set_pipeline(&self.weights_update);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(
            descriptor.layer.logical_output_size.div_ceil(16),
            descriptor.layer.logical_input_size.div_ceil(16),
            1,
        );
    }

    pub fn dispatch_bias_update(&self, descriptor: DispatchBiasUpdateDescriptor) {
        let bind_group = descriptor
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Bias Update Bind Group"),
                layout: &self.bias_update.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: descriptor.global.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: descriptor.layer.metadata.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: descriptor.cache.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: descriptor.layer.bias.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: descriptor.layer.b_m_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: descriptor.layer.b_v_buffer.as_entire_binding(),
                    },
                ],
            });

        let mut compute_pass = descriptor
            .command_encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor::default());

        compute_pass.set_pipeline(&self.bias_update);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(descriptor.layer.logical_output_size.div_ceil(16), 1, 1);
    }

    fn create_matmul(device: &wgpu::Device) -> wgpu::ComputePipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Matrix Multiplication Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/matmul.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Matrix Multiplication Bind Group"),
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
                        ty: wgpu::BufferBindingType::Uniform,
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
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
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

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Matrix Multiplication Pipeline Layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Matrix Multiplication Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        })
    }

    fn create_matmul_op(device: &wgpu::Device) -> wgpu::ComputePipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Matrix Multiplication Optimized Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/matmul_op.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Matrix Multiplication Optimized Bind Group"),
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
                        ty: wgpu::BufferBindingType::Uniform,
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
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
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

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Matrix Multiplication Optimized Pipeline Layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Matrix Multiplication Optimized Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        })
    }

    fn create_bias_add(device: &wgpu::Device) -> wgpu::ComputePipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Bias Addition Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/bias_add.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bias Addition Bind Group Layout"),
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
                        ty: wgpu::BufferBindingType::Uniform,
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

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Bias Addition Pipeline Layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Bias Addition Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        })
    }

    fn create_softmax(device: &wgpu::Device) -> wgpu::ComputePipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Softmax Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/softmax.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Softmax Bind Group"),
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
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
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

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Softmax Pipeline Layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Softmax Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        })
    }

    fn create_error_calc(device: &wgpu::Device) -> wgpu::ComputePipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Error Calculation Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/error_calc.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Error Calculation Bind Group"),
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
                        ty: wgpu::BufferBindingType::Uniform,
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
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
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

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Error Calculation Pipeline Layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Error Calculation Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        })
    }

    fn create_delta_calc(device: &wgpu::Device) -> wgpu::ComputePipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Delta Calculation Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/delta_calc.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Delta Calculation Bind Group"),
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
                        ty: wgpu::BufferBindingType::Uniform,
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
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
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

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Delta Calculation Pipeline Layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Delta Calculation Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        })
    }

    fn create_error_update(device: &wgpu::Device) -> wgpu::ComputePipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Error Update Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/error_update.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Error Update Bind Group"),
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
                        ty: wgpu::BufferBindingType::Uniform,
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
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
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

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Error Update Pipeline Layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Error Update Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        })
    }

    fn create_weights_update(device: &wgpu::Device) -> wgpu::ComputePipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Weights Update Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/weights_update.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Weights Update Bind Group"),
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
                        ty: wgpu::BufferBindingType::Uniform,
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
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
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
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
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
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Weights Update Pipeline Layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Weights Update Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        })
    }

    fn create_bias_update(device: &wgpu::Device) -> wgpu::ComputePipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Bias Update Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/bias_update.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bias Update Bind Group"),
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
                        ty: wgpu::BufferBindingType::Uniform,
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
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
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

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Bias Update Pipeline Layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Bias Update Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        })
    }
}

pub struct DispatchMatmulDescriptor<'a> {
    pub device: &'a wgpu::Device,
    pub command_encoder: &'a mut wgpu::CommandEncoder,
    pub layer: &'a Layer,
    pub global: &'a wgpu::Buffer,
    pub calc: &'a wgpu::Buffer,
    pub cache: &'a wgpu::Buffer,
}

pub struct DispatchBiasAddDescriptor<'a> {
    pub device: &'a wgpu::Device,
    pub command_encoder: &'a mut wgpu::CommandEncoder,
    pub layer: &'a Layer,
    pub global: &'a wgpu::Buffer,
    pub calc: &'a wgpu::Buffer,
}

pub struct DispatchSoftmaxDescriptor<'a> {
    pub device: &'a wgpu::Device,
    pub command_encoder: &'a mut wgpu::CommandEncoder,
    pub layer: &'a Layer,
    pub global: &'a wgpu::Buffer,
    pub calc: &'a wgpu::Buffer,
}

pub struct DispatchErrorCalcDescriptor<'a> {
    pub device: &'a wgpu::Device,
    pub command_encoder: &'a mut wgpu::CommandEncoder,
    pub layer: &'a Layer,
    pub global: &'a wgpu::Buffer,
    pub calc: &'a wgpu::Buffer,
    pub expected: &'a wgpu::Buffer,
    pub error: &'a wgpu::Buffer,
}

pub struct DispatchDeltaCalcDescriptor<'a> {
    pub device: &'a wgpu::Device,
    pub command_encoder: &'a mut wgpu::CommandEncoder,
    pub layer: &'a Layer,
    pub global: &'a wgpu::Buffer,
    pub error: &'a wgpu::Buffer,
    pub prev_output: &'a wgpu::Buffer,
    pub cache: &'a wgpu::Buffer,
}

pub struct DispatchErrorUpdateDescriptor<'a> {
    pub device: &'a wgpu::Device,
    pub command_encoder: &'a mut wgpu::CommandEncoder,
    pub layer: &'a Layer,
    pub global: &'a wgpu::Buffer,
    pub cache: &'a wgpu::Buffer,
    pub error: &'a wgpu::Buffer,
}

pub struct DispatchWeightsUpdateDescriptor<'a> {
    pub device: &'a wgpu::Device,
    pub command_encoder: &'a mut wgpu::CommandEncoder,
    pub layer: &'a Layer,
    pub global: &'a wgpu::Buffer,
    pub cache: &'a wgpu::Buffer,
}

pub struct DispatchBiasUpdateDescriptor<'a> {
    pub device: &'a wgpu::Device,
    pub command_encoder: &'a mut wgpu::CommandEncoder,
    pub layer: &'a Layer,
    pub global: &'a wgpu::Buffer,
    pub cache: &'a wgpu::Buffer,
}
