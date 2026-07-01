use wgpu::util::DeviceExt;

use crate::mlp::pipelines::*;

const F32_SIZE: u64 = std::mem::size_of::<f32>() as u64;

pub struct MultiLayerPerceptron {
    // handles to gpu resources
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pipelines: Pipelines,

    // actual nn data
    global_data: wgpu::Buffer,

    calculation_buf: wgpu::Buffer,
    expected_buf: wgpu::Buffer,
    cache_buf: wgpu::Buffer,
    error_buf: wgpu::Buffer,

    prefetch_global_data: wgpu::Buffer,
    prefetch_input_buf: wgpu::Buffer,
    prefetch_expected_buf: wgpu::Buffer,

    // metadata
    pub layers: Vec<Layer>,
    logical_input_size: u32,
    physical_input_size: u32,

    logical_output_size: u32,
    physical_output_size: u32,

    logical_batch_size: u32,
    physical_batch_size: u32,

    max_size: u32,
    timestep: u32,
    learning_rate: f32,

    cur_batch_size: u32,
    next_batch_size: u32,
}

impl MultiLayerPerceptron {
    pub fn new(descriptor: &MLPDescriptor) -> Self {
        // initialize gpu resources
        let instance = wgpu::Instance::default();

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .expect("Failed to retrieve gpu device adapter");

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("MultiLayerPerceptron Device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            ..Default::default()
        }))
        .expect("Failed to retrieve gpu device");

        let pipelines = Pipelines::new(&device);

        // create physical sizes
        let logical_input_size = descriptor.input_size;
        let logical_output_size = descriptor.output_size;
        let logical_batch_size = descriptor.max_batch_size;

        let physical_input_size = descriptor.input_size.next_multiple_of(4);
        let physical_output_size = descriptor.output_size.next_multiple_of(4);
        let physical_batch_size = descriptor.max_batch_size.next_multiple_of(4);

        // create calculation buffer
        let max_h_nodes = descriptor.h_layer_sizes.iter().max();

        let max_input = if let Some(max_h_nodes) = max_h_nodes {
            logical_input_size.max(*max_h_nodes)
        } else {
            logical_input_size
        };

        let max_output = if let Some(max_h_nodes) = max_h_nodes {
            logical_output_size.max(*max_h_nodes)
        } else {
            logical_output_size
        };

        let max_nodes = max_input.max(max_output).next_multiple_of(4);

        let global_data = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Global Data"),
            contents: bytemuck::bytes_of(&GlobalData {
                physical_batch_size,
                learning_rate: descriptor.learning_rate,
                logical_batch_size: descriptor.max_batch_size,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1.0e-8,
                timestep: 1,
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let calculation_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Calculation Buffer"),
            size: max_nodes as u64 * physical_batch_size as u64 * F32_SIZE,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let cache_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Multipurpose Caching Buffer"),
            size: max_nodes as u64 * physical_batch_size as u64 * F32_SIZE,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let error_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Error Buffer"),
            size: max_nodes as u64 * physical_batch_size as u64 * F32_SIZE,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let expected_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Expected Values Buffer"),
            size: physical_output_size as u64 * physical_batch_size as u64 * F32_SIZE,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let prefetch_input_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Prefetch Input Buffer"),
            size: max_nodes as u64 * physical_batch_size as u64 * F32_SIZE,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let prefetch_expected_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Prefetch Expected Values Buffer"),
            size: physical_output_size as u64 * physical_batch_size as u64 * F32_SIZE,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let prefetch_global_data = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Prefetch Global Buffer"),
            size: std::mem::size_of::<GlobalData>() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // create layers
        let mut layers = Vec::with_capacity(descriptor.h_layer_sizes.len() + 1);
        let mut cur_input_size = logical_input_size;

        for cur_output_size in &descriptor.h_layer_sizes {
            layers.push(Layer::create(
                &device,
                max_nodes,
                cur_input_size,
                *cur_output_size,
                logical_batch_size,
                false,
            ));

            cur_input_size = *cur_output_size;
        }

        layers.push(Layer::create(
            &device,
            max_nodes,
            cur_input_size,
            logical_output_size,
            logical_batch_size,
            true,
        ));

        Self {
            device,
            queue,
            pipelines,

            layers,
            logical_input_size,
            physical_input_size,
            logical_output_size,
            physical_output_size,
            logical_batch_size,
            physical_batch_size,

            max_size: max_nodes,
            timestep: 1,
            learning_rate: descriptor.learning_rate,

            cur_batch_size: descriptor.max_batch_size,
            next_batch_size: descriptor.max_batch_size,

            global_data,
            prefetch_global_data,

            calculation_buf,
            expected_buf,
            cache_buf,
            error_buf,

            prefetch_input_buf,
            prefetch_expected_buf,
        }
    }

    pub fn load_data(&mut self, input_data: &[f32], expected: &[f32], current_batch_size: u32) {
        assert!(current_batch_size <= self.logical_batch_size);
        assert!(input_data.len() == current_batch_size as usize * self.logical_input_size as usize);
        assert!(expected.len() == current_batch_size as usize * self.logical_output_size as usize);

        self.next_batch_size = current_batch_size;

        self.queue.write_buffer(
            &self.prefetch_global_data,
            0,
            bytemuck::bytes_of(&GlobalData {
                logical_batch_size: current_batch_size,
                physical_batch_size: self.physical_batch_size,
                learning_rate: self.learning_rate,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
                timestep: self.timestep + 1,
            }),
        );

        let mut input_data_padded =
            Vec::with_capacity(self.max_size as usize * self.physical_batch_size as usize);

        let mut expected_data_padded = Vec::with_capacity(
            self.physical_output_size as usize * self.physical_batch_size as usize,
        );

        for y in 0..self.physical_batch_size {
            for x in 0..self.max_size {
                if x >= self.logical_input_size || y >= current_batch_size {
                    input_data_padded.push(0.0);
                } else {
                    input_data_padded.push(
                        input_data[y as usize * self.logical_input_size as usize + x as usize],
                    );
                }
            }
        }

        for y in 0..self.physical_batch_size {
            for x in 0..self.physical_output_size {
                if x >= self.logical_output_size || y >= current_batch_size {
                    expected_data_padded.push(0.0);
                } else {
                    expected_data_padded.push(
                        expected[y as usize * self.logical_output_size as usize + x as usize],
                    );
                }
            }
        }

        // println!("input: {input_data_padded:?}");
        // println!("expected: {expected_data_padded:?}");

        self.queue.write_buffer(
            &self.prefetch_input_buf,
            0,
            bytemuck::cast_slice(&input_data_padded),
        );

        self.queue.write_buffer(
            &self.prefetch_expected_buf,
            0,
            bytemuck::cast_slice(&expected_data_padded),
        );
    }

    pub fn test(&mut self) -> u32 {
        macro_rules! print_buf_u {
            ($buf:expr, $width:expr, $height:expr, $mul:expr, $prefix:expr) => {
                let data_raw = self.read_from_buffer($buf, $width * $height * $mul);
                let data: &[u32] = bytemuck::cast_slice(&data_raw);

                println!("{} [", $prefix);

                for y in 0..$height {
                    println!(
                        "{:?}",
                        &data[(y * $width) as usize..((y + 1) * $width) as usize]
                    );
                }

                println!("]");
            };
        }

        self.cur_batch_size = self.next_batch_size;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        // load prefetched data into calculation buffers
        encoder.copy_buffer_to_buffer(
            &self.prefetch_global_data,
            0,
            &self.global_data,
            0,
            std::mem::size_of::<GlobalData>() as u64,
        );

        encoder.copy_buffer_to_buffer(
            &self.prefetch_input_buf,
            0,
            &self.calculation_buf,
            0,
            self.max_size as u64 * self.physical_batch_size as u64 * F32_SIZE,
        );
        encoder.copy_buffer_to_buffer(
            &self.prefetch_expected_buf,
            0,
            &self.expected_buf,
            0,
            self.physical_output_size as u64 * self.physical_batch_size as u64 * F32_SIZE,
        );

        self.queue.submit(Some(encoder.finish()));

        // get prediction
        self.calculate();

        let raw_data = self.read_from_buffer(
            &self.calculation_buf,
            self.max_size as u64 * self.physical_batch_size as u64 * F32_SIZE,
        );
        let data: &[f32] = bytemuck::cast_slice(&raw_data);

        let raw_labels = self.read_from_buffer(
            &self.expected_buf,
            self.physical_output_size as u64 * self.physical_batch_size as u64 * F32_SIZE,
        );
        let labels: &[f32] = bytemuck::cast_slice(&raw_labels);

        let mut amount = 0;

        for i in 0..self.cur_batch_size {
            let data_idx = i * self.max_size;
            let label_idx = i * self.physical_output_size;

            let mut is_correct = true;

            for offset in 0..self.logical_output_size {
                if (data[(data_idx + offset) as usize] - labels[(label_idx + offset) as usize])
                    .abs()
                    >= 0.5
                {
                    is_correct = false;
                }
            }
            amount += if is_correct { 1 } else { 0 };
        }

        amount
    }

    pub fn train(&mut self) {
        self.cur_batch_size = self.next_batch_size;
        self.timestep += 1;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        // load prefetched data into calculation buffers
        encoder.copy_buffer_to_buffer(
            &self.prefetch_global_data,
            0,
            &self.global_data,
            0,
            std::mem::size_of::<GlobalData>() as u64,
        );
        encoder.copy_buffer_to_buffer(
            &self.prefetch_input_buf,
            0,
            &self.calculation_buf,
            0,
            self.max_size as u64 * self.physical_batch_size as u64 * F32_SIZE,
        );
        encoder.copy_buffer_to_buffer(
            &self.prefetch_expected_buf,
            0,
            &self.expected_buf,
            0,
            self.physical_output_size as u64 * self.physical_batch_size as u64 * F32_SIZE,
        );

        self.queue.submit(Some(encoder.finish()));

        // get prediction
        self.calculate();

        // update weights
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        macro_rules! print_buf {
            ($buf:expr, $width:expr, $height:expr, $mul:expr, $prefix:expr) => {
                let data_raw = self.read_from_buffer($buf, $width * $height * $mul);
                let data: &[f32] = bytemuck::cast_slice(&data_raw);

                println!("{} [", $prefix);

                for y in 0..$height {
                    println!(
                        "{:?}",
                        &data[(y * $width) as usize..((y + 1) * $width) as usize]
                    );
                }

                println!("]");
            };
        }

        self.pipelines
            .dispatch_error_calc(DispatchErrorCalcDescriptor {
                device: &self.device,
                command_encoder: &mut encoder,
                layer: self.layers.last().unwrap(),
                global: &self.global_data,
                calc: &self.calculation_buf,
                expected: &self.expected_buf,
                error: &self.error_buf,
            });

        for (i, layer) in self.layers.iter().enumerate().rev() {
            // calculate delta
            self.pipelines
                .dispatch_delta_calc(DispatchDeltaCalcDescriptor {
                    device: &self.device,
                    command_encoder: &mut encoder,
                    layer,

                    global: &self.global_data,
                    error: &self.error_buf,
                    prev_output: &layer.prev_output,
                    cache: &self.cache_buf,
                });

            // update error
            if i != 0 {
                self.pipelines
                    .dispatch_error_update(DispatchErrorUpdateDescriptor {
                        device: &self.device,
                        command_encoder: &mut encoder,
                        layer,
                        global: &self.global_data,
                        cache: &self.cache_buf,
                        error: &self.error_buf,
                    });
            }

            // update weights
            self.pipelines
                .dispatch_weights_update(DispatchWeightsUpdateDescriptor {
                    device: &self.device,
                    command_encoder: &mut encoder,
                    layer,
                    global: &self.global_data,
                    cache: &self.cache_buf,
                });

            // update bias
            self.pipelines
                .dispatch_bias_update(DispatchBiasUpdateDescriptor {
                    device: &self.device,
                    command_encoder: &mut encoder,
                    layer,
                    global: &self.global_data,
                    cache: &self.cache_buf,
                });
        }

        let a = self.queue.submit(Some(encoder.finish()));

        // Debug: Print all buffers
        // print_buf!(
        //     &self.calculation_buf,
        //     self.max_size as u64,
        //     self.physical_batch_size as u64,
        //     F32_SIZE,
        //     "Result: "
        // );
        //
        // print_buf!(
        //     &self.expected_buf,
        //     self.physical_output_size as u64,
        //     self.physical_batch_size as u64,
        //     F32_SIZE,
        //     "Expected: "
        // );
        //
        //
        // print_buf!(
        //     &self.cache_buf,
        //     self.max_size as u64,
        //     self.physical_batch_size as u64,
        //     F32_SIZE,
        //     "Delta: "
        // );
    }

    fn calculate(&mut self) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        for layer in &self.layers {
            // save current input
            encoder.copy_buffer_to_buffer(
                &self.calculation_buf,
                0,
                &layer.prev_input,
                0,
                self.max_size as u64 * self.physical_batch_size as u64 * F32_SIZE,
            );

            // multiply current input by weights of the layer (AxI * IxO => AxO)
            self.pipelines.dispatch_matmul(DispatchMatmulDescriptor {
                device: &self.device,
                command_encoder: &mut encoder,
                layer,

                global: &self.global_data,
                calc: &self.calculation_buf,
                cache: &self.cache_buf,
            });

            // swap cache and calc_buf
            std::mem::swap(&mut self.calculation_buf, &mut self.cache_buf);

            // add bias (AxI + (1xI * identity_A) => AxI)
            self.pipelines.dispatch_bias_add(DispatchBiasAddDescriptor {
                device: &self.device,
                command_encoder: &mut encoder,
                layer,
                global: &self.global_data,
                calc: &self.calculation_buf,
            });

            // save current output
            encoder.copy_buffer_to_buffer(
                &self.calculation_buf,
                0,
                &layer.prev_output,
                0,
                self.max_size as u64 * self.physical_batch_size as u64 * F32_SIZE,
            );
        }

        // apply the softmax function to the output
        self.pipelines.dispatch_softmax(DispatchSoftmaxDescriptor {
            device: &self.device,
            command_encoder: &mut encoder,
            layer: self.layers.last().unwrap(),
            global: &self.global_data,
            calc: &self.calculation_buf,
        });

        self.queue.submit(Some(encoder.finish()));
    }

    fn read_from_buffer(&self, buf: &wgpu::Buffer, amount: u64) -> Vec<u8> {
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: amount,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        encoder.copy_buffer_to_buffer(buf, 0, &staging_buffer, 0, amount);
        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);

        let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();

        buffer_slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());

        let _ = self.device.poll(wgpu::wgt::PollType::Wait {
            submission_index: None,
            timeout: None,
        });

        pollster::block_on(rx.receive()).unwrap().unwrap();

        let data = buffer_slice.get_mapped_range();

        data.to_vec()
    }
}

pub struct Layer {
    pub weights: wgpu::Buffer,
    pub bias: wgpu::Buffer,

    pub prev_input: wgpu::Buffer,
    pub prev_output: wgpu::Buffer,

    pub w_m_buffer: wgpu::Buffer,
    pub w_v_buffer: wgpu::Buffer,
    pub b_m_buffer: wgpu::Buffer,
    pub b_v_buffer: wgpu::Buffer,

    pub metadata: wgpu::Buffer,

    pub logical_input_size: u32,
    pub physical_input_size: u32,
    pub logical_output_size: u32,
    pub physical_output_size: u32,
    pub logical_batch_size: u32,
    pub physical_batch_size: u32,
}

impl Layer {
    pub fn create(
        device: &wgpu::Device,
        max_size: u32,
        input_size: u32,
        output_size: u32,
        batch_size: u32,
        is_output: bool,
    ) -> Self {
        let physical_input_size = input_size.next_multiple_of(4);
        let physical_output_size = output_size.next_multiple_of(4);
        let physical_batch_size = batch_size.next_multiple_of(4);

        let metadata = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("layer metadata buffer"),
            contents: bytemuck::bytes_of(&LayerMetadata {
                is_output: if is_output { 1 } else { 0 },
                max_size,
                logical_input_size: input_size,
                physical_input_size,
                logical_output_size: output_size,
                physical_output_size,
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_SRC,
        });

        let weights = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("layer weights buffer"),
            contents: bytemuck::cast_slice(&create_random_data(output_size, input_size)),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let bias = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("layer bias buffer"),
            contents: bytemuck::cast_slice(&create_random_data(output_size, 1)),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let prev_input = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("layer prev input buffer"),
            size: max_size as u64 * physical_batch_size as u64 * F32_SIZE,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let prev_output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("layer prev output buffer"),
            size: max_size as u64 * physical_batch_size as u64 * F32_SIZE,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let w_m_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("layer weights m buffer"),
            size: physical_output_size as u64 * physical_input_size as u64 * F32_SIZE,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let w_v_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("layer weights b buffer"),
            size: physical_output_size as u64 * physical_input_size as u64 * F32_SIZE,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let b_m_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("layer bias m buffer"),
            size: physical_output_size as u64 * F32_SIZE,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let b_v_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("layer bias b buffer"),
            size: physical_output_size as u64 * F32_SIZE,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        Self {
            weights,
            bias,
            prev_input,
            prev_output,
            w_m_buffer,
            w_v_buffer,
            b_m_buffer,
            b_v_buffer,
            metadata,
            logical_input_size: input_size,
            physical_input_size,
            logical_output_size: output_size,
            physical_output_size,
            logical_batch_size: batch_size,
            physical_batch_size,
        }
    }
}

pub struct MLPDescriptor {
    pub learning_rate: f32,
    pub input_size: u32,
    pub output_size: u32,
    pub max_batch_size: u32,
    pub h_layer_sizes: Vec<u32>,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GlobalData {
    logical_batch_size: u32,
    physical_batch_size: u32,
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    timestep: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct LayerMetadata {
    is_output: u32,
    max_size: u32,
    logical_input_size: u32,
    physical_input_size: u32,
    logical_output_size: u32,
    physical_output_size: u32,
}

fn create_random_data(width: u32, height: u32) -> Vec<f32> {
    let width_padded = width.next_multiple_of(4);
    let height_padded = height.next_multiple_of(4);
    let mut res = Vec::with_capacity((width_padded * height_padded) as usize);

    for y in 0..height_padded {
        for x in 0..width_padded {
            if y >= height || x >= width {
                res.push(0.0);
            } else {
                res.push(rand::random_range(-0.5..0.5));
            }
        }
    }

    res
}
