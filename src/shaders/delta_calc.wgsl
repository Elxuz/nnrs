struct GlobalData {
    logical_batch_size: u32,
    physical_batch_size: u32,
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    timestep: u32,
}

struct LayerMetadata {
    is_output: u32,
    max_size: u32,
    logical_input_size: u32,
    physical_input_size: u32,
    logical_output_size: u32,
    physical_output_size: u32,
}

@group(0) @binding(0) var<uniform> global_data: GlobalData;
@group(0) @binding(1) var<uniform> metadata: LayerMetadata;

@group(0) @binding(2) var<storage, read> error: array<f32>;       // is size p_batch_size x max_size;       data in l_batch_size x l_output_size 
@group(0) @binding(3) var<storage, read> prev_output: array<f32>; // is size p_batch_size x max_size;       data in l_batch_size x l_output_size 
@group(0) @binding(4) var<storage, read_write> cache: array<f32>; // is size p_batch_size x max_size;       data in l_batch_size x l_output_size 

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.y;
    let col = global_id.x;

    if row >= global_data.logical_batch_size || col >= metadata.logical_output_size {
        return;
    }

    let idx = row * metadata.max_size + col;
    if prev_output[idx] > 0.0 || metadata.is_output == 1u {
        cache[idx] = error[idx];
    } else {
        cache[idx] = 0.0;
    }
}
