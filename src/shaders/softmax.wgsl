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
@group(0) @binding(2) var<storage, read_write> calculation: array<f32>; // is size p_batch_size x max_size;       data in l_batch_size x l_output_size 

@compute @workgroup_size(64, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;

    if row >= global_data.logical_batch_size {
        return;
    }

    let row_offset = row * metadata.max_size;

    var max = -3.40282327e+38f;
    for (var x = 0u; x < metadata.logical_output_size; x = x + 1u) {
        let val = calculation[row_offset + x];
        if val > max {
            max = val;
        }
    }

    var sum = 0.0;
    for (var x = 0u; x < metadata.logical_output_size; x = x + 1u) {
        let val = exp(calculation[row_offset + x] - max);
        sum = sum + val;
    }

    for (var x = 0u; x < metadata.logical_output_size; x = x + 1u) {
        let val = exp(calculation[row_offset + x] - max);

        calculation[row_offset + x] = val / sum;
    }
}
