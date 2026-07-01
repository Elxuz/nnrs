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

@group(0) @binding(2) var<storage, read> cache: array<f32>;       // is size p_batch_size x max_size;       data in l_batch_size x l_output_size   
@group(0) @binding(3) var<storage, read_write> bias: array<f32>;  // is size 4 x p_output_size;             data in 1 x l_output_size 

@group(0) @binding(4) var<storage, read_write> bias_m: array<f32>;// is size 4 x p_output_size;             data in 1 x l_output_size 
@group(0) @binding(5) var<storage, read_write> bias_v: array<f32>;// is size 4 x p_output_size;             data in 1 x l_output_size 

@compute @workgroup_size(64, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let col = global_id.x;

    if col >= metadata.logical_output_size {
        return;
    }

    let t_float = f32(global_data.timestep);
    let bias_correction1 = 1.0 - pow(global_data.beta1, t_float);
    let bias_correction2 = 1.0 - pow(global_data.beta2, t_float);

    var delta_bias = 0.0;
    for (var k = 0u; k < global_data.logical_batch_size; k = k + 1) {
        delta_bias = delta_bias + cache[k * metadata.max_size + col];
    }

    delta_bias = delta_bias / f32(global_data.logical_batch_size);

    // bias[col] = bias[col] - delta_bias * global_data.learning_rate;

    bias_m[col] = global_data.beta1 * bias_m[col] + (1.0 - global_data.beta1) * delta_bias;
    bias_v[col] = global_data.beta2 * bias_v[col] + (1.0 - global_data.beta2) * (delta_bias * delta_bias);

    let m_hat = bias_m[col] / bias_correction1;
    let v_hat = bias_v[col] / bias_correction2;

    bias[col] = bias[col] - (global_data.learning_rate / (sqrt(v_hat) + global_data.epsilon)) * m_hat;
}
