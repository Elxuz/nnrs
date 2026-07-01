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

@group(0) @binding(3) var<storage, read> prev_input: array<f32>;      // is size p_batch_size x max_size;       data in l_batch_size x l_input_size 
@group(0) @binding(2) var<storage, read> cache: array<f32>;           // is size p_batch_size x max_size;       data in l_batch_size x l_output_size 
@group(0) @binding(4) var<storage, read_write> weights: array<f32>;   // is size p_input_size x p_output_size;  data in l_input_size x l_output_size 

@group(0) @binding(5) var<storage, read_write> weights_m: array<f32>; // is size p_input_size x p_output_size;  data in l_input_size x l_output_size 
@group(0) @binding(6) var<storage, read_write> weights_v: array<f32>; // is size p_input_size x p_output_size;  data in l_input_size x l_output_size 

var<workgroup> tile_cache: array<array<f32, 16>, 16>;
var<workgroup> tile_prev: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    let row = global_id.y;
    let col = global_id.x;

    let lrow = local_id.y;
    let lcol = local_id.x;

    let tiles = (global_data.logical_batch_size + 15u) / 16u;

    var delta_weight = 0.0;
    for (var tile = 0u; tile < tiles; tile = tile + 1u) {
        // load data into tiles
        let tile_x = tile * 16u + lcol;
        let tile_y = tile * 16u + lrow;

        if row < metadata.logical_input_size && tile_x < global_data.logical_batch_size {
            tile_prev[lrow][lcol] = prev_input[tile_x * metadata.max_size + row];
        } else {
            tile_prev[lrow][lcol] = 0.0;
        }

        if col < metadata.logical_output_size && tile_y < global_data.logical_batch_size {
            tile_cache[lrow][lcol] = cache[tile_y * metadata.max_size + col];
        } else {
            tile_cache[lrow][lcol] = 0.0;
        }

        // use tiles to calculate
        workgroupBarrier();
        for (var k = 0u; k < 16; k = k + 1u) {
            delta_weight = delta_weight + tile_prev[lrow][k] * tile_cache[k][lcol];
        }

        workgroupBarrier();
    }
    // for (var k = 0u; k < global_data.logical_batch_size; k = k + 1u) {
    // delta_weight = delta_weight + prev_input[k * metadata.max_size + row] * cache[k * metadata.max_size + col];
    // }

    if row >= metadata.logical_input_size || col >= metadata.logical_output_size {
        return;
    }

    let t_float = f32(global_data.timestep);
    let bias_correction1 = 1.0 - pow(global_data.beta1, t_float);
    let bias_correction2 = 1.0 - pow(global_data.beta2, t_float);

    delta_weight = delta_weight / f32(global_data.logical_batch_size);

    let idx = row * metadata.physical_output_size + col;

    // if delta_weight > 1.0 {
    // delta_weight = 1.0;
    // } else if delta_weight < -1.0 {
    // delta_weight = -1.0;
    // }
    // weights[idx] = weights[idx] - delta_weight * global_data.learning_rate;

    weights_m[idx] = global_data.beta1 * weights_m[idx] + (1.0 - global_data.beta1) * delta_weight;
    weights_v[idx] = global_data.beta2 * weights_v[idx] + (1.0 - global_data.beta2) * (delta_weight * delta_weight);

    let m_hat = weights_m[idx] / bias_correction1;
    let v_hat = weights_v[idx] / bias_correction2;

    weights[idx] = weights[idx] - (global_data.learning_rate / (sqrt(v_hat) + global_data.epsilon)) * m_hat;
}
