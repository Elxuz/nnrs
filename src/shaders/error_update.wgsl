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
@group(0) @binding(3) var<storage, read> weights: array<f32>;     // is size p_input_size x p_output_size;  data in l_input_size x l_output_size 
@group(0) @binding(4) var<storage, read_write> error: array<f32>; // is size p_batch_size x max_size;       data in l_batch_size x l_input_size

var<workgroup> tile_cache: array<array<f32, 16>, 16>;
var<workgroup> tile_weights: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    let row = global_id.y;
    let col = global_id.x;

    let lrow = local_id.y;
    let lcol = local_id.x;

    let tiles = (metadata.logical_output_size + 15u) / 16u;

    var sum = 0.0;

    for (var tile = 0u; tile < tiles; tile = tile + 1u) {
        // load data into tiles
        let tile_x = tile * 16u + lcol;
        let tile_y = tile * 16u + lrow;

        if row < global_data.logical_batch_size && tile_x < metadata.logical_output_size {
            tile_cache[lrow][lcol] = cache[row * metadata.max_size + tile_x];
        } else {
            tile_cache[lrow][lcol] = 0.0;
        }

        if col < metadata.logical_input_size && tile_y < metadata.logical_output_size {
            tile_weights[lcol][lrow] = weights[col * metadata.physical_output_size + tile_y];
        } else {
            tile_weights[lcol][lrow] = 0.0;
        }

        // use tiles to calculate
        workgroupBarrier();
        for (var k = 0u; k < 16; k = k + 1u) {
            sum = sum + tile_cache[lrow][k] * tile_weights[lcol][k];
        }

        workgroupBarrier();
    }

    if row < global_data.logical_batch_size && col < metadata.logical_input_size {
        error[row * metadata.max_size + col] = sum;
    }

    // if row >= global_data.logical_batch_size || col >= metadata.logical_input_size {
    // return;
    // }
    // 
    // for (var k = 0u; k < metadata.logical_output_size; k = k + 1u) {
    // sum = sum + cache[row * metadata.max_size + k] * weights[col * metadata.physical_output_size + k];
    // }
    // 
    // error[row * metadata.max_size + col] = sum;
}
