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

@group(0) @binding(2) var<storage, read> calculation: array<vec4<f32>>; // is size p_batch_size x max_size;       data in l_batch_size x l_input_size
@group(0) @binding(3) var<storage, read> weights: array<vec4<f32>>;     // is size p_input_size x p_output_size;  data in l_input_size x l_output_size
@group(0) @binding(4) var<storage, read_write> cache: array<vec4<f32>>; // is size p_batch_size x max_size;       data in l_batch_size x l_output_size

var<workgroup> tile_calc: array<array<mat4x4<f32>, 8>, 8>;
var<workgroup> tile_weights: array<array<mat4x4<f32>, 8>, 8>;

// goal is to calculate calculation * weights, since wgpu stores data column-wise, this shader is making use of the identity A*B = (B^T * A^T)^T
// 8x8 threads to compute a 32x32 submatrix
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    // start of submatrix in result cache
    let col = global_id.x;     // cache has vec4<f32>, so no '* 4' necessary
    let row = global_id.y * 4;

    // used to index the 4x4 submatrices inside the tiles
    let lcol = local_id.x;
    let lrow = local_id.y;

    // row and col describe indices of 4x4 matrices, therefore they need to multiplied by 4 to get the actual lengths
    if global_id.y * 4u >= global_data.physical_batch_size || global_id.x * 4u >= metadata.max_size {
        return;
    }

    // strides
    let calc_stride = metadata.max_size / 4u;
    let weights_stride = metadata.physical_output_size / 4u;
    let cache_stride = metadata.max_size / 4u;

    // initialize a 4x4 sum submatrix
    var sum = mat4x4<f32>(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0));

    let num_tiles = (metadata.max_size + 31u) / 32u;

    for (var tile = 0u; tile < num_tiles; tile = tile + 1u) {
        let tile_x = 8 * tile + lcol;        // each tile skips 8 vec4s => 32 floats
        let tile_y = 4 * (8 * tile + lrow);  // each tile skips 32 floats

        // load the data into the tiles
        tile_calc[lrow][lcol] = mat4x4<f32>(
            calculation[(row + 0) * calc_stride + tile_x],
            calculation[(row + 1) * calc_stride + tile_x],
            calculation[(row + 2) * calc_stride + tile_x],
            calculation[(row + 3) * calc_stride + tile_x],
        );

        tile_weights[lrow][lcol] = mat4x4<f32>(
            weights[(tile_y + 0) * weights_stride + col],
            weights[(tile_y + 1) * weights_stride + col],
            weights[(tile_y + 2) * weights_stride + col],
            weights[(tile_y + 3) * weights_stride + col],
        );

        workgroupBarrier();

        for (var k = 0u; k < 8u; k = k + 1u) {
            // identity is used here
            sum = sum + tile_weights[k][lcol] * tile_calc[lrow][k];
        }

        workgroupBarrier();
    }

    cache[(row + 0) * cache_stride + col] = sum[0];
    cache[(row + 1) * cache_stride + col] = sum[1];
    cache[(row + 2) * cache_stride + col] = sum[2];
    cache[(row + 3) * cache_stride + col] = sum[3];
}

