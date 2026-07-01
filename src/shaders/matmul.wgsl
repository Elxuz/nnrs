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

// Shared memory sized exactly to our 16x16 workgroup execution path
var<workgroup> tile_calc: array<array<vec4<f32>, 4>, 16>;
var<workgroup> tile_weights: array<array<vec4<f32>, 4>, 16>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let row = global_id.y;
    let col = global_id.x;

    let lrow = local_id.y;
    let lcol = local_id.x;

    let calc_stride = metadata.max_size / 4u;
    let weights_stride = metadata.physical_output_size / 4u;
    let cache_stride = calc_stride;

    let num_tiles = (metadata.logical_input_size + 15u) / 16u;

    var sum = 0.0;

    for (var tile = 0u; tile < num_tiles; tile = tile + 1u) {
        if lcol < 4 {
            let tile_off_x = tile * 4u + lcol;
            tile_calc[lrow][lcol] = calculation[row * calc_stride + tile_off_x];

            let tile_off_y = tile * 16u + lrow;
            tile_weights[lrow][lcol] = weights[tile_off_y * weights_stride + col / 4 + lcol];
        }

        workgroupBarrier();

        for (var k = 0u; k < 4u; k = k + 1u) {
            let vec1 = tile_calc[lrow][k];
            let vec2 = vec4<f32>(
                tile_weights[k * 4][lcol / 4][lcol % 4],
                tile_weights[k * 4 + 1][lcol / 4][lcol % 4],
                tile_weights[k * 4 + 2][lcol / 4][lcol % 4],
                tile_weights[k * 4 + 3][lcol / 4][lcol % 4],
            );
            sum = sum + dot(vec1, vec2);
        }

        workgroupBarrier();
    }

    if row < global_data.logical_batch_size && col < metadata.logical_output_size {
        cache[row * cache_stride + col / 4][col % 4] = sum;
    }
}
