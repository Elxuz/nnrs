struct SoftmaxUniforms {
    rows: u32,
    cols: u32,
}

@group(0) @binding(0) var<uniform> uniforms: SoftmaxUniforms;
@group(0) @binding(1) var<storage, read_write> matrix: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;

    if row >= uniforms.rows {
        return;
    }

    let row_offset = row * uniforms.cols;

    var max_val = -3.40282347e+38f;
    for (var j = 0u; j < uniforms.cols; j = j + 1u) {
        let val = matrix[row_offset + j];
        if val > max_val {
            max_val = val;
        }
    }

    var sum = 0.0;
    for (var j = 0u; j < uniforms.cols; j = j + 1u) {
        let exp_val = exp(matrix[row_offset + j] - max_val);
        sum = sum + exp_val;
    }

    for (var j = 0u; j < uniforms.cols; j = j + 1u) {
        let idx = row_offset + j;
        matrix[idx] = exp(matrix[idx] - max_val) / sum;
    }
}
