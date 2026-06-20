struct MatrixUniforms {
    a_rows: u32,
    a_cols: u32,
    b_cols: u32,
    padding: u32,
}

@group(0) @binding(0) var<uniform> uniforms: MatrixUniforms;
@group(0) @binding(1) var<storage, read> matrix_a: array<f32>;
@group(0) @binding(2) var<storage, read> matrix_b: array<f32>;
@group(0) @binding(3) var<storage, read_write> matrix_out: array<f32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.y;
    let col = global_id.x;

    if row >= uniforms.a_rows || col >= uniforms.b_cols {
        return;
    }

    var sum = 0.0;
    for (var k = 0u; k < uniforms.a_cols; k = k + 1u) {
        let index_a = row * uniforms.a_cols + k;
        let index_b = k * uniforms.b_cols + col;
        sum = sum + matrix_a[index_a] * matrix_b[index_b];
    }

    let out_idx = row * uniforms.b_cols + col;
    matrix_out[out_idx] = sum;
}
