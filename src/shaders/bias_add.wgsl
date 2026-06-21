struct BiasUniforms {
    rows: u32,
    cols: u32,
}

@group(0) @binding(0) var<uniform> uniforms: BiasUniforms;
@group(0) @binding(1) var<storage, read_write> matrix: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.y;
    let col = global_id.x;

    if row >= uniforms.rows || col >= uniforms.cols {
        return;
    }

    let idx = row * uniforms.cols + col;

    matrix[idx] = matrix[idx] + bias[col];
}
