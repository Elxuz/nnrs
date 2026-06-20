struct ErrorUniform {
    size: u32,
    batch_size: f32,
}

@group(0) @binding(0) var<uniform> uniforms: ErrorUniform;
@group(0) @binding(1) var<storage, read> predictions: array<f32>;
@group(0) @binding(2) var<storage, read> expected: array<f32>;
@group(0) @binding(3) var<storage, read_write> error_out: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if idx >= uniforms.size {
        return;
    }

    error_out[idx] = (predictions[idx] - expected[idx]) / uniforms.batch_size;
}
