struct DeltaUniforms {
    size: u32,
    is_output: u32,
}

@group(0) @binding(0) var<uniform> uniforms: DeltaUniforms;
@group(0) @binding(1) var<storage, read> error_grad: array<f32>;
@group(0) @binding(2) var<storage, read> prev_output: array<f32>;
@group(0) @binding(3) var<storage, read_write> delta_out: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if idx >= uniforms.size {
        return;
    }

    if uniforms.is_output == 1u || prev_output[idx] > 0.0 {
        delta_out[idx] = error_grad[idx];
    } else {
        delta_out[idx] = 0.0;
    }
}
