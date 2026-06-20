struct UpdateUniforms {
    batch_size: u32,
    input_nodes: u32,
    output_nodes: u32,
    learning_rate: f32,
}

@group(0) @binding(0) var<uniform> uniforms: UpdateUniforms;
@group(0) @binding(1) var<storage, read> prev_input: array<f32>;
@group(0) @binding(2) var<storage, read> delta: array<f32>;
@group(0) @binding(3) var<storage, read_write> weights: array<f32>;
@group(0) @binding(4) var<storage, read_write> bias: array<f32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.y;
    let col = global_id.x;

    // weights
    if row < uniforms.input_nodes && col < uniforms.output_nodes {
        var weight_grad = 0.0;

        for (var i = 0u; i < uniforms.batch_size; i = i + 1u) {
            let input_idx = i * uniforms.input_nodes + row;
            let delta_idx = i * uniforms.output_nodes + col;
            weight_grad = weight_grad + prev_input[input_idx] * delta[delta_idx];
        }

        if weight_grad > 1.0 { weight_grad = 1.0; }
        if weight_grad < -1.0 { weight_grad = -1.0; }

        let weight_idx = row * uniforms.output_nodes + col;
        weights[weight_idx] = weights[weight_idx] - (weight_grad * uniforms.learning_rate);
    }

    // bias (only first row)
    if row == 0u && col < uniforms.output_nodes {
        var bias_grad = 0.0;

        for (var i = 0u; i < uniforms.batch_size; i = i + 1u) {
            bias_grad = bias_grad + delta[i * uniforms.output_nodes + col];
        }

        bias[col] = bias[col] - (bias_grad * uniforms.learning_rate);
    }
}
