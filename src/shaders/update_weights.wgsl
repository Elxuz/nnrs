struct UpdateUniforms {
    batch_size: u32,
    input_nodes: u32,
    output_nodes: u32,
}

struct LearningState {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    timestep: u32,
}

@group(0) @binding(0) var<uniform> uniforms: UpdateUniforms;
@group(0) @binding(1) var<uniform> learning_state: LearningState;

@group(0) @binding(2) var<storage, read> prev_input: array<f32>;
@group(0) @binding(3) var<storage, read> delta: array<f32>;

@group(0) @binding(4) var<storage, read_write> weights: array<f32>;
@group(0) @binding(5) var<storage, read_write> bias: array<f32>;

@group(0) @binding(6) var<storage, read_write> weights_m: array<f32>;
@group(0) @binding(7) var<storage, read_write> weights_v: array<f32>;

@group(0) @binding(8) var<storage, read_write> bias_m: array<f32>;
@group(0) @binding(9) var<storage, read_write> bias_v: array<f32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.y;
    let col = global_id.x;

    let idx = row * uniforms.output_nodes + col;

    let t_float = f32(learning_state.timestep);
    let bias_correction1 = 1.0 - pow(learning_state.beta1, t_float);
    let bias_correction2 = 1.0 - pow(learning_state.beta2, t_float);

    // weights
    if row < uniforms.input_nodes && col < uniforms.output_nodes {
        var weight_grad = 0.0;

        for (var i = 0u; i < uniforms.batch_size; i = i + 1u) {
            let input_idx = i * uniforms.input_nodes + row;
            let delta_idx = i * uniforms.output_nodes + col;
            weight_grad = weight_grad + prev_input[input_idx] * delta[delta_idx];
        }

        // constant learn rate scaling
        // if weight_grad > 1.0 { weight_grad = 1.0; }
        // if weight_grad < -1.0 { weight_grad = -1.0; }

        // let weight_idx = row * uniforms.output_nodes + col;
        // weights[weight_idx] = weights[weight_idx] - (weight_grad * learning_state.learning_rate);

        // dynamic learn rate scaling
        weights_m[idx] = learning_state.beta1 * weights_m[idx] + (1.0 - learning_state.beta1) * weight_grad;
        weights_v[idx] = learning_state.beta2 * weights_v[idx] + (1.0 - learning_state.beta2) * (weight_grad * weight_grad);

        let m_hat = weights_m[idx] / bias_correction1;
        let v_hat = weights_v[idx] / bias_correction2;

        weights[idx] = weights[idx] - (learning_state.learning_rate / (sqrt(v_hat) + learning_state.epsilon)) * m_hat;
    }

    // bias (only first row)
    if row == 0u && col < uniforms.output_nodes {
        var bias_grad = 0.0;

        for (var i = 0u; i < uniforms.batch_size; i = i + 1u) {
            bias_grad = bias_grad + delta[i * uniforms.output_nodes + col];
        }

        // constant learn rate scaling
        // bias[col] = bias[col] - (bias_grad * learning_state.learning_rate);

        // dynamic learn rate scaling
        bias_m[col] = learning_state.beta1 * bias_m[col] + (1.0 - learning_state.beta1) * bias_grad;
        bias_v[col] = learning_state.beta2 * bias_v[col] + (1.0 - learning_state.beta2) * (bias_grad * bias_grad);

        let b_m_hat = bias_m[col] / bias_correction1;
        let b_v_hat = bias_v[col] / bias_correction2;

        bias[col] = bias[col] - (learning_state.learning_rate / (sqrt(b_v_hat) + learning_state.epsilon)) * b_m_hat;
    }
}
