// 정규화 함수
pub fn layer_norm(
    out: &mut [f32],
    x: &[f32],
    weight: &[f32],
    bias: &[f32],
    eps: f32, // 0으로 나누기 방지
) {
    let n_embd = x.len(); // 768

    // Mean
    let mut sum = 0.0;
    for &val in x.iter() {
        sum += val;
    }
    let mean = sum / n_embd as f32;

    // Variance
    let mut var_sum = 0.0;
    for &val in x.iter() {
        let diff = val - mean;
        var_sum += diff * diff;
    }
    let variance = var_sum / n_embd as f32;

    let inv_std = 1.0 / (variance + eps).sqrt(); // 표준편차의 역수(나눗셈은 비쌈)

    for i in 0..n_embd {
        let normalized = (x[i] - mean) * inv_std;

        out[i] = (normalized * weight[i]) + bias[i];
    }
}

pub fn softmax(x: &mut [f32]) {
    let n = x.len();
    if n == 0 { return; }

    let mut max_val = x[0];
    for &val in x.iter() {
        if val > max_val {
            max_val = val;
        }
    }

    let mut sum = 0.0;
    for i in 0..n {
        x[i] = (x[i] - max_val).exp();
        sum += x[i];
    }

    let inv_sum = 1.0 / sum;
    for i in 0..n {
        x[i] *= inv_sum;
    }
}

// 행렬곱 함수
// m: 단어 수(seq_len), k: input 크기 768, n: qkv를 합친 출력 차원 2304
pub fn matmul(
    c: &mut [f32],
    a: &[f32],
    b: &[f32],
    bias: &[f32],
    m: usize,
    k: usize,
    n: usize
) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = bias[j]; // 연산 수를 최대한 줄이기 위해 bias에서 시작(나중에 어차피 더할거 그냥 여기부터 시작하는 것)
            for l in 0..k {
                sum += a[i * k + l] * b[l * n + j]; // 메모리 효율때문에 1차원 배열을 사용
            }
            c[i * n + j] = sum;
        }
    }
}

// 공부중
pub fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.7978845608028654 * (x + 0.044715 * x * x * x)).tanh())
}