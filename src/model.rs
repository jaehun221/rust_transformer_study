use crate::config::Config;
use crate::tensor:: { gelu, layer_norm, matmul, softmax };
use crate::sampling::argmax;
use crate::weights::load_gpt2;

pub struct AttentionWeights { // multihead-attention 단계에서 사용
    pub c_attn: Vec<f32>,
    pub c_attn_bias: Vec<f32>,
    pub c_proj: Vec<f32>,
    pub c_proj_bias: Vec<f32>,
}

pub struct MlpWeights { // Feed-Forward-Network 역할
    pub c_fc: Vec<f32>,
    pub c_fc_bias: Vec<f32>,
    pub c_proj: Vec<f32>,
    pub c_proj_bias: Vec<f32>,
}

pub struct LayerNormWeights { // 정규화 할때 표준편차를 구한 후 곱해주는 가중치
    pub weight: Vec<f32>, // gamma
    pub bias: Vec<f32>, // beta
}

pub struct Layer { // 층마다 수행하는 연산
    pub ln_1: LayerNormWeights,
    pub ln_2: LayerNormWeights,
    pub attn: AttentionWeights,
    pub mlp: MlpWeights,
}

pub struct Gpt2 {
    pub config: Config,
    pub wte: Vec<f32>, // Word Token Embedding
    pub wpe: Vec<f32>, // Word Position Embedding
    pub layers: Vec<Layer>,
    pub ln_f: LayerNormWeights,
}

impl Gpt2 { // forward pass 구현
    pub fn load(config_path: &str, weight_path: &str) -> Self {
        load_gpt2(config_path, weight_path)
    }

    pub fn forward_hidden(&self, tokens: &[usize]) -> Vec<f32> {
        // self.validate_tokens(tokens); // 나중에 토큰 검증용

        let seq_len = tokens.len();
        let n_embd = self.config.n_embd;

        // token embedding + position embedding
        let mut hidden_states = self.embed_tokens(tokens);

        // Layer Nomalization
        for layer in &self.layers {
            self.run_attention(layer, &mut hidden_states, seq_len);
            self.run_mlp(layer, &mut hidden_states, seq_len);
        }

        let mut final_hidden = vec![0.0; seq_len * n_embd];

        for pos in 0..seq_len {
            let start = pos * n_embd;
            let end = start + n_embd;

            layer_norm(
                &mut final_hidden[start..end],
                &hidden_states[start..end],
                &self.ln_f.weight,
                &self.ln_f.bias,
                self.config.layer_norm_epsilon,
            );
        }

        final_hidden

    }
    
    pub fn last_logits(&self, tokens: &[usize]) -> Vec<f32> {
        let hidden = self.forward_hidden(tokens);

        let seq_len = tokens.len();
        let n_embd = self.config.n_embd;
        let vocab_size = self.config.vocab_size;

        let last_start = (seq_len - 1) * n_embd;
        let last_hidden = &hidden[last_start..last_start + n_embd];

        let mut logits = vec![0.0; vocab_size];

        for token_id in 0..vocab_size {
            let mut sum = 0.0;
            for d in 0..n_embd {
                sum += last_hidden[d] * self.wte[token_id * n_embd + d];
            }
            logits[token_id] = sum;
        }

        logits
    }

    pub fn generate_greedy(
        &self,
        mut tokens: Vec<usize>,
        max_new_tokens: usize,
        eos_token: Option<usize>,
    ) -> Vec<usize> {
        for _ in 0..max_new_tokens {
            let logits = self.last_logits(&tokens);
            let next = argmax(&logits);

            tokens.push(next);

            if Some(next) == eos_token {
                break;
            }

            if tokens.len() >= self.config.n_positions {
                break;
            }
        }

        tokens
    }

    // token id를 hidden vector로 변환
    // hidden[pos] = wte[token] + wpe[pos]
    fn embed_tokens(&self, tokens: &[usize]) -> Vec<f32> {
        let seq_len = tokens.len();
        let n_embd = self.config.n_embd;

        let mut hidden_states = vec![0.0; seq_len * n_embd];

        for (pos, &token) in tokens.iter().enumerate() {
            let wte_start = token * n_embd;
            let wpe_start = pos * n_embd;
            let out_start = pos * n_embd;

            for i in 0..n_embd {
                hidden_states[out_start + i] = self.wte[wte_start + i] + self.wpe[wpe_start + i];
            }
        }

        hidden_states
    }

    fn run_attention(&self, layer: &Layer, hidden_states: &mut [f32], seq_len: usize) {
        let n_embd = self.config.n_embd;
        let mut norm_x = vec![0.0; seq_len * n_embd];
        
        for pos in 0..seq_len {
            let start = pos * n_embd;
            let end = start + n_embd;

            layer_norm(
                &mut norm_x[start..end],
                &hidden_states[start..end],
                &layer.ln_1.weight,
                &layer.ln_1.bias,
                self.config.layer_norm_epsilon
            );
        }

        let mut qkv = vec![0.0; seq_len * n_embd * 3]; // 단어수(seq_len) * embedding 길이만큼 배열 생성

        matmul(
            &mut qkv,
            &norm_x,
            &layer.attn.c_attn,
            &layer.attn.c_attn_bias,
            seq_len,
            n_embd,
            n_embd * 3,
        );

        let n_head = self.config.n_head;
        let head_dim = n_embd / n_head;

        let scale = 1.0 / (head_dim as f32).sqrt();

        let mut scores = vec![0.0; seq_len];
        let mut attn_out = vec![0.0; seq_len * n_embd];
        for h in 0..n_head {

            for i in 0..seq_len {
                for j in 0..=i {
                    let mut dot = 0.0;
                    
                    for d in 0..head_dim {
                        let q_idx = i * (n_embd * 3) + h * head_dim + d;
                        let k_idx = j * (n_embd * 3) + n_embd + h * head_dim + d;

                        dot += qkv[q_idx] * qkv[k_idx]; // dot product
                    }

                    scores[j] = dot * scale;
                }

                softmax(&mut scores[0..=i]); // 점수를 확률로 변환

                for j in 0..=i {
                    let prob = scores[j];

                    for d in 0..head_dim {
                        let v_idx = j * (n_embd * 3) + (n_embd * 2) + h * head_dim +d;
                        let out_idx = i * n_embd + h * head_dim + d;

                        attn_out[out_idx] += prob * qkv[v_idx];
                    }
                }
            }
        }

        let mut proj_out = vec![0.0; seq_len * n_embd];

        matmul(
            &mut proj_out,
            &attn_out,
            &layer.attn.c_proj,
            &layer.attn.c_proj_bias,
            seq_len,
            n_embd,
            n_embd,
        );

        for i in 0..(seq_len * n_embd) {
            hidden_states[i] += proj_out[i];
        }
    }

    fn run_mlp(&self, layer: &Layer, hidden_states: &mut [f32], seq_len: usize) {
    let n_embd = self.config.n_embd;

    // MLP 입력: norm_x2 = ln_2(hidden_states)
    // Attention이 끝난 hidden_states를 LayerNorm
    let mut norm_x2 = vec![0.0; seq_len * n_embd];

    for pos in 0..seq_len {
        let start = pos * n_embd;
        let end = start + n_embd;

        layer_norm(
            &mut norm_x2[start..end],
            &hidden_states[start..end],
            &layer.ln_2.weight,
            &layer.ln_2.bias,
            self.config.layer_norm_epsilon,
        );
    }

    // GPT-2 small 기준 보통 n_inner = 4 * n_embd
    // c_fc_bias 길이를 사용하면 config에 n_inner가 없어도 동작한다.
    let n_inner = layer.mlp.c_fc_bias.len();

    // 첫 번째 MLP projection
    // [seq_len, n_embd] -> [seq_len, n_inner]
    let mut fc_out = vec![0.0; seq_len * n_inner];

    matmul(
        &mut fc_out,
        &norm_x2,
        &layer.mlp.c_fc,
        &layer.mlp.c_fc_bias,
        seq_len,
        n_embd,
        n_inner,
    );

    // GELU activation
    for v in &mut fc_out {
        *v = gelu(*v);
    }

    // 두 번째 MLP projection
    // [seq_len, n_inner] -> [seq_len, n_embd]
    let mut mlp_out = vec![0.0; seq_len * n_embd];

    matmul(
        &mut mlp_out,
        &fc_out,
        &layer.mlp.c_proj,
        &layer.mlp.c_proj_bias,
        seq_len,
        n_inner,
        n_embd,
    );

    // residual connection
    // hidden_states = hidden_states + mlp(ln_2(hidden_states))
    for i in 0..(seq_len * n_embd) {
        hidden_states[i] += mlp_out[i];
    }
}
}