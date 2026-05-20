use serde::Deserialize; // JSON이랑 struct 맵핑할 때 사용

#[derive(Deserialize, Debug)]
pub struct Config { // config.json과 mapping됨
    pub vocab_size: usize,
    pub n_positions: usize,
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub layer_norm_epsilon: f32,
}