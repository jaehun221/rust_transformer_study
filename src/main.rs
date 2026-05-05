use memmap2::MmapOptions;
use serde::Deserialize;
use std::{fmt::format, fs::File};
use safetensors::SafeTensors;

#[derive(Deserialize, Debug)]
struct Config {
    vocab_size: usize,
    n_positions: usize,
    n_embd: usize,
    n_layer: usize,
    n_head: usize,
    layer_norm_epsilon: f32,
}

struct AttentionWeights {
    c_attn: Vec<f32>,
    c_proj: Vec<f32>,
}

struct MlpWeights { // Feed-Forward-Network 역할
    c_fc: Vec<f32>,
    c_proj: Vec<f32>,
}

struct Layer {
    attn: AttentionWeights,
    mlp: MlpWeights,
}

struct Gpt2 {
    config: Config,
    wte: Vec<f32>, // Word Token Embedding
    wpe: Vec<f32>, // Word Position Embedding
    layers: Vec<Layer>,
}

impl Gpt2 {
    pub fn forward(&self, tokens: &[usize]) -> Vec<f32> {
        let seq_len = tokens.len();
        let n_embd = self.config.n_embd;

        let mut hidden_states = vec![0.0; seq_len * n_embd];

        for (pos, &token) in tokens.iter().enumerate() {

            let wte_start = token * n_embd;
            let wte_slice = &self.wte[wte_start .. wte_start + n_embd];

            let wpe_start = pos * n_embd;
            let wpe_slice = &self.wpe[wpe_start .. wpe_start + n_embd];

            let out_start = pos * n_embd;

            for i in 0..n_embd {
                hidden_states[out_start + i] = wte_slice[i] + wpe_slice[i];
            }
        }

        hidden_states
    }
}

fn get_tensor(tensors: &SafeTensors, name: &str) -> Vec<f32> {
    let tensor = tensors.tensor(name).expect("weight not found");
    tensor.data()
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("slice failed")))
        .collect()
}

fn main() {
    let file = File::open("models/config.json").expect("File open Err");
    let config: Config = serde_json::from_reader(file).expect("JSON Err");

    let file = File::open("models/model.safetensors").expect("weight file Err");

    let mmap = unsafe { MmapOptions::new().map(&file).expect("Memory Mapping Failed") };
    let tensors = SafeTensors::deserialize(&mmap).expect("SafeTensors 로드 실패");

    let mut layers = Vec::new();
    for i in 0..config.n_layer {
        let attn = AttentionWeights {
            c_attn: get_tensor(&tensors, &format!("h.{}.attn.c_attn.weight", i)),
            c_proj: get_tensor(&tensors, &format!("h.{}.attn.c_proj.weight", i)),
        };

        let mlp = MlpWeights {
            c_fc: get_tensor(&tensors, &format!("h.{}.mlp.c_fc.weight", i)),
            c_proj: get_tensor(&tensors, &format!("h.{}.mlp.c_proj.weight", i)),
        };
        layers.push(Layer { attn, mlp });
    }

    let gpt2 = Gpt2 {
        config,
        wte: get_tensor(&tensors, "wte.weight"),
        wpe: get_tensor(&tensors, "wpe.weight"),
        layers,
    };

    println!("GPT-2 Layer: {} {}", gpt2.layers.len(), gpt2.wte.len());

}