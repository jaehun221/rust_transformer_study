use memmap2::MmapOptions;
use serde::Deserialize;
use std::fs::File;
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

struct MlpWeights {
    c_fc: Vec<f32>,
    c_proj: Vec<f32>,
}

struct Layer {
    mlp: MlpWeights,
}

struct Gpt2 {
    config: Config,
    wte: Vec<f32>, // Word Token Embedding
    wpe: Vec<f32>, // Word Position Embedding
    layers: Vec<Layer>,
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
        let mlp = MlpWeights {
            c_fc: get_tensor(&tensors, &format!("h.{}.mlp.c_fc.weight", i)),
            c_proj: get_tensor(&tensors, &format!("h.{}.mlp.c_proj.weight", i)),
        };
        layers.push(Layer { mlp });
    }

    let gpt2 = Gpt2 {
        config,
        wte: get_tensor(&tensors, "wte.weight"),
        wpe: get_tensor(&tensors, "wpe.weight"),
        layers,
    };
}