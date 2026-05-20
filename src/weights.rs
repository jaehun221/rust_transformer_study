use std::fs::File;

use memmap2::MmapOptions;
use safetensors::SafeTensors;

use crate::config::Config;
use crate::model::{ AttentionWeights, Gpt2, Layer, LayerNormWeights, MlpWeights, };

pub fn load_gpt2(config_path: &str, weight_path: &str) -> Gpt2 {

    let file = File::open(config_path).unwrap_or_else(|err| panic!("failed to open {}: {}", config_path, err));
    let config: Config = serde_json::from_reader(file).unwrap_or_else(|err| panic!("failed to parse {}: {}", config_path, err));

    let file = File::open(weight_path).unwrap_or_else(|err| panic!("failed to open {}: {}", weight_path, err));

    let mmap = unsafe {
        MmapOptions::new()
            .map(&file)
            .expect("Memory Mapping Failed")
    }; // 용량이 크기에 디스크에서 필요한 부분만 접근 -> 운영체제 접근은 unsafe 필요

    let tensors = SafeTensors::deserialize(&mmap[..]).expect("deserialize failed");

    let mut layers = Vec::new();
    for i in 0..config.n_layer { // layer가 12 

        let ln_1 = LayerNormWeights {
            weight: get_tensor(&tensors, &format!("h.{}.ln_1.weight", i)),
            bias: get_tensor(&tensors, &format!("h.{}.ln_1.bias", i)),
        };

        let ln_2 = LayerNormWeights {
            weight: get_tensor(&tensors, &format!("h.{}.ln_2.weight", i)),
            bias: get_tensor(&tensors, &format!("h.{}.ln_2.bias", i)),
        };

        let attn = AttentionWeights {
            c_attn: get_tensor(&tensors, &format!("h.{}.attn.c_attn.weight", i)),
            c_attn_bias: get_tensor(&tensors, &format!("h.{}.attn.c_attn.bias", i)),
            c_proj: get_tensor(&tensors, &format!("h.{}.attn.c_proj.weight", i)),
            c_proj_bias: get_tensor(&tensors, &format!("h.{}.attn.c_proj.bias", i)),
        };

        let mlp = MlpWeights {
            c_fc: get_tensor(&tensors, &format!("h.{}.mlp.c_fc.weight", i)),
            c_fc_bias: get_tensor(&tensors, &format!("h.{}.mlp.c_fc.bias", i)),
            c_proj: get_tensor(&tensors, &format!("h.{}.mlp.c_proj.weight", i)),
            c_proj_bias: get_tensor(&tensors, &format!("h.{}.mlp.c_proj.bias", i)),
        };

        layers.push(Layer { ln_1, ln_2, attn, mlp });
    }

        let ln_f = LayerNormWeights {
            weight: get_tensor(&tensors, "ln_f.weight"),
            bias: get_tensor(&tensors, "ln_f.bias"),
        };

        let wte = get_tensor(&tensors, "wte.weight");
        let wpe = get_tensor(&tensors, "wpe.weight");

    Gpt2 {
        config,
        wte,
        wpe,
        layers,
        ln_f,
    }
    
}

// 특정한 tensor를 name으로 추출        Ex): h.{}.attn.c_attn.weight
// model.safetensors 내부에서 추출한다
fn get_tensor(tensors: &SafeTensors, name: &str) -> Vec<f32> {
    let tensor = tensors.tensor(name).expect("weight not found");
    tensor.data()
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("slice failed")))
        .collect()
}