mod config;
mod sampling;
mod tokenizer;
mod model;
mod tensor;
mod weights;

use model::Gpt2;
use crate::tokenizer::Gpt2Tokenizer;

fn main() {
    let config_path = "models/config.json";
    let weight_path = "models/model.safetensors";

    let gpt2 = Gpt2::load(config_path, weight_path);

    let tokenizer = Gpt2Tokenizer::from_file("models/tokenizer.json");

    let prompt = "Rust is a systems programming language";
    let tokens = tokenizer.encode(prompt);

    let output_tokens = gpt2.generate_greedy(
        tokens,
        20,
        Some(50256),
    );

    let text = tokenizer.decode(&output_tokens);

    println!("prompt: {}", prompt);
    println!("gpt-2: {}", text);

}