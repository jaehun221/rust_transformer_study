pub struct Gpt2Tokenizer {
    tokenizer: tokenizers::Tokenizer,
}

impl Gpt2Tokenizer {
    pub fn from_file(path: &str) -> Self {
        let tokenizer = tokenizers::Tokenizer::from_file(path)
            .expect("failed to load tokenizer");

        Self { tokenizer }
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        let encoding = self.tokenizer
            .encode(text, true)
            .expect("failed to encode text");

        encoding
            .get_ids()
            .iter()
            .map(|&id| id as usize)
            .collect()
    }

    pub fn decode(&self, tokens: &[usize]) -> String {
        let ids: Vec<u32> = tokens.iter().map(|&id| id as u32).collect();

        self.tokenizer
            .decode(&ids, true)
            .expect("failed to decode tokens")
    }
}