## Download GPT-2 files

```bash
pip install -U "huggingface_hub[cli]"


hf download openai-community/gpt2 \
  config.json \
  tokenizer.json \
  merges.txt \
  model.safetensors \
  vocab.json \
  --local-dir ./models
