# Zen4 Pro Max

80B MoE abliterated foundation model with 3B active parameters.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Overview

Zen4 Pro Max is an 80B mixture-of-experts model with only 3B parameters active per forward pass. Abliterated to remove refusal behaviors, delivering uncensored, high-quality output at inference costs comparable to a 3B dense model.

| Property | Value |
|----------|-------|
| Total Parameters | 80B |
| Active Parameters | 3B |
| Architecture | MoE |
| Context | 128K |
| Abliterated | Yes |
| License | Apache 2.0 |

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("zenlm/zen4-pro-max", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("zenlm/zen4-pro-max")

messages = [{"role": "user", "content": "Explain the trade-offs of mixture-of-experts architectures."}]
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
output = model.generate(inputs, max_new_tokens=512)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Hardware Requirements

| Precision | VRAM |
|-----------|------|
| FP16 | ~160 GB |
| GPTQ 4-bit | ~40 GB |
| GGUF Q4_K_M | ~45 GB |

## Related

- [zen-family](https://github.com/zenlm/zen-family) — Complete model family documentation
- [zen-eco](https://huggingface.co/zenlm/zen-eco) — 4B efficient model
- [Zen LM](https://github.com/zenlm) — Full model family

Apache 2.0 · [Zen LM](https://zenlm.org) · [Hanzo AI](https://hanzo.ai)
