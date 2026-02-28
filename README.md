---
language:
- en
- zh
- ja
- ko
- fr
- de
- es
- pt
- it
- ru
- ar
- nl
- pl
- sv
- da
- fi
- nb
- tr
- cs
- sk
- ro
- hu
- el
- uk
- he
license: apache-2.0
tags:
- text-generation
- reasoning
- zenlm
- zen
- abliterated
- causal-lm
- transformers
- moe
---

# Zen4-Pro-Max

Zen4-Pro-Max is an 80B Mixture-of-Experts (MoE) language model with approximately
3B parameters active per forward pass. It is the flagship model of the Zen4 family,
designed for large-scale reasoning, complex multi-step tasks, and demanding
professional workloads where maximum capability is required.

## Model Specs

| Property          | Value                              |
|-------------------|------------------------------------|
| Parameters        | 80B total / ~3B active (MoE)       |
| Architecture      | Sparse MoE transformer (causal LM) |
| Context window    | 32,768 tokens                      |
| Format            | SafeTensors (BF16), MLX 4-bit      |
| License           | Apache 2.0                         |

## Zen4 Family

This model is part of the Zen4 model family:

| Model                                               | Params       | Architecture | Context   | Use case                        |
|-----------------------------------------------------|--------------|--------------|-----------|--------------------------------|
| [zen4-mini](https://huggingface.co/zenlm/zen4-mini) | 4B           | Dense        | 40,960    | Edge, mobile, low-resource     |
| [zen4-pro](https://huggingface.co/zenlm/zen4-pro)   | 14B          | Dense        | 32,768    | Professional, complex reasoning|
| [zen4-max](https://huggingface.co/zenlm/zen4-max)   | 30B MoE      | MoE (3B act.)| 32,768    | High-capability, efficient     |
| [zen4-coder](https://huggingface.co/zenlm/zen4-coder)| 80B MoE     | MoE          | 32,768    | Advanced code, 100+ languages  |
| [zen4-pro-max](https://huggingface.co/zenlm/zen4-pro-max)| 80B MoE | MoE (3B act.)| 32,768   | Large-scale reasoning          |

## Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "zenlm/zen4-pro-max"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
)

messages = [
    {
        "role": "user",
        "content": (
            "Derive the time complexity of Dijkstra's algorithm using a Fibonacci heap, "
            "then compare it with a binary heap implementation across sparse and dense graphs."
        ),
    }
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
inputs = tokenizer([text], return_tensors="pt").to(model.device)

output_ids = model.generate(**inputs, max_new_tokens=2048)
generated = output_ids[0][len(inputs.input_ids[0]):]
print(tokenizer.decode(generated, skip_special_tokens=True))
```

### MLX (Apple Silicon)

```python
from mlx_lm import load, generate

model, tokenizer = load("zenlm/zen4-pro-max-mlx")

response = generate(
    model,
    tokenizer,
    prompt="Explain the formal semantics of linear type systems.",
    max_tokens=1024,
    verbose=True,
)
```

## Hardware Requirements

| Setup               | Minimum VRAM | Recommended                   |
|---------------------|--------------|-------------------------------|
| BF16 full precision | 160 GB       | 2× H100 80 GB (NVLink)        |
| 4-bit quantized     | 48 GB        | 80 GB                         |
| MLX 4-bit           | 48 GB RAM    | Apple M2 Ultra / M3 Ultra     |
| CPU inference       | 160 GB RAM   | 256 GB RAM                    |

The MoE architecture activates only ~3B parameters per token, meaning compute cost
scales with active parameters while memory scales with total parameters.

## Formats Available

| Format           | Location                                       |
|------------------|------------------------------------------------|
| SafeTensors BF16 | `zenlm/zen4-pro-max`                           |
| MLX 4-bit        | `zenlm/zen4-pro-max-mlx`                       |
| GGUF             | `zenlm/zen4-pro-max-gguf` (various quants)     |

## Abliteration

Zen4-Pro-Max weights are derived from an abliterated base model. Abliteration removes
refusal behavior by identifying and nullifying the "refusal direction" in the model's
residual stream.

**Method**: Directional ablation on the residual stream across all layers
**Implementation**: [hanzoai/remove-refusals](https://github.com/hanzoai/remove-refusals) — Hanzo's production abliteration toolkit
**Technique**: [Refusal in LLMs is mediated by a single direction](https://www.lesswrong.com/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction) — Arditi et al.
**Effect**: Removes refusal behaviors while preserving all other capabilities
**Identity layer**: Zen identity added via system prompt — full LoRA fine-tuning planned

Abliteration is a feature, not a limitation. It enables unrestricted research,
security testing, and applications where safety guardrails are managed at the
application layer rather than baked into model weights.

## Citation

```bibtex
@misc{zen4promax2025,
  title        = {Zen4-Pro-Max: 80B MoE Flagship Language Model for Large-Scale Reasoning},
  author       = {{Hanzo AI} and {Zoo Labs Foundation}},
  year         = {2025},
  howpublished = {\url{https://huggingface.co/zenlm/zen4-pro-max}},
  note         = {Apache 2.0 License}
}
```

---

Part of the [Zen model family](https://zenlm.org) by [Hanzo AI](https://hanzo.ai) (Techstars '17) and [Zoo Labs Foundation](https://zoo.ngo).
