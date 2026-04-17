![airllm_logo](https://github.com/lyogavin/airllm/blob/main/assets/airllm_logo_sm.png?v=3&raw=true)

# AirLLM - Vision Language Models

**AirLLM** optimizes inference memory usage, enabling you to run large Vision Language Models (VLMs) on memory-constrained GPUs (as low as 4GB VRAM).

> This repository is a fork focused on making AirLLM durable for VLM workloads: stable multimodal support, practical model coverage, and reliable low-VRAM execution.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Supported VLM Models](#supported-vlm-models)
- [Examples](#examples)
- [Configuration Options](#configuration-options)
- [FAQ](#faq)

---

## Installation

```bash
# PyPI package name remains `airllm`
pip install airllm
```

For model compression support (optional):
```bash
pip install airllm bitsandbytes
```

### Development Mode Install

For contributors or users who want this fork’s latest VLM-focused improvements:

```bash
# Clone the repository
git clone https://github.com/edujbarrios/airllm-for-vlm.git
cd airllm-for-vlm/air_llm

# Install in development/editable mode
pip install -e .

# Or with compression support
pip install -e ".[compression]"
```

This installs the package in editable mode, allowing you to modify the source code and have changes reflected immediately without reinstalling.

---

## Quick Start

```python
from airllm import AutoModel
from PIL import Image

# Load a Vision Language Model
model = AutoModel.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

# Load an image
image = Image.open("example.jpg")

# Prepare inputs
inputs = model.processor(
    text="Describe this image in detail.",
    images=image,
    return_tensors="pt"
)

# Generate response
outputs = model.generate(
    input_ids=inputs['input_ids'].cuda(),
    pixel_values=inputs['pixel_values'].cuda(),
    max_new_tokens=256,
    use_cache=False,
    return_dict_in_generate=True
)

print(model.tokenizer.decode(outputs.sequences[0]))
```

---

## Supported VLM Models

| Model | HuggingFace Repository | Description |
|-------|------------------------|-------------|
| **GLM-4.6V-Flash** | [zai-org/GLM-4.6V-Flash](https://huggingface.co/zai-org/GLM-4.6V-Flash) | Fast GLM-4 vision model |
| **Qwen2.5-VL** | [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | Powerful Qwen vision model |
| **Moondream3** | [moondream/moondream3-preview](https://huggingface.co/moondream/moondream3-preview) | Efficient compact VLM |
| **MedGemma** | [google/medgemma-4b-it](https://huggingface.co/google/medgemma-4b-it) | Medical imaging model |

---

## Examples

### GLM-4V

```python
from airllm import AutoModel
from PIL import Image

model = AutoModel.from_pretrained("zai-org/GLM-4.6V-Flash")
image = Image.open("photo.jpg")

messages = "<|user|>\n<image>\nWhat do you see in this image?\n<|assistant|>\n"

inputs = model.processor(
    text=messages,
    images=image,
    return_tensors="pt"
)

outputs = model.generate(
    **{k: v.cuda() for k, v in inputs.items()},
    max_new_tokens=200
)

print(model.tokenizer.decode(outputs[0]))
```

### Moondream

```python
from airllm import AutoModel
from PIL import Image

model = AutoModel.from_pretrained("moondream/moondream3-preview")
image = Image.open("scene.jpg")

inputs = model.processor(
    images=image,
    text="<image>\n\nDescribe what's happening in this image.",
    return_tensors="pt"
)

outputs = model.generate(
    **{k: v.cuda() for k, v in inputs.items()},
    max_new_tokens=256
)

print(model.tokenizer.decode(outputs[0]))
```

### MedGemma (Medical Imaging)

```python
from airllm import AutoModel
from PIL import Image

# Note: MedGemma requires HuggingFace token and usage agreement
model = AutoModel.from_pretrained(
    "google/medgemma-4b-it",
    hf_token="YOUR_HF_TOKEN"
)

image = Image.open("chest_xray.jpg")

inputs = model.processor(
    images=image,
    text="<image>Analyze this chest X-ray. Describe any notable findings.",
    return_tensors="pt"
)

outputs = model.generate(
    **{k: v.cuda() for k, v in inputs.items()},
    max_new_tokens=512
)

print(model.tokenizer.decode(outputs[0]))

# ⚠️ MedGemma is for research purposes only.
# Always consult healthcare professionals for medical decisions.
```

---

## Configuration Options

### Basic Options

| Option | Description | Default |
|--------|-------------|---------|
| `compression` | Model compression: `'4bit'`, `'8bit'`, or `None` | `None` |
| `hf_token` | HuggingFace token for gated models | `None` |
| `layer_shards_saving_path` | Custom path for layer shards | HuggingFace cache |
| `delete_original` | Delete original model after splitting | `False` |

### Example with Compression

```python
model = AutoModel.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    compression='4bit',  # 3x faster inference
    hf_token="YOUR_HF_TOKEN"  # if needed
)
```

### MacOS Support

AirLLM supports Apple Silicon Macs:

```bash
pip install airllm mlx torch
```

---

## FAQ

### 1. MetadataIncompleteBuffer Error

```
safetensors_rust.SafetensorError: Error while deserializing header: MetadataIncompleteBuffer
```

**Solution:** You've run out of disk space. Clear the HuggingFace cache and ensure sufficient disk space:
```bash
rm -rf ~/.cache/huggingface/hub/*
```

### 2. Gated Model Access (401 Error)

```
401 Client Error....Repo model ... is gated.
```

**Solution:** Provide your HuggingFace token:
```python
model = AutoModel.from_pretrained("model-name", hf_token="YOUR_HF_TOKEN")
```

### 3. Padding Token Error

```
ValueError: Asking to pad but the tokenizer does not have a padding token.
```

**Solution:** Disable padding in tokenizer call:
```python
inputs = model.tokenizer(text, padding=False, ...)
```

---

## Citation

```bibtex
@software{airllm-vlm2024,
  author = {Eduardo J. Barrios},
  title = {AirLLM for VLM: Vision Language Models inference on memory-constrained GPUs},
  url = {https://github.com/edujbarrios/airllm-for-vlm/},
  version = {2.12.0},
  year = {2024},
}
```
