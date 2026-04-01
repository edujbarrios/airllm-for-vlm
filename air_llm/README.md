![airllm_logo](https://github.com/lyogavin/airllm/blob/main/assets/airllm_logo_sm.png?v=3&raw=true)

# AirLLM - Vision Language Models

**AirLLM** optimizes inference memory usage, enabling you to run large Vision Language Models (VLMs) on memory-constrained GPUs (as low as 4GB VRAM).

## Installation

```bash
pip install airllm
```

For model compression support (optional):
```bash
pip install airllm bitsandbytes
```

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

## Supported VLM Models

| Model | HuggingFace Repository | Description |
|-------|------------------------|-------------|
| **GLM-4.6V-Flash** | [zai-org/GLM-4.6V-Flash](https://huggingface.co/zai-org/GLM-4.6V-Flash) | Fast GLM-4 vision model |
| **Qwen2.5-VL** | [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | Powerful Qwen vision model |
| **Moondream3** | [moondream/moondream3-preview](https://huggingface.co/moondream/moondream3-preview) | Efficient compact VLM |
| **MedGemma** | [google/medgemma-4b-it](https://huggingface.co/google/medgemma-4b-it) | Medical imaging model |

## Configuration Options

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

## MacOS Support

AirLLM supports Apple Silicon Macs:

```bash
pip install airllm mlx torch
```

## Acknowledgements

Based on SimJeg's work from the Kaggle LLM Science Exam competition:
- [GitHub @SimJeg](https://github.com/SimJeg)
- [Kaggle Code](https://www.kaggle.com/code/simjeg/platypus2-70b-with-wikipedia-rag)

## Citation

```bibtex
@software{airllm2023,
  author = {Gavin Li},
  title = {AirLLM: scaling large language models on low-end commodity computers},
  url = {https://github.com/lyogavin/airllm/},
  version = {2.12.0},
  year = {2023},
}
```

## Contributing

Contributions, ideas and discussions are welcome!

If you find it useful, please ⭐ the repo!

[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://bmc.link/lyogavinQ)
