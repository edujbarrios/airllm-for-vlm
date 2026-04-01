![airllm_logo](https://github.com/lyogavin/airllm/blob/main/assets/airllm_logo_sm.png?v=3&raw=true)

[**Quickstart**](#quickstart) | 
[**Vision Language Models**](#vision-language-models) |
[**Configurations**](#configurations) | 
[**MacOS**](#macos) | 
[**Example notebooks**](#example-python-notebook) | 
[**FAQ**](#faq)

**AirLLM** optimizes inference memory usage, allowing 70B large language models to run inference on a single 4GB GPU card without quantization, distillation and pruning. And you can run **405B Llama3.1** on **8GB vram** now.

**🆕 NEW: Vision Language Model (VLM) Support!** - Now supports multimodal models including GLM-4V, Qwen2.5-VL, Moondream, and MedGemma.


## Table of Contents

* [Quick start](#quickstart)
* [Vision Language Models](#vision-language-models)
* [Model Compression](#model-compression---3x-inference-speed-up)
* [Configurations](#configurations)
* [Run on MacOS](#macos)
* [Example notebooks](#example-python-notebook)
* [Supported Models](#supported-models)
* [Acknowledgement](#acknowledgement)
* [FAQ](#faq)

## Quickstart

### 1. Install package

First, install the airllm pip package.

```bash
pip install airllm
```

### 2. Inference

Then, initialize AirLLMLlama2, pass in the huggingface repo ID of the model being used, or the local path, and inference can be performed similar to a regular transformer model.

(*You can also specify the path to save the splitted layered model through **layer_shards_saving_path** when init AirLLMLlama2.*

```python
from airllm import AutoModel

MAX_LENGTH = 128
# could use hugging face model repo id:
model = AutoModel.from_pretrained("garage-bAInd/Platypus2-70B-instruct")

# or use model's local path...
#model = AutoModel.from_pretrained("/home/ubuntu/.cache/huggingface/hub/models--garage-bAInd--Platypus2-70B-instruct/snapshots/b585e74bcaae02e52665d9ac6d23f4d0dbc81a0f")

input_text = [
        'What is the capital of United States?',
        #'I like',
    ]

input_tokens = model.tokenizer(input_text,
    return_tensors="pt", 
    return_attention_mask=False, 
    truncation=True, 
    max_length=MAX_LENGTH, 
    padding=False)
           
generation_output = model.generate(
    input_tokens['input_ids'].cuda(), 
    max_new_tokens=20,
    use_cache=True,
    return_dict_in_generate=True)

output = model.tokenizer.decode(generation_output.sequences[0])

print(output)

```
 
 
Note: During inference, the original model will first be decomposed and saved layer-wise. Please ensure there is sufficient disk space in the huggingface cache directory.
 

## Vision Language Models

AirLLM now supports Vision Language Models (VLMs) for multimodal inference! This allows you to run large vision-language models on memory-constrained GPUs.

### Supported VLM Models

| Model | HuggingFace Repository | Description |
|-------|------------------------|-------------|
| **GLM-4.6V-Flash** | [zai-org/GLM-4.6V-Flash](https://huggingface.co/zai-org/GLM-4.6V-Flash) | Fast GLM-4 vision model |
| **Qwen2.5-VL** | [Qwen/Qwen2.5-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct) | Powerful Qwen vision model |
| **Moondream3** | [moondream/moondream3-preview](https://huggingface.co/moondream/moondream3-preview) | Efficient compact VLM |
| **MedGemma** | [google/medgemma-4b-it](https://huggingface.co/google/medgemma-4b-it) | Medical imaging model |

### VLM Quick Start

```python
from airllm import AutoModel
from PIL import Image

# Load a Vision Language Model
model = AutoModel.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

# Load and process an image
image = Image.open("example.jpg")

# Prepare inputs using the processor
inputs = model.processor(
    text="Describe this image in detail.",
    images=image,
    return_tensors="pt"
)

# Generate response
generation_output = model.generate(
    input_ids=inputs['input_ids'].cuda(),
    pixel_values=inputs['pixel_values'].cuda(),
    max_new_tokens=256,
    use_cache=False,
    return_dict_in_generate=True
)

output = model.tokenizer.decode(generation_output.sequences[0])
print(output)
```

### VLM Examples

<details>
<summary>GLM-4V Example</summary>

```python
from airllm import AutoModel
from PIL import Image

model = AutoModel.from_pretrained("zai-org/GLM-4.6V-Flash")

image = Image.open("photo.jpg")

# GLM-4V uses a specific chat format
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
</details>

<details>
<summary>Moondream Example</summary>

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
</details>

<details>
<summary>MedGemma Medical Imaging Example</summary>

```python
from airllm import AutoModel
from PIL import Image

# Note: MedGemma requires HuggingFace token and usage agreement
model = AutoModel.from_pretrained(
    "google/medgemma-4b-it",
    hf_token="YOUR_HF_TOKEN"
)

# Load medical image (e.g., X-ray)
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

# Important: MedGemma is for research purposes only
# Always consult healthcare professionals for medical decisions
```
</details>

### VLM Configuration Options

VLM models support all standard AirLLM configurations plus additional options:

* **processor**: Automatically loaded for handling multimodal inputs
* **pixel_values**: Image tensor input for vision encoder
* **image_sizes**: Optional image dimension information

```python
model = AutoModel.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    compression='4bit',  # Enable compression for faster inference
    device="cuda:0",
    max_seq_len=2048,
)
```

## Model Compression - 3x Inference Speed Up!

We just added model compression based on block-wise quantization-based model compression. Which can further **speed up the inference speed** for up to **3x** , with **almost ignorable accuracy loss!** (see more performance evaluation and why we use block-wise quantization in [this paper](https://arxiv.org/abs/2212.09720))

![speed_improvement](https://github.com/lyogavin/airllm/blob/main/assets/airllm2_time_improvement.png?v=2&raw=true)

#### How to enable model compression speed up:

* Step 1. make sure you have [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) installed by `pip install -U bitsandbytes `
* Step 2. make sure airllm verion later than 2.0.0: `pip install -U airllm` 
* Step 3. when initialize the model, passing the argument compression ('4bit' or '8bit'):

```python
model = AutoModel.from_pretrained("garage-bAInd/Platypus2-70B-instruct",
                     compression='4bit' # specify '8bit' for 8-bit block-wise quantization 
                    )
```

#### What are the differences between model compression and quantization?

Quantization normally needs to quantize both weights and activations to really speed things up. Which makes it harder to maintain accuracy and avoid the impact of outliers in all kinds of inputs.

While in our case the bottleneck is mainly at the disk loading, we only need to make the model loading size smaller. So, we get to only quantize the weights' part, which is easier to ensure the accuracy.

## Configurations
 
When initialize the model, we support the following configurations:

* **compression**: supported options: 4bit, 8bit for 4-bit or 8-bit block-wise quantization, or by default None for no compression
* **profiling_mode**: supported options: True to output time consumptions or by default False
* **layer_shards_saving_path**: optionally another path to save the splitted model
* **hf_token**: huggingface token can be provided here if downloading gated models like: *meta-llama/Llama-2-7b-hf*
* **prefetching**: prefetching to overlap the model loading and compute. By default, turned on. For now, only AirLLMLlama2 supports this.
* **delete_original**: if you don't have too much disk space, you can set delete_original to true to delete the original downloaded hugging face model, only keep the transformed one to save half of the disk space. 

## MacOS

Just install airllm and run the code the same as on linux. See more in [Quick Start](#quickstart).

* make sure you installed [mlx](https://github.com/ml-explore/mlx?tab=readme-ov-file#installation) and torch
* you probably need to install python native see more [here](https://stackoverflow.com/a/65432861/21230266)
* only [Apple silicon](https://support.apple.com/en-us/HT211814) is supported

Example [python notebook] (https://github.com/lyogavin/airllm/blob/main/air_llm/examples/run_on_macos.ipynb)


## Supported Models

### Text-Only Language Models

| Architecture | Example Models |
|-------------|----------------|
| **Llama/Llama2/Llama3** | meta-llama/Llama-2-7b-hf, meta-llama/Llama-3.1-405B |
| **Qwen/Qwen2** | Qwen/Qwen-7B, Qwen/Qwen2-72B |
| **ChatGLM** | THUDM/chatglm3-6b-base |
| **Baichuan** | baichuan-inc/Baichuan2-7B-Base |
| **InternLM** | internlm/internlm-chat-7b |
| **Mistral** | mistralai/Mistral-7B-Instruct-v0.1 |
| **Mixtral** | mistralai/Mixtral-8x7B-v0.1 |

### Vision Language Models (VLMs)

| Model | Repository | Description |
|-------|------------|-------------|
| **GLM-4V** | [zai-org/GLM-4.6V-Flash](https://huggingface.co/zai-org/GLM-4.6V-Flash) | Fast multimodal GLM |
| **Qwen2.5-VL** | [Qwen/Qwen2.5-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct) | Advanced Qwen vision model |
| **Moondream** | [moondream/moondream3-preview](https://huggingface.co/moondream/moondream3-preview) | Efficient compact VLM |
| **MedGemma** | [google/medgemma-4b-it](https://huggingface.co/google/medgemma-4b-it) | Medical imaging AI |


## Example Python Notebook

Example colabs here:

<a target="_blank" href="https://colab.research.google.com/github/lyogavin/airllm/blob/main/air_llm/examples/run_all_types_of_models.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

#### example of other models (ChatGLM, QWen, Baichuan, Mistral, etc):

<details>


* ChatGLM:

```python
from airllm import AutoModel
MAX_LENGTH = 128
model = AutoModel.from_pretrained("THUDM/chatglm3-6b-base")
input_text = ['What is the capital of China?',]
input_tokens = model.tokenizer(input_text,
    return_tensors="pt", 
    return_attention_mask=False, 
    truncation=True, 
    max_length=MAX_LENGTH, 
    padding=True)
generation_output = model.generate(
    input_tokens['input_ids'].cuda(), 
    max_new_tokens=5,
    use_cache= True,
    return_dict_in_generate=True)
model.tokenizer.decode(generation_output.sequences[0])
```

* QWen:

```python
from airllm import AutoModel
MAX_LENGTH = 128
model = AutoModel.from_pretrained("Qwen/Qwen-7B")
input_text = ['What is the capital of China?',]
input_tokens = model.tokenizer(input_text,
    return_tensors="pt", 
    return_attention_mask=False, 
    truncation=True, 
    max_length=MAX_LENGTH)
generation_output = model.generate(
    input_tokens['input_ids'].cuda(), 
    max_new_tokens=5,
    use_cache=True,
    return_dict_in_generate=True)
model.tokenizer.decode(generation_output.sequences[0])
```


* Baichuan, InternLM, Mistral, etc:

```python
from airllm import AutoModel
MAX_LENGTH = 128
model = AutoModel.from_pretrained("baichuan-inc/Baichuan2-7B-Base")
#model = AutoModel.from_pretrained("internlm/internlm-20b")
#model = AutoModel.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
input_text = ['What is the capital of China?',]
input_tokens = model.tokenizer(input_text,
    return_tensors="pt", 
    return_attention_mask=False, 
    truncation=True, 
    max_length=MAX_LENGTH)
generation_output = model.generate(
    input_tokens['input_ids'].cuda(), 
    max_new_tokens=5,
    use_cache=True,
    return_dict_in_generate=True)
model.tokenizer.decode(generation_output.sequences[0])
```


</details>


#### To request other model support: [here](https://docs.google.com/forms/d/e/1FAIpQLSe0Io9ANMT964Zi-OQOq1TJmnvP-G3_ZgQDhP7SatN0IEdbOg/viewform?usp=sf_link)



## Acknowledgement

A lot of the code are based on SimJeg's great work in the Kaggle exam competition. Big shoutout to SimJeg:

[GitHub account @SimJeg](https://github.com/SimJeg), 
[the code on Kaggle](https://www.kaggle.com/code/simjeg/platypus2-70b-with-wikipedia-rag), 
[the associated discussion](https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/446414).


## FAQ

### 1. MetadataIncompleteBuffer

safetensors_rust.SafetensorError: Error while deserializing header: MetadataIncompleteBuffer

If you run into this error, most possible cause is you run out of disk space. The process of splitting model is very disk-consuming. See [this](https://huggingface.co/TheBloke/guanaco-65B-GPTQ/discussions/12). You may need to extend your disk space, clear huggingface [.cache](https://huggingface.co/docs/datasets/cache) and rerun. 

### 2. ValueError: max() arg is an empty sequence

Most likely you are loading QWen or ChatGLM model with Llama2 class. Try the following:

For QWen model: 

```python
from airllm import AutoModel #<----- instead of AirLLMLlama2
AutoModel.from_pretrained(...)
```

For ChatGLM model: 

```python
from airllm import AutoModel #<----- instead of AirLLMLlama2
AutoModel.from_pretrained(...)
```

### 3. 401 Client Error....Repo model ... is gated.

Some models are gated models, needs huggingface api token. You can provide hf_token:

```python
model = AutoModel.from_pretrained("meta-llama/Llama-2-7b-hf", #hf_token='HF_API_TOKEN')
```

### 4. ValueError: Asking to pad but the tokenizer does not have a padding token.

Some model's tokenizer doesn't have padding token, so you can set a padding token or simply turn the padding config off:

 ```python
input_tokens = model.tokenizer(input_text,
    return_tensors="pt", 
    return_attention_mask=False, 
    truncation=True, 
    max_length=MAX_LENGTH, 
    padding=False  #<-----------   turn off padding 
)
```

## Citing AirLLM

If you find
AirLLM useful in your research and wish to cite it, please use the following
BibTex entry:

```
@software{airllm2023,
  author = {Gavin Li},
  title = {AirLLM: scaling large language models on low-end commodity computers},
  url = {https://github.com/lyogavin/airllm/},
  version = {0.0},
  year = {2023},
}
```


## Contribution 

Welcomed contributions, ideas and discussions!

If you find it useful, please ⭐ or buy me a coffee! 🙏

[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://bmc.link/lyogavinQ)
