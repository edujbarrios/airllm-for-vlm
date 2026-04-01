"""
AirLLM - Optimized Vision Language Model (VLM) inference for memory-constrained environments.

Supports running large VLMs on memory-constrained GPUs (as low as 4GB VRAM) by using 
layer-by-layer inference with strategic GPU memory management.
"""

from sys import platform

is_on_mac_os = False

if platform == "darwin":
    is_on_mac_os = True

if is_on_mac_os:
    from .airllm_llama_mlx import AirLLMLlamaMlx
    from .auto_model import AutoModel
else:
    # Base classes
    from .airllm_base import AirLLMBaseModel
    
    # Vision Language Model (VLM) implementations
    from .airllm_vlm_base import AirLLMVLMBase
    from .airllm_glm_vlm import AirLLMGLMVLM
    from .airllm_qwen_vlm import AirLLMQwenVLM, AirLLMQwen2VLM
    from .airllm_moondream import AirLLMMoondream
    from .airllm_medgemma import AirLLMMedGemma
    
    # Factory class
    from .auto_model import AutoModel
    
    # Utilities
    from .utils import split_and_save_layers
    from .utils import NotEnoughSpaceException
    from .utils import compress_layer_state_dict, uncompress_layer_state_dict

# Version
__version__ = "2.12.0"

# List of supported VLM models
SUPPORTED_VLM_MODELS = [
    "zai-org/GLM-4.6V-Flash",
    "Qwen/Qwen2.5-VL-32B-Instruct",
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "moondream/moondream3-preview",
    "google/medgemma-4b-it",
]

