"""
AutoModel factory for AirLLM Vision Language Models.

Automatically detects VLM model architecture and returns the appropriate
AirLLM VLM implementation class.
"""

import importlib
from transformers import AutoConfig
from sys import platform

is_on_mac_os = False

if platform == "darwin":
    is_on_mac_os = True

if is_on_mac_os:
    from airllm import AirLLMLlamaMlx


# VLM (Vision Language Model) architecture patterns
VLM_ARCHITECTURE_PATTERNS = [
    # GLM-4V patterns
    ("GLM4VForConditionalGeneration", "AirLLMGLMVLM"),
    ("GLMVForConditionalGeneration", "AirLLMGLMVLM"),
    ("CogVLM", "AirLLMGLMVLM"),
    
    # Qwen-VL patterns  
    ("Qwen2VLForConditionalGeneration", "AirLLMQwenVLM"),
    ("QwenVLForConditionalGeneration", "AirLLMQwenVLM"),
    ("Qwen2.5VL", "AirLLMQwenVLM"),
    
    # Moondream patterns
    ("MoondreamForConditionalGeneration", "AirLLMMoondream"),
    ("Moondream", "AirLLMMoondream"),
    
    # MedGemma / PaliGemma patterns
    ("PaliGemmaForConditionalGeneration", "AirLLMMedGemma"),
    ("MedGemma", "AirLLMMedGemma"),
    ("GemmaForVision", "AirLLMMedGemma"),
    
    # Generic VLM patterns (fallback)
    ("ForConditionalGeneration", None),  # Marker for VLM detection
    ("VisionLanguage", None),
    ("VLM", None),
    ("MultiModal", None),
]


class AutoModel:
    """
    Factory class for automatically selecting the appropriate AirLLM VLM model class.
    
    Supports Vision Language Models (VLMs) only.
    
    Example usage:
    ```python
    from airllm import AutoModel
    
    # Vision Language Model  
    model = AutoModel.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    ```
    """
    
    def __init__(self):
        raise EnvironmentError(
            "AutoModel is designed to be instantiated "
            "using the `AutoModel.from_pretrained(pretrained_model_name_or_path)` method."
        )
    
    @classmethod
    def _is_vlm_architecture(cls, architecture_name, model_name=None):
        """
        Check if the architecture indicates a Vision Language Model.
        
        Parameters
        ----------
        architecture_name : str
            The model architecture name from config.architectures
        model_name : str, optional
            The model repository name/path for additional detection
            
        Returns
        -------
        bool
            True if the model appears to be a VLM
        """
        arch_lower = architecture_name.lower()
        
        # Check for common VLM indicators in architecture name
        vlm_indicators = [
            'vision', 'vlm', 'vl', 'multimodal', 'image', 
            'conditional', 'paligemma', 'moondream', 'medgemma',
            'cogvlm', 'llava', 'glm4v', 'qwen2vl', 'qwenvl'
        ]
        
        for indicator in vlm_indicators:
            if indicator in arch_lower:
                return True
        
        # Check model name for VLM patterns
        if model_name:
            model_lower = model_name.lower()
            # Check standard indicators
            for indicator in vlm_indicators:
                if indicator in model_lower:
                    return True
            
            # Additional pattern checks for specific naming conventions
            # GLM models often have version numbers like "4.6V" or "4V"
            import re
            if re.search(r'glm.*\d+.*v', model_lower):
                return True
            # Check for "-vl-" or "-vlm-" patterns
            if '-vl-' in model_lower or '-vlm-' in model_lower:
                return True
        
        return False
    
    @classmethod
    def get_module_class(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        """
        Determine the appropriate module and class for a given model.
        
        Parameters
        ----------
        pretrained_model_name_or_path : str
            Model path or HuggingFace repo ID
        
        Returns
        -------
        tuple
            (module_name, class_name) for the appropriate AirLLM implementation
        """
        if 'hf_token' in kwargs:
            print(f"using hf_token")
            config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path, 
                trust_remote_code=True, 
                token=kwargs['hf_token']
            )
        else:
            config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path, 
                trust_remote_code=True
            )
        
        architecture = config.architectures[0] if config.architectures else ""
        
        # Check for VLM architectures
        if cls._is_vlm_architecture(architecture, pretrained_model_name_or_path):
            for pattern, class_name in VLM_ARCHITECTURE_PATTERNS:
                if class_name and pattern in architecture:
                    print(f"Detected VLM architecture: {architecture} -> {class_name}")
                    return "airllm", class_name
            
            # Check specific model repo names for VLM detection
            model_lower = pretrained_model_name_or_path.lower()
            
            if "glm" in model_lower and ("v" in model_lower or "vision" in model_lower):
                print(f"Detected GLM VLM model: {pretrained_model_name_or_path}")
                return "airllm", "AirLLMGLMVLM"
            
            if "qwen" in model_lower and ("vl" in model_lower or "vision" in model_lower):
                print(f"Detected Qwen VLM model: {pretrained_model_name_or_path}")
                return "airllm", "AirLLMQwenVLM"
            
            if "moondream" in model_lower:
                print(f"Detected Moondream model: {pretrained_model_name_or_path}")
                return "airllm", "AirLLMMoondream"
            
            if "medgemma" in model_lower or "paligemma" in model_lower:
                print(f"Detected MedGemma/PaliGemma model: {pretrained_model_name_or_path}")
                return "airllm", "AirLLMMedGemma"
            
            # Generic VLM fallback - use base VLM class
            print(f"Detected generic VLM architecture: {architecture}")
            return "airllm", "AirLLMVLMBase"
        
        # Non-VLM models are not supported in this VLM-only version
        raise ValueError(
            f"Model architecture '{architecture}' is not a supported Vision Language Model (VLM). "
            f"This version of AirLLM only supports VLMs. "
            f"Supported VLM patterns: GLM-4V, Qwen2.5-VL, Moondream, MedGemma/PaliGemma."
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        """
        Load a pretrained AirLLM Vision Language Model.
        
        Automatically detects the VLM architecture and returns the
        appropriate AirLLM implementation.
        
        Parameters
        ----------
        pretrained_model_name_or_path : str
            Model path or HuggingFace repo ID for a VLM
        
        Returns
        -------
        AirLLMVLMBase
            The loaded VLM model instance
        
        Raises
        ------
        ValueError
            If the model is not a supported Vision Language Model
        """
        if is_on_mac_os:
            return AirLLMLlamaMlx(pretrained_model_name_or_path, *inputs, **kwargs)

        module, cls_name = AutoModel.get_module_class(
            pretrained_model_name_or_path, *inputs, **kwargs
        )
        module = importlib.import_module(module)
        class_ = getattr(module, cls_name)
        return class_(pretrained_model_name_or_path, *inputs, **kwargs)