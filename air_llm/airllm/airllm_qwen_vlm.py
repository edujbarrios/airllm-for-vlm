"""
AirLLM implementation for Qwen2.5-VL (Qwen Vision Language) models.

Supports models like:
- Qwen/Qwen2.5-VL-32B-Instruct
- Qwen/Qwen2.5-VL-7B-Instruct
- Qwen/Qwen2-VL-7B-Instruct
"""

import torch
from transformers import GenerationConfig
from .airllm_vlm_base import AirLLMVLMBase


class AirLLMQwenVLM(AirLLMVLMBase):
    """
    AirLLM implementation for Qwen2.5-VL Vision Language Models.
    
    Qwen-VL models feature a powerful vision encoder with dynamic resolution
    support and efficient image-text alignment.
    
    Example usage:
    ```python
    from airllm import AutoModel
    from PIL import Image
    
    model = AutoModel.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    
    # Process image and text
    image = Image.open("example.jpg")
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe this image in detail."}
            ]
        }
    ]
    
    # Use processor for multi-modal inputs
    inputs = model.processor(
        text=model.processor.apply_chat_template(messages, add_generation_prompt=True),
        images=image,
        return_tensors="pt"
    )
    
    outputs = model.generate(**inputs, max_new_tokens=256)
    print(model.tokenizer.decode(outputs[0]))
    ```
    """
    
    def __init__(self, *args, **kwargs):
        super(AirLLMQwenVLM, self).__init__(*args, **kwargs)
    
    def get_use_better_transformer(self):
        """Qwen-VL models typically don't use BetterTransformer."""
        return False
    
    def get_generation_config(self):
        """Get generation config for Qwen-VL."""
        try:
            return GenerationConfig.from_pretrained(self.model_local_path)
        except Exception:
            return GenerationConfig(
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
            )
    
    def set_layer_names_dict(self):
        """
        Set layer name mappings for Qwen2.5-VL architecture.
        
        Qwen2.5-VL uses:
        - visual: Vision encoder (ViT-based)
        - model.embed_tokens: Text embedding
        - model.layers: Transformer layers
        - model.norm: Final normalization
        - lm_head: Output projection
        """
        self.layer_names_dict = {
            'embed': 'model.embed_tokens',
            'layer_prefix': 'model.layers',
            'norm': 'model.norm',
            'lm_head': 'lm_head',
            'vision_tower': 'visual',
            'vision_projector': None,  # Qwen-VL integrates projection internally
        }
    
    def get_sequence_len(self, seq):
        """Get sequence length."""
        return seq.shape[1]
    
    def get_past_key_values_cache_seq_len(self, past_key_values):
        """Get cache sequence length."""
        return past_key_values[0][0].shape[2]
    
    def get_pos_emb_args(self, len_p, len_s):
        """Qwen-VL uses rotary embeddings computed internally."""
        return {}
    
    def get_past_key_value_args(self, k_cache, v_cache):
        """Get KV cache arguments."""
        return {'past_key_value': (k_cache, v_cache)}
    
    def get_attention_mask_args(self, full_attention_mask, len_p, len_s):
        """Get attention mask arguments."""
        return {'attention_mask': full_attention_mask[:, :, -len_s:, -len_p - len_s:]}
    
    def get_position_ids_args(self, full_position_ids, len_p, len_s):
        """Get position IDs arguments."""
        return {'position_ids': full_position_ids[:, len_p:len_p + len_s]}
    
    def _merge_image_text_embeddings(self, text_embeds, image_features, input_ids):
        """
        Merge image features with text embeddings for Qwen-VL.
        
        Qwen-VL uses special <image> tokens to mark where image
        features should be inserted in the text sequence.
        """
        if image_features is None or image_features.numel() == 0:
            return text_embeds
        
        # Qwen-VL expects image features at specific positions
        # For simplicity, prepend image features
        batch_size = text_embeds.shape[0]
        
        if image_features.dim() == 2:
            image_features = image_features.unsqueeze(0)
        
        if image_features.shape[0] != batch_size:
            image_features = image_features.expand(batch_size, -1, -1)
        
        return torch.cat([image_features, text_embeds], dim=1)


class AirLLMQwen2VLM(AirLLMQwenVLM):
    """
    AirLLM implementation for Qwen2-VL models.
    
    This is an alias for AirLLMQwenVLM with the same architecture.
    """
    pass
