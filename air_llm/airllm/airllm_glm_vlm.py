"""
AirLLM implementation for GLM-4V models (Vision Language Models).

Supports models like:
- zai-org/GLM-4.6V-Flash
- THUDM/glm-4v-9b
"""

from transformers import GenerationConfig
from .airllm_vlm_base import AirLLMVLMBase


class AirLLMGLMVLM(AirLLMVLMBase):
    """
    AirLLM implementation for GLM-4V Vision Language Models.
    
    GLM-4V models use a vision transformer encoder with a connector
    to project image features into the language model space.
    
    Example usage:
    ```python
    from airllm import AutoModel
    
    model = AutoModel.from_pretrained("zai-org/GLM-4.6V-Flash")
    
    # Process image and text
    from PIL import Image
    image = Image.open("example.jpg")
    
    inputs = model.processor(
        text="<|user|>\\nDescribe this image.\\n<|assistant|>\\n",
        images=image,
        return_tensors="pt"
    )
    
    outputs = model.generate(**inputs, max_new_tokens=100)
    print(model.tokenizer.decode(outputs[0]))
    ```
    """
    
    def __init__(self, *args, **kwargs):
        super(AirLLMGLMVLM, self).__init__(*args, **kwargs)
    
    def get_use_better_transformer(self):
        """GLM-4V models typically don't use BetterTransformer."""
        return False
    
    def get_generation_config(self):
        """Get generation config for GLM-4V."""
        try:
            return GenerationConfig.from_pretrained(self.model_local_path)
        except Exception:
            return GenerationConfig(
                max_new_tokens=256,
                do_sample=True,
                temperature=0.8,
                top_p=0.8,
            )
    
    def set_layer_names_dict(self):
        """
        Set layer name mappings for GLM-4V architecture.
        
        GLM-4V uses:
        - vision_model: Vision encoder
        - connector: Image-to-text projection
        - transformer: Language model backbone
        """
        self.layer_names_dict = {
            'embed': 'transformer.embedding.word_embeddings',
            'layer_prefix': 'transformer.encoder.layers',
            'norm': 'transformer.encoder.final_layernorm',
            'lm_head': 'transformer.output_layer',
            'vision_tower': 'vision_model',
            'vision_projector': 'connector',
            'rotary_pos_emb': 'transformer.rotary_pos_emb',
        }
    
    def get_sequence_len(self, seq):
        """GLM models use different sequence dimension."""
        return seq.shape[0]
    
    def get_past_key_values_cache_seq_len(self, past_key_values):
        """Get cache sequence length for GLM-4V."""
        return past_key_values[0][0].shape[0]
    
    def get_pos_emb_args(self, len_p, len_s):
        """Get rotary positional embedding arguments for GLM-4V."""
        try:
            rotary_pos_emb = self.model.transformer.rotary_pos_emb(self.config.seq_length)
            rotary_pos_emb = rotary_pos_emb[None, :len_s]
            rotary_pos_emb = rotary_pos_emb.transpose(0, 1).contiguous()
            return {'rotary_pos_emb': rotary_pos_emb}
        except Exception:
            return {}
    
    def get_past_key_value_args(self, k_cache, v_cache):
        """Get KV cache arguments for GLM-4V."""
        return {'kv_cache': (k_cache, v_cache)}
    
    def get_attention_mask_args(self, full_attention_mask, len_p, len_s):
        """GLM-4V uses different attention mask handling."""
        return {'attention_mask': None}
    
    def get_position_ids_args(self, full_position_ids, len_p, len_s):
        """GLM-4V handles position IDs internally."""
        return {}
    
    def _merge_image_text_embeddings(self, text_embeds, image_features, input_ids):
        """
        Merge image features with text embeddings for GLM-4V.
        
        GLM-4V typically uses special tokens to mark image positions
        in the input sequence.
        """
        # For GLM-4V, image features are typically prepended or 
        # inserted at specific positions marked by special tokens
        if image_features is not None and image_features.numel() > 0:
            # Prepend image features to text embeddings
            return torch.cat([image_features, text_embeds], dim=1)
        return text_embeds


# Import torch for _merge_image_text_embeddings
import torch
