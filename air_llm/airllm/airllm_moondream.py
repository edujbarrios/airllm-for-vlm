"""
AirLLM implementation for Moondream models.

Supports models like:
- moondream/moondream3-preview
- moondream/moondream2
- vikhyatk/moondream2
"""

import torch
from transformers import GenerationConfig
from .airllm_vlm_base import AirLLMVLMBase


class AirLLMMoondream(AirLLMVLMBase):
    """
    AirLLM implementation for Moondream Vision Language Models.
    
    Moondream is a compact and efficient vision-language model designed
    for resource-constrained environments. It provides strong visual
    understanding capabilities in a smaller footprint.
    
    Example usage:
    ```python
    from airllm import AutoModel
    from PIL import Image
    
    model = AutoModel.from_pretrained("moondream/moondream3-preview")
    
    # Process image
    image = Image.open("example.jpg")
    
    # Encode image and generate
    inputs = model.processor(
        images=image,
        text="<image>\\n\\nDescribe this image.",
        return_tensors="pt"
    )
    
    outputs = model.generate(**inputs, max_new_tokens=256)
    print(model.tokenizer.decode(outputs[0]))
    ```
    """
    
    def __init__(self, *args, **kwargs):
        super(AirLLMMoondream, self).__init__(*args, **kwargs)
    
    def get_use_better_transformer(self):
        """Moondream models typically don't use BetterTransformer."""
        return False
    
    def get_generation_config(self):
        """Get generation config for Moondream."""
        try:
            return GenerationConfig.from_pretrained(self.model_local_path)
        except Exception:
            return GenerationConfig(
                max_new_tokens=256,
                do_sample=False,
                use_cache=True,
            )
    
    def set_layer_names_dict(self):
        """
        Set layer name mappings for Moondream architecture.
        
        Moondream uses:
        - vision_encoder: Vision transformer backbone
        - vision_proj: Projection layer for image features
        - text_model: Language model backbone (typically Phi-based)
        """
        self.layer_names_dict = {
            'embed': 'text_model.model.embed_tokens',
            'layer_prefix': 'text_model.model.layers',
            'norm': 'text_model.model.norm',
            'lm_head': 'text_model.lm_head',
            'vision_tower': 'vision_encoder',
            'vision_projector': 'vision_proj',
        }
    
    def get_sequence_len(self, seq):
        """Get sequence length."""
        return seq.shape[1]
    
    def get_past_key_values_cache_seq_len(self, past_key_values):
        """Get cache sequence length."""
        return past_key_values[0][0].shape[2]
    
    def get_pos_emb_args(self, len_p, len_s):
        """Moondream uses rotary embeddings computed internally."""
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
        Merge image features with text embeddings for Moondream.
        
        Moondream uses <image> tokens to mark where image features
        should be inserted.
        """
        if image_features is None or image_features.numel() == 0:
            return text_embeds
        
        batch_size = text_embeds.shape[0]
        
        if image_features.dim() == 2:
            image_features = image_features.unsqueeze(0)
        
        if image_features.shape[0] != batch_size:
            image_features = image_features.expand(batch_size, -1, -1)
        
        # Prepend image features to text embeddings
        return torch.cat([image_features, text_embeds], dim=1)
    
    def process_images(self, images, **kwargs):
        """
        Process images for Moondream.
        
        Parameters
        ----------
        images : Union[Image, List[Image]]
            Input image(s) to process
        
        Returns
        -------
        dict
            Processed image inputs
        """
        if self.processor is not None:
            return self.processor(images=images, return_tensors="pt")
        raise NotImplementedError("Moondream requires a processor for image handling")
