"""
AirLLM implementation for MedGemma models.

Supports models like:
- google/medgemma-4b-it
- google/medgemma-27b-text-it
"""

import torch
from transformers import GenerationConfig
from .airllm_vlm_base import AirLLMVLMBase


class AirLLMMedGemma(AirLLMVLMBase):
    """
    AirLLM implementation for MedGemma Vision Language Models.
    
    MedGemma is Google's medical-focused multimodal model built on the
    Gemma architecture. It's designed for medical image analysis and
    clinical text understanding.
    
    Note: MedGemma requires acceptance of usage terms on HuggingFace.
    You'll need to provide an hf_token when loading the model.
    
    Example usage:
    ```python
    from airllm import AutoModel
    from PIL import Image
    
    model = AutoModel.from_pretrained(
        "google/medgemma-4b-it",
        hf_token="YOUR_HF_TOKEN"
    )
    
    # Process medical image
    image = Image.open("xray.jpg")
    
    inputs = model.processor(
        images=image,
        text="<image>Analyze this chest X-ray and describe any findings.",
        return_tensors="pt"
    )
    
    outputs = model.generate(**inputs, max_new_tokens=512)
    print(model.tokenizer.decode(outputs[0]))
    ```
    
    Important Notes:
    - MedGemma is intended for research and educational purposes
    - It should not be used for clinical diagnosis without proper validation
    - Always follow medical AI usage guidelines and regulations
    """
    
    def __init__(self, *args, **kwargs):
        super(AirLLMMedGemma, self).__init__(*args, **kwargs)
    
    def get_use_better_transformer(self):
        """MedGemma models typically don't use BetterTransformer."""
        return False
    
    def get_generation_config(self):
        """Get generation config for MedGemma."""
        try:
            return GenerationConfig.from_pretrained(self.model_local_path)
        except Exception:
            return GenerationConfig(
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=40,
            )
    
    def set_layer_names_dict(self):
        """
        Set layer name mappings for MedGemma architecture.
        
        MedGemma is based on PaliGemma/Gemma architecture:
        - vision_tower: SigLIP vision encoder
        - multi_modal_projector: Image-to-text projection
        - language_model: Gemma language model backbone
        """
        self.layer_names_dict = {
            'embed': 'language_model.model.embed_tokens',
            'layer_prefix': 'language_model.model.layers',
            'norm': 'language_model.model.norm',
            'lm_head': 'language_model.lm_head',
            'vision_tower': 'vision_tower',
            'vision_projector': 'multi_modal_projector',
        }
    
    def get_sequence_len(self, seq):
        """Get sequence length."""
        return seq.shape[1]
    
    def get_past_key_values_cache_seq_len(self, past_key_values):
        """Get cache sequence length."""
        return past_key_values[0][0].shape[2]
    
    def get_pos_emb_args(self, len_p, len_s):
        """MedGemma uses rotary embeddings computed internally."""
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
        Merge image features with text embeddings for MedGemma.
        
        MedGemma follows PaliGemma's approach of prepending image
        tokens to the text sequence.
        """
        if image_features is None or image_features.numel() == 0:
            return text_embeds
        
        batch_size = text_embeds.shape[0]
        
        if image_features.dim() == 2:
            image_features = image_features.unsqueeze(0)
        
        if image_features.shape[0] != batch_size:
            image_features = image_features.expand(batch_size, -1, -1)
        
        # Prepend image features
        return torch.cat([image_features, text_embeds], dim=1)
    
    def process_images(self, images, **kwargs):
        """
        Process medical images for MedGemma.
        
        Parameters
        ----------
        images : Union[Image, List[Image]]
            Input medical image(s) to process
        
        Returns
        -------
        dict
            Processed image inputs
        """
        if self.processor is not None:
            return self.processor(images=images, return_tensors="pt")
        raise NotImplementedError("MedGemma requires a processor for image handling")
