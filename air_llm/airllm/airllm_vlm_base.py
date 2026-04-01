"""
Base class for Vision Language Models (VLM) in AirLLM.

This module provides the foundation for supporting multimodal models that can 
process both images and text, such as GLM-4V, Qwen2.5-VL, Moondream, and MedGemma.
"""

from typing import List, Optional, Tuple, Union
from tqdm import tqdm
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor

import torch
from transformers import (
    AutoConfig, 
    AutoModelForCausalLM, 
    AutoTokenizer, 
    AutoProcessor,
    GenerationMixin, 
    GenerationConfig
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from accelerate import init_empty_weights
from accelerate.utils.modeling import set_module_tensor_to_device

from .profiler import LayeredProfiler
from .utils import clean_memory, load_layer, find_or_create_local_splitted_path

try:
    import bitsandbytes as bnb
    bitsandbytes_installed = True
except ImportError:
    bitsandbytes_installed = False

try:
    from transformers.cache_utils import Cache, DynamicCache
    cache_utils_installed = True
except ImportError:
    cache_utils_installed = False


class AirLLMVLMBase(GenerationMixin):
    """
    Base class for Vision Language Models in AirLLM.
    
    This class extends the layer-by-layer inference approach to support
    multimodal models that process both images and text. It handles:
    - Vision encoder layers
    - Image-text projection layers
    - Text transformer layers
    - Proper handling of pixel_values and image inputs
    
    Subclasses should override:
    - set_layer_names_dict(): Define model-specific layer name mappings
    - get_model_class(): Return the appropriate AutoModel class
    - process_images(): Handle model-specific image preprocessing
    """
    
    # Flag to indicate this is a VLM model
    is_vlm = True
    
    def set_layer_names_dict(self):
        """
        Set the layer name mappings for the model architecture.
        
        Subclasses should override this method to provide model-specific
        layer name mappings. For VLMs, this includes:
        - Vision encoder layers
        - Image projection layers  
        - Text embedding layers
        - Transformer layers
        - Output layers
        """
        self.layer_names_dict = {
            'embed': 'model.embed_tokens',
            'layer_prefix': 'model.layers',
            'norm': 'model.norm',
            'lm_head': 'lm_head',
            # VLM-specific layer names (to be overridden by subclasses)
            'vision_tower': None,  # Vision encoder
            'vision_projector': None,  # Image-to-text projection
        }
    
    def get_model_class(self):
        """
        Return the AutoModel class to use for this model.
        
        Subclasses can override to return model-specific classes.
        Default returns AutoModelForCausalLM for compatibility.
        """
        return AutoModelForCausalLM
    
    def __init__(
        self, 
        model_local_path_or_repo_id, 
        device="cuda:0", 
        dtype=torch.float16, 
        max_seq_len=512,
        layer_shards_saving_path=None, 
        profiling_mode=False, 
        compression=None,
        hf_token=None, 
        prefetching=True, 
        delete_original=False
    ):
        """
        Initialize a Vision Language Model with layer-by-layer inference.
        
        Parameters
        ----------
        model_local_path_or_repo_id : str or Path
            Path to the local model checkpoint or HuggingFace repo ID
        device : str, optional
            Device to run inference on, by default "cuda:0"
        dtype : torch.dtype, optional
            Data type for model weights, by default torch.float16
        max_seq_len : int, optional
            Maximum sequence length, by default 512
        layer_shards_saving_path : str, optional
            Optional path to save layered shards model file
        profiling_mode : bool, optional
            Whether to profile model loading time, by default False
        compression : str, optional
            '4bit' or '8bit' for compression, by default None
        hf_token : str, optional
            HuggingFace API token for gated models
        prefetching : bool, optional
            Whether to enable layer prefetching, by default True
        delete_original : bool, optional
            Whether to delete original model files after splitting
        """
        self.profiling_mode = profiling_mode
        self.profiler = LayeredProfiler()
        
        self.total_disk_loading_time = None
        self.total_gpu_loading_time = None
        self.total_compression_overhead_time = None
        self._supports_cache_class = False
        self.hf_quantizer = None
        
        if compression is not None:
            if not bitsandbytes_installed:
                raise ImportError(
                    'WARNING: bitsandbytes not found. Compression needs bitsandbytes. '
                    'To use compression, please install bitsandbytes: `pip install bitsandbytes`'
                )
        
        self.compression = compression
        self.hf_token = hf_token
        
        # Set layer names
        self.set_layer_names_dict()
        
        # Find or create local splitted path
        self.model_local_path, self.checkpoint_path = find_or_create_local_splitted_path(
            model_local_path_or_repo_id,
            layer_shards_saving_path,
            compression=compression,
            layer_names=self.layer_names_dict,
            hf_token=hf_token,
            delete_original=delete_original
        )
        
        self.running_device = device
        self.device = torch.device(self.running_device)
        self.running_dtype = dtype
        self.dtype = self.running_dtype
        
        # Load config
        if hf_token is not None:
            self.config = AutoConfig.from_pretrained(
                self.model_local_path, token=hf_token, trust_remote_code=True
            )
        else:
            self.config = AutoConfig.from_pretrained(
                self.model_local_path, trust_remote_code=True
            )
        
        self.generation_config = self.get_generation_config()
        
        # Load tokenizer and processor
        self.tokenizer = self.get_tokenizer(hf_token=hf_token)
        self.processor = self.get_processor(hf_token=hf_token)
        
        # Initialize model
        self.init_model()
        
        # Get layer count
        model_attr = self.model
        for attr_name in self.layer_names_dict["layer_prefix"].split("."):
            model_attr = getattr(model_attr, attr_name)
        layers_count = len(model_attr)
        
        # Build layer names list
        self.layer_names = self._build_layer_names(layers_count)
        
        self.max_seq_len = max_seq_len
        self.main_input_name = "input_ids"
        
        # Prefetching setup
        self.prefetching = prefetching
        if self.compression is not None:
            self.prefetching = False
            print(f"Compression mode enabled, prefetching disabled.")
        
        if prefetching and device.startswith("cuda"):
            self.stream = torch.cuda.Stream()
        else:
            self.stream = None
    
    def _build_layer_names(self, layers_count):
        """Build the list of layer names for the model."""
        layer_names = []
        
        # Add vision-specific layers if defined
        if self.layer_names_dict.get('vision_tower'):
            layer_names.append(self.layer_names_dict['vision_tower'])
        if self.layer_names_dict.get('vision_projector'):
            layer_names.append(self.layer_names_dict['vision_projector'])
        
        # Add standard transformer layers
        layer_names.append(self.layer_names_dict['embed'])
        layer_names.extend([
            f'{self.layer_names_dict["layer_prefix"]}.{i}' 
            for i in range(layers_count)
        ])
        layer_names.append(self.layer_names_dict['norm'])
        layer_names.append(self.layer_names_dict['lm_head'])
        
        return layer_names
    
    def get_generation_config(self):
        """Get generation configuration for the model."""
        try:
            return GenerationConfig.from_pretrained(self.model_local_path)
        except Exception:
            return GenerationConfig()
    
    def get_tokenizer(self, hf_token=None):
        """Get the tokenizer for the model."""
        if hf_token is not None:
            return AutoTokenizer.from_pretrained(
                self.model_local_path, token=hf_token, trust_remote_code=True
            )
        return AutoTokenizer.from_pretrained(
            self.model_local_path, trust_remote_code=True
        )
    
    def get_processor(self, hf_token=None):
        """
        Get the processor for handling multimodal inputs.
        
        For VLMs, the processor typically handles both text tokenization
        and image preprocessing.
        """
        try:
            if hf_token is not None:
                return AutoProcessor.from_pretrained(
                    self.model_local_path, token=hf_token, trust_remote_code=True
                )
            return AutoProcessor.from_pretrained(
                self.model_local_path, trust_remote_code=True
            )
        except Exception as e:
            print(f"Could not load processor, falling back to tokenizer only: {e}")
            return None
    
    def get_use_better_transformer(self):
        """Whether to use BetterTransformer optimization."""
        return False  # VLMs often need custom attention
    
    def init_model(self):
        """Initialize the model with empty weights."""
        self.model = None
        model_class = self.get_model_class()
        
        try:
            print(f"Initializing VLM with config: {type(self.config).__name__}")
            with init_empty_weights():
                self.model = model_class.from_config(self.config, trust_remote_code=True)
        except Exception as e:
            print(f"Could not initialize with model class, trying AutoModelForCausalLM: {e}")
            with init_empty_weights():
                self.model = AutoModelForCausalLM.from_config(self.config, trust_remote_code=True)
        
        self.model.eval()
        self.model.tie_weights()
        
        self.set_layers_from_layer_names()
        
        # Move buffers to device
        for buffer_name, buffer in self.model.named_buffers():
            set_module_tensor_to_device(
                self.model, buffer_name, self.running_device, 
                value=buffer, dtype=self.running_dtype
            )
    
    def set_layers_from_layer_names(self):
        """Set up layer references from layer names."""
        self.layers = []
        
        # Vision layers (if present)
        if self.layer_names_dict.get('vision_tower'):
            try:
                model_attr = self.model
                for attr_name in self.layer_names_dict["vision_tower"].split("."):
                    model_attr = getattr(model_attr, attr_name)
                self.layers.append(model_attr)
            except AttributeError:
                print(f"Vision tower not found in model architecture")
        
        if self.layer_names_dict.get('vision_projector'):
            try:
                model_attr = self.model
                for attr_name in self.layer_names_dict["vision_projector"].split("."):
                    model_attr = getattr(model_attr, attr_name)
                self.layers.append(model_attr)
            except AttributeError:
                print(f"Vision projector not found in model architecture")
        
        # Embedding layer
        model_attr = self.model
        for attr_name in self.layer_names_dict["embed"].split("."):
            model_attr = getattr(model_attr, attr_name)
        self.layers.append(model_attr)
        
        # Transformer layers
        model_attr = self.model
        for attr_name in self.layer_names_dict["layer_prefix"].split("."):
            model_attr = getattr(model_attr, attr_name)
        self.layers.extend(list(model_attr))
        
        # Norm layer
        model_attr = self.model
        for attr_name in self.layer_names_dict["norm"].split("."):
            model_attr = getattr(model_attr, attr_name)
        self.layers.append(model_attr)
        
        # LM head
        model_attr = self.model
        for attr_name in self.layer_names_dict["lm_head"].split("."):
            model_attr = getattr(model_attr, attr_name)
        self.layers.append(model_attr)
    
    def load_layer_to_cpu(self, layer_name):
        """Load a layer to CPU memory."""
        t = time.time()
        
        load_layer_output = load_layer(self.checkpoint_path, layer_name, self.profiling_mode)
        elapsed_time = time.time() - t
        
        if self.profiling_mode:
            state_dict, compression_time = load_layer_output
            disk_loading_time = elapsed_time - compression_time
            self.profiler.add_profiling_time('load_safe_tensor', disk_loading_time)
            self.profiler.add_profiling_time('compression_time', compression_time)
        else:
            state_dict = load_layer_output
        
        # Pin memory for prefetching
        if self.prefetching:
            t = time.time()
            if torch.cuda.is_available():
                for k in state_dict.keys():
                    state_dict[k].pin_memory()
            elapsed_time = time.time() - t
            if self.profiling_mode:
                self.profiler.add_profiling_time('pin_memory_to_trigger_load', elapsed_time)
        
        return state_dict
    
    def move_layer_to_device(self, state_dict):
        """Move layer weights to the running device."""
        layers = []
        for param_name, param in state_dict.items():
            if self.hf_quantizer is None:
                layers.append(param_name)
            else:
                if '.weight' in param_name:
                    layer_name = param_name[:param_name.index(".weight") + len(".weight")]
                    if layer_name not in layers:
                        layers.append(layer_name)
        
        for param_name in layers:
            if (self.hf_quantizer is None or
                not self.hf_quantizer.check_quantized_param(
                    self.model, param_value=None, param_name=param_name, state_dict={}
                )):
                set_module_tensor_to_device(
                    self.model, param_name, self.running_device, 
                    value=state_dict[param_name], dtype=self.running_dtype
                )
            else:
                self.hf_quantizer.create_quantized_param(
                    self.model, state_dict[param_name], param_name, 
                    self.running_device, state_dict
                )
        return layers
    
    def can_generate(self):
        """Check if the model can generate text."""
        return True
    
    def process_images(self, images, **kwargs):
        """
        Process images for the model.
        
        This method should be overridden by subclasses to handle
        model-specific image preprocessing.
        
        Parameters
        ----------
        images : Union[Image, List[Image]]
            Input image(s) to process
        
        Returns
        -------
        dict
            Processed image inputs ready for the model
        """
        if self.processor is not None:
            return self.processor(images=images, return_tensors="pt")
        raise NotImplementedError(
            "Subclasses must implement process_images() or provide a processor"
        )
    
    def prepare_inputs_for_generation(
        self, 
        input_ids, 
        past_key_values=None, 
        attention_mask=None, 
        inputs_embeds=None,
        pixel_values=None,
        image_sizes=None,
        **kwargs
    ):
        """
        Prepare inputs for text generation.
        
        This method extends the base implementation to support
        multimodal inputs including pixel_values for images.
        """
        if past_key_values is not None:
            past_length = self.get_past_key_values_cache_seq_len(past_key_values)
            
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                remove_prefix_length = input_ids.shape[1] - 1
            
            input_ids = input_ids[:, remove_prefix_length:]
        
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]
        
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        
        model_inputs.update({
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        })
        
        # Add vision inputs if present (only for first generation step)
        if past_key_values is None:
            if pixel_values is not None:
                model_inputs["pixel_values"] = pixel_values
            if image_sizes is not None:
                model_inputs["image_sizes"] = image_sizes
        
        return model_inputs
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def get_past_key_values_cache_seq_len(self, past_key_values):
        """Get the sequence length from past key values cache."""
        return past_key_values[0][0].shape[2]
    
    def get_sequence_len(self, seq):
        """Get the sequence length."""
        return seq.shape[1]
    
    def get_pos_emb_args(self, len_p, len_s):
        """Get positional embedding arguments."""
        return {}
    
    def get_past_key_value_args(self, k_cache, v_cache):
        """Get past key value arguments."""
        return {'past_key_value': (k_cache, v_cache)}
    
    def get_attention_mask_args(self, full_attention_mask, len_p, len_s):
        """Get attention mask arguments."""
        return {'attention_mask': full_attention_mask[:, :, -len_s:, -len_p - len_s:]}
    
    def get_position_ids_args(self, full_position_ids, len_p, len_s):
        """Get position IDs arguments."""
        return {'position_ids': full_position_ids[:, len_p:len_p + len_s]}
    
    def run_lm_head(self, layer, seq):
        """Run the language model head."""
        return layer(seq).float()
    
    def run_norm(self, layer, seq):
        """Run the normalization layer."""
        return layer(seq)
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Forward pass for Vision Language Models.
        
        This extends the base forward pass to handle multimodal inputs,
        including pixel_values for image data.
        
        Parameters
        ----------
        input_ids : torch.LongTensor, optional
            Input token IDs
        attention_mask : torch.Tensor, optional
            Attention mask
        position_ids : torch.LongTensor, optional
            Position IDs
        past_key_values : List[torch.FloatTensor], optional
            Past key values for caching
        inputs_embeds : torch.FloatTensor, optional
            Input embeddings
        pixel_values : torch.FloatTensor, optional
            Pixel values for images (batch, channels, height, width)
        image_sizes : List[List[int]], optional
            Sizes of input images
        labels : torch.LongTensor, optional
            Labels for computing loss
        use_cache : bool, optional
            Whether to use key-value caching
        output_attentions : bool, optional
            Whether to output attention weights
        output_hidden_states : bool, optional
            Whether to output hidden states
        return_dict : bool, optional
            Whether to return a dict
        
        Returns
        -------
        Union[Tuple, CausalLMOutputWithPast]
            Model outputs
        """
        if cache_utils_installed:
            use_cache = False
        
        if self.profiling_mode:
            self.profiler.clear_profiling_time()
            forward_start = time.process_time()
            forward_start_wall = time.time()
        
        # Reinitialize model
        del self.model
        clean_memory()
        self.init_model()
        
        batch = [input_ids_unit.to(self.running_device).unsqueeze(0) for input_ids_unit in input_ids]
        n_seq = len(batch[0])
        
        # Handle pixel values if present
        if pixel_values is not None:
            pixel_values = pixel_values.to(self.running_device).to(self.running_dtype)
        
        # Create attention mask
        attention_mask = torch.ones(self.max_seq_len, self.max_seq_len)
        attention_mask = attention_mask.triu(diagonal=1)[None, None, ...] == 0
        attention_mask = attention_mask.to(self.running_device)
        position_ids = torch.arange(self.max_seq_len, dtype=torch.long, device=self.running_device)[None, :]
        
        kv_cache_list = [] if use_cache else None
        if use_cache:
            for x in self.layers:
                kv_cache_list.append(([], []))
        
        all_hidden_states = [[] for _ in range(len(self.layers))] if output_hidden_states else None
        all_self_attns = [[] for _ in range(len(self.layers))] if output_attentions else None
        
        # Store image features for later use
        image_features = None
        
        with torch.inference_mode(), ThreadPoolExecutor() as executor:
            # Load first layer
            if self.prefetching:
                future = executor.submit(self.load_layer_to_cpu, self.layer_names[0])
            
            for i, (layer_name, layer) in tqdm(
                enumerate(zip(self.layer_names, self.layers)),
                desc=f'running layers({self.running_device})',
                total=len(self.layers)
            ):
                if self.prefetching:
                    if self.profiling_mode:
                        t = time.time()
                    state_dict = future.result()
                    if self.profiling_mode:
                        elapsed_time = time.time() - t
                        self.profiler.add_profiling_time('load_safe_tensor_cpu_wait', elapsed_time)
                    
                    if self.profiling_mode:
                        t = time.time()
                    moved_layers = self.move_layer_to_device(state_dict)
                    if self.profiling_mode:
                        elapsed_time = time.time() - t
                        self.profiler.add_profiling_time('create_layer_from_state_dict', elapsed_time)
                    
                    # Kick off next layer loading
                    if (i + 1) < len(self.layer_names):
                        if self.profiling_mode:
                            t = time.time()
                        future = executor.submit(self.load_layer_to_cpu, self.layer_names[i + 1])
                        if self.profiling_mode:
                            elapsed_time = time.time() - t
                            self.profiler.add_profiling_time('kick_off_load_cpu', elapsed_time)
                else:
                    state_dict = self.load_layer_to_cpu(layer_name)
                    if self.profiling_mode:
                        t = time.time()
                    moved_layers = self.move_layer_to_device(state_dict)
                    if self.profiling_mode:
                        elapsed_time = time.time() - t
                        self.profiler.add_profiling_time('create_layer_from_safe_tensor', elapsed_time)
                
                # Run layer
                for j, seq in enumerate(batch):
                    # Handle vision tower
                    if (self.layer_names_dict.get('vision_tower') and 
                        layer_name == self.layer_names_dict['vision_tower']):
                        if pixel_values is not None:
                            image_features = layer(pixel_values)
                    
                    # Handle vision projector
                    elif (self.layer_names_dict.get('vision_projector') and 
                          layer_name == self.layer_names_dict['vision_projector']):
                        if image_features is not None:
                            image_features = layer(image_features)
                    
                    # Handle embedding layer
                    elif layer_name == self.layer_names_dict['embed']:
                        batch[j] = layer(seq)
                        # Merge image features with text embeddings if available
                        if image_features is not None:
                            batch[j] = self._merge_image_text_embeddings(
                                batch[j], image_features, input_ids[j]
                            )
                    
                    # Handle norm layer
                    elif layer_name == self.layer_names_dict['norm']:
                        batch[j] = self.run_norm(layer, seq)
                        if output_attentions:
                            all_hidden_states[i].append(batch[j])
                    
                    # Handle LM head
                    elif layer_name == self.layer_names_dict['lm_head']:
                        batch[j] = self.run_lm_head(layer, seq)
                    
                    # Handle transformer layers
                    else:
                        if output_attentions:
                            all_hidden_states[i].append(seq)
                        
                        if past_key_values is not None:
                            k_cache, v_cache = past_key_values[i - 1]
                            len_p = self.get_past_key_values_cache_seq_len(past_key_values)
                            len_s = self.get_sequence_len(seq)
                            
                            position_ids_args = self.get_position_ids_args(position_ids, len_p, len_s)
                            attention_mask_args = self.get_attention_mask_args(attention_mask, len_p, len_s)
                            past_key_value_args = self.get_past_key_value_args(k_cache, v_cache)
                            
                            kwargs = {'use_cache': True}
                            pos_embed_args = self.get_pos_emb_args(len_p, len_s)
                            kwargs = {
                                **kwargs, **past_key_value_args, **pos_embed_args, 
                                **attention_mask_args, **position_ids_args
                            }
                            
                            layer_outputs = layer(seq, **kwargs)
                            new_seq = layer_outputs[0]
                            
                            if output_attentions:
                                all_self_attns[i].append(layer_outputs[1])
                            
                            if use_cache:
                                (k_cache, v_cache) = layer_outputs[2 if output_attentions else 1]
                                kv_cache_list[i][0].append(k_cache)
                                kv_cache_list[i][1].append(v_cache)
                        else:
                            len_seq = self.get_sequence_len(seq)
                            
                            pos_embed_args = self.get_pos_emb_args(0, len_seq)
                            attention_mask_args = self.get_attention_mask_args(attention_mask, 0, len_seq)
                            position_ids_args = self.get_position_ids_args(position_ids, 0, len_seq)
                            
                            if not use_cache:
                                kwargs = {
                                    'use_cache': False,
                                    'attention_mask': attention_mask[:, :, -len_seq:, -len_seq:],
                                }
                                kwargs = {**kwargs, **pos_embed_args, **attention_mask_args, **position_ids_args}
                                new_seq = layer(seq, **kwargs)[0]
                            else:
                                kwargs = {
                                    'use_cache': True,
                                    'attention_mask': attention_mask[:, :, -len_seq:, -len_seq:],
                                }
                                kwargs = {**kwargs, **pos_embed_args, **attention_mask_args, **position_ids_args}
                                layer_out = layer(seq, **kwargs)
                                new_seq, (k_cache, v_cache) = layer_out
                                kv_cache_list[i][0].append(k_cache)
                                kv_cache_list[i][1].append(v_cache)
                        
                        batch[j] = new_seq
                
                if output_hidden_states:
                    all_hidden_states += (torch.cat(batch, 0),)
                
                # Clean up layer
                if self.hf_quantizer is not None:
                    for param_name in moved_layers:
                        set_module_tensor_to_device(self.model, param_name, 'meta')
                else:
                    layer.to("meta")
                
                layer.to("meta")
                clean_memory()
        
        logits = torch.cat(batch, 0)
        
        if use_cache:
            kv_cache_list = kv_cache_list[1:-2]
            for i in range(len(kv_cache_list)):
                kv_cache_list[i] = (
                    torch.cat(kv_cache_list[i][0], 0), 
                    torch.cat(kv_cache_list[i][1], 0)
                )
        
        if output_attentions:
            all_self_attns = all_self_attns[0:-2]
            for i in range(len(all_self_attns)):
                all_self_attns[i] = torch.cat(all_self_attns[i], 0)
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states[0:-2]
            for i in range(len(all_hidden_states)):
                all_hidden_states[i] = torch.cat(all_hidden_states[i], 0)
        
        if not return_dict:
            return tuple(
                v for v in [
                    logits,
                    tuple(kv_cache_list) if kv_cache_list is not None else None,
                    tuple(all_hidden_states) if all_hidden_states is not None else None,
                    tuple(all_self_attns) if all_self_attns is not None else None
                ] if v is not None
            )
        
        if self.profiling_mode:
            forward_elapsed_time = time.process_time() - forward_start
            forward_elapsed_time_wall = time.time() - forward_start_wall
            self.profiler.print_profiling_time()
            print(f"total infer process time: {forward_elapsed_time:.04f}")
            print(f"total infer wall time: {forward_elapsed_time_wall:.04f}")
            self.profiler.clear_profiling_time()
        
        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=tuple(kv_cache_list) if kv_cache_list is not None else None,
            hidden_states=tuple(all_hidden_states) if all_hidden_states is not None else None,
            attentions=tuple(all_self_attns) if all_hidden_states is not None else None,
        )
    
    def _merge_image_text_embeddings(self, text_embeds, image_features, input_ids):
        """
        Merge image features with text embeddings.
        
        This method should be overridden by subclasses to implement
        model-specific image-text embedding merging logic.
        
        Parameters
        ----------
        text_embeds : torch.Tensor
            Text embeddings from the embedding layer
        image_features : torch.Tensor
            Image features from the vision encoder/projector
        input_ids : torch.Tensor
            Original input token IDs (for finding image token positions)
        
        Returns
        -------
        torch.Tensor
            Merged embeddings
        """
        # Default implementation: prepend image features to text embeddings
        return torch.cat([image_features, text_embeds], dim=1)
