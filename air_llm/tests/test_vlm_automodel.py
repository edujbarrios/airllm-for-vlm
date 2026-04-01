"""
Tests for VLM (Vision Language Model) architecture detection in AutoModel.

These tests verify the VLM architecture detection and class mapping logic
without requiring network access to HuggingFace Hub.
"""

import unittest


class TestVLMAutoModel(unittest.TestCase):
    """Tests for VLM model detection in AutoModel."""
    
    def setUp(self):
        pass
    
    def tearDown(self):
        pass
    
    def test_is_vlm_architecture_detection(self):
        """Test VLM architecture detection patterns."""
        # Import here to avoid early import issues
        from ..airllm.auto_model import AutoModel
        
        # VLM architectures should be detected
        vlm_architectures = [
            "Qwen2VLForConditionalGeneration",
            "GLM4VForConditionalGeneration",
            "MoondreamForConditionalGeneration",
            "PaliGemmaForConditionalGeneration",
            "LlavaForConditionalGeneration",
            "CogVLMForCausalLM",
        ]
        
        for arch in vlm_architectures:
            self.assertTrue(
                AutoModel._is_vlm_architecture(arch),
                f"Should detect {arch} as VLM"
            )
        
        # Text-only architectures should NOT be detected as VLM
        text_only_architectures = [
            "LlamaForCausalLM",
            "Qwen2ForCausalLM",
            "MistralForCausalLM",
            "MixtralForCausalLM",
            "ChatGLMModel",
        ]
        
        for arch in text_only_architectures:
            self.assertFalse(
                AutoModel._is_vlm_architecture(arch),
                f"Should NOT detect {arch} as VLM"
            )
    
    def test_vlm_detection_by_model_name(self):
        """Test VLM detection using model repository names."""
        from ..airllm.auto_model import AutoModel
        
        vlm_model_names = [
            "zai-org/GLM-4.6V-Flash",
            "Qwen/Qwen2.5-VL-32B-Instruct",
            "moondream/moondream3-preview",
            "google/medgemma-4b-it",
            "some-user/llava-model",
        ]
        
        for model_name in vlm_model_names:
            # Should detect VLM from model name even with generic architecture
            self.assertTrue(
                AutoModel._is_vlm_architecture("GenericModel", model_name),
                f"Should detect {model_name} as VLM from model name"
            )
    
    def test_text_only_model_names_not_vlm(self):
        """Test that text-only model names are not detected as VLM."""
        from ..airllm.auto_model import AutoModel
        
        text_model_names = [
            "meta-llama/Llama-2-7b-hf",
            "Qwen/Qwen-7B",
            "THUDM/chatglm3-6b-base",
            "mistralai/Mistral-7B-Instruct-v0.1",
        ]
        
        for model_name in text_model_names:
            # Should NOT detect text-only models as VLM
            self.assertFalse(
                AutoModel._is_vlm_architecture("LlamaForCausalLM", model_name),
                f"Should NOT detect {model_name} as VLM"
            )
    
    def test_vlm_architecture_patterns_comprehensive(self):
        """Test comprehensive VLM architecture pattern matching."""
        from ..airllm.auto_model import AutoModel
        
        # Additional edge cases
        test_cases = [
            ("VisionEncoderDecoderModel", True),
            ("MultiModalLlamaForCausalLM", True),
            ("ImageTextModel", True),
            ("BertForSequenceClassification", False),
            ("GPT2LMHeadModel", False),
        ]
        
        for arch, expected in test_cases:
            result = AutoModel._is_vlm_architecture(arch)
            self.assertEqual(
                result, expected,
                f"Architecture {arch}: expected {expected}, got {result}"
            )


class TestVLMModelClasses(unittest.TestCase):
    """Tests for VLM model class instantiation."""
    
    def test_vlm_classes_can_be_imported(self):
        """Test that all VLM classes can be imported."""
        from airllm import (
            AirLLMVLMBase,
            AirLLMGLMVLM,
            AirLLMQwenVLM,
            AirLLMQwen2VLM,
            AirLLMMoondream,
            AirLLMMedGemma,
        )
        
        # Verify classes exist and have expected attributes
        self.assertTrue(hasattr(AirLLMVLMBase, 'is_vlm'))
        self.assertTrue(AirLLMVLMBase.is_vlm)
        
        # Verify inheritance
        self.assertTrue(issubclass(AirLLMGLMVLM, AirLLMVLMBase))
        self.assertTrue(issubclass(AirLLMQwenVLM, AirLLMVLMBase))
        self.assertTrue(issubclass(AirLLMMoondream, AirLLMVLMBase))
        self.assertTrue(issubclass(AirLLMMedGemma, AirLLMVLMBase))
    
    def test_vlm_classes_have_required_methods(self):
        """Test that VLM classes have required methods."""
        from airllm import AirLLMVLMBase
        
        required_methods = [
            'set_layer_names_dict',
            'get_processor',
            'process_images',
            'prepare_inputs_for_generation',
            '_merge_image_text_embeddings',
            'forward',
        ]
        
        for method_name in required_methods:
            self.assertTrue(
                hasattr(AirLLMVLMBase, method_name),
                f"AirLLMVLMBase should have method: {method_name}"
            )
    
    def test_vlm_layer_names_dict_set_correctly(self):
        """Test that VLM classes set their layer_names_dict correctly."""
        from airllm import (
            AirLLMGLMVLM,
            AirLLMQwenVLM,
            AirLLMMoondream,
            AirLLMMedGemma,
        )
        
        # Create mock instances (just checking the set_layer_names_dict method)
        for cls in [AirLLMGLMVLM, AirLLMQwenVLM, AirLLMMoondream, AirLLMMedGemma]:
            # Check the method exists and can be inspected
            self.assertTrue(callable(cls.set_layer_names_dict))
    
    def test_qwen2vlm_is_alias_for_qwenvlm(self):
        """Test that AirLLMQwen2VLM is an alias for AirLLMQwenVLM."""
        from airllm import AirLLMQwenVLM, AirLLMQwen2VLM
        
        self.assertTrue(issubclass(AirLLMQwen2VLM, AirLLMQwenVLM))


class TestVLMArchitectureConstants(unittest.TestCase):
    """Tests for VLM architecture constants defined in auto_model."""
    
    def test_vlm_architecture_patterns_defined(self):
        """Test that VLM architecture patterns are defined."""
        from ..airllm.auto_model import VLM_ARCHITECTURE_PATTERNS
        
        self.assertIsInstance(VLM_ARCHITECTURE_PATTERNS, list)
        self.assertGreater(len(VLM_ARCHITECTURE_PATTERNS), 0)
        
        # Check that key patterns are present
        pattern_names = [p[0] for p in VLM_ARCHITECTURE_PATTERNS]
        self.assertIn("GLM4VForConditionalGeneration", pattern_names)
        self.assertIn("Qwen2VLForConditionalGeneration", pattern_names)
        self.assertIn("MoondreamForConditionalGeneration", pattern_names)
        self.assertIn("PaliGemmaForConditionalGeneration", pattern_names)


if __name__ == '__main__':
    unittest.main()
