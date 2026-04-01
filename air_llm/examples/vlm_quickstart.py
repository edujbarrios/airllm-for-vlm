"""
AirLLM Vision Language Model Quick Start Example

This example demonstrates how to use AirLLM with various Vision Language Models
to perform image understanding tasks on memory-constrained GPUs.
"""

from airllm import AutoModel
from PIL import Image
import sys


def run_qwen_vlm_example(image_path: str):
    """Example using Qwen2.5-VL model."""
    print("Loading Qwen2.5-VL model...")
    model = AutoModel.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    
    image = Image.open(image_path)
    
    inputs = model.processor(
        text="Describe this image in detail.",
        images=image,
        return_tensors="pt"
    )
    
    print("Generating response...")
    outputs = model.generate(
        input_ids=inputs['input_ids'].cuda(),
        pixel_values=inputs['pixel_values'].cuda(),
        max_new_tokens=256,
        use_cache=False,
        return_dict_in_generate=True
    )
    
    response = model.tokenizer.decode(outputs.sequences[0])
    print(f"\nQwen2.5-VL Response:\n{response}")
    return response


def run_glm4v_example(image_path: str):
    """Example using GLM-4V model."""
    print("Loading GLM-4V model...")
    model = AutoModel.from_pretrained("zai-org/GLM-4.6V-Flash")
    
    image = Image.open(image_path)
    
    messages = "<|user|>\n<image>\nWhat do you see in this image?\n<|assistant|>\n"
    
    inputs = model.processor(
        text=messages,
        images=image,
        return_tensors="pt"
    )
    
    print("Generating response...")
    outputs = model.generate(
        **{k: v.cuda() for k, v in inputs.items()},
        max_new_tokens=200
    )
    
    response = model.tokenizer.decode(outputs[0])
    print(f"\nGLM-4V Response:\n{response}")
    return response


def run_moondream_example(image_path: str):
    """Example using Moondream model."""
    print("Loading Moondream model...")
    model = AutoModel.from_pretrained("moondream/moondream3-preview")
    
    image = Image.open(image_path)
    
    inputs = model.processor(
        images=image,
        text="<image>\n\nDescribe what's happening in this image.",
        return_tensors="pt"
    )
    
    print("Generating response...")
    outputs = model.generate(
        **{k: v.cuda() for k, v in inputs.items()},
        max_new_tokens=256
    )
    
    response = model.tokenizer.decode(outputs[0])
    print(f"\nMoondream Response:\n{response}")
    return response


def run_with_compression(image_path: str):
    """Example using 4-bit compression for faster inference."""
    print("Loading Qwen2.5-VL with 4-bit compression...")
    model = AutoModel.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        compression='4bit'  # Enable 4-bit compression for 3x faster inference
    )
    
    image = Image.open(image_path)
    
    inputs = model.processor(
        text="What objects can you identify in this image?",
        images=image,
        return_tensors="pt"
    )
    
    print("Generating response with compressed model...")
    outputs = model.generate(
        input_ids=inputs['input_ids'].cuda(),
        pixel_values=inputs['pixel_values'].cuda(),
        max_new_tokens=256,
        use_cache=False,
        return_dict_in_generate=True
    )
    
    response = model.tokenizer.decode(outputs.sequences[0])
    print(f"\nCompressed Model Response:\n{response}")
    return response


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python vlm_quickstart.py <image_path> [model]")
        print("\nAvailable models: qwen, glm4v, moondream, compressed")
        print("\nExample: python vlm_quickstart.py photo.jpg qwen")
        sys.exit(1)
    
    image_path = sys.argv[1]
    model_choice = sys.argv[2].lower() if len(sys.argv) > 2 else "qwen"
    
    if model_choice == "qwen":
        run_qwen_vlm_example(image_path)
    elif model_choice == "glm4v":
        run_glm4v_example(image_path)
    elif model_choice == "moondream":
        run_moondream_example(image_path)
    elif model_choice == "compressed":
        run_with_compression(image_path)
    else:
        print(f"Unknown model: {model_choice}")
        print("Available models: qwen, glm4v, moondream, compressed")
        sys.exit(1)
