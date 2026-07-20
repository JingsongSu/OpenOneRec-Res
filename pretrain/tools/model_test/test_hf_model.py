#!/usr/bin/env python3
"""HuggingFace Model Testing Tool

A unified tool for testing HuggingFace models with both direct text generation
and chat template modes. Supports thinking mode and ground truth comparison.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional, Union

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def load_model(
    model_path: str,
    device: str = "auto",
    torch_dtype: torch.dtype = torch.bfloat16
) -> tuple:
    """Load HuggingFace model and tokenizer.
    
    Args:
        model_path: Path to model directory
        device: Device mapping (default: "auto")
        torch_dtype: Data type for model (default: bfloat16)
    
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model from: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    logger.info("Tokenizer loaded")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device,
        trust_remote_code=True
    )
    logger.info("Model loaded")
    
    return model, tokenizer


def print_model_info(model) -> None:
    """Print model information.
    
    Args:
        model: Loaded model instance
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    
    logger.info("=" * 60)
    logger.info("Model Information:")
    logger.info(f"  Device: {device}")
    logger.info(f"  Data Type: {dtype}")
    logger.info(f"  Vocab Size: {model.config.vocab_size}")
    logger.info(f"  Hidden Size: {model.config.hidden_size}")
    if hasattr(model.config, 'num_hidden_layers'):
        logger.info(f"  Num Layers: {model.config.num_hidden_layers}")
    logger.info("=" * 60)


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    do_sample: bool = True,
    show_input_ids: bool = False
) -> str:
    """Generate text from a direct prompt (without chat template).
    
    Args:
        model: Model instance
        tokenizer: Tokenizer instance
        prompt: Input prompt text
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        repetition_penalty: Repetition penalty
        do_sample: Whether to use sampling
        show_input_ids: Whether to print input token IDs
    
    Returns:
        Generated text (only the newly generated part)
    """
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    if show_input_ids:
        logger.info(f"Input IDs: {inputs['input_ids']}")
    
    with torch.no_grad():
        generate_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample
        )
    
    output = tokenizer.batch_decode(
        generate_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False
    )[0]
    
    # Return only the newly generated part
    generated_text = output[len(prompt):].strip()
    return generated_text


def generate_chat(
    model,
    tokenizer,
    messages: List[dict],
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.2,
    enable_thinking: bool = False,
    add_generation_prompt: bool = True,
    show_template: bool = False
) -> str:
    """Generate text using chat template.
    
    Args:
        model: Model instance
        tokenizer: Tokenizer instance
        messages: List of message dicts with 'role' and 'content' keys
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        repetition_penalty: Repetition penalty
        enable_thinking: Whether to enable thinking mode
        add_generation_prompt: Whether to add generation prompt
        show_template: Whether to print the formatted template
    
    Returns:
        Generated text (only the newly generated part)
    """
    # Apply chat template
    template_kwargs = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    
    if enable_thinking:
        template_kwargs["enable_thinking"] = True
    
    text = tokenizer.apply_chat_template(messages, **template_kwargs)
    
    if show_template:
        logger.info(f"Chat Template:\n{text}\n" + "=" * 60)
    
    # Tokenize and generate
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=False,
        truncation=False
    )
    
    device = next(model.parameters()).device
    inputs = inputs.to(device)
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=True
        )
    
    output_text = tokenizer.batch_decode(
        output,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False
    )[0]
    
    # Return only the newly generated part
    generated_text = output_text[len(text):].strip()
    return generated_text


def load_test_cases_from_file(file_path: Union[str, Path]) -> tuple:
    """Load test cases from JSON file.
    
    Expected format:
    {
        "test_cases": [
            {
                "type": "text" or "chat",
                "input": "prompt text" or [{"role": "...", "content": "..."}],
                "ground_truth": "expected output" (optional)
            }
        ]
    }
    
    Args:
        file_path: Path to JSON file
    
    Returns:
        Tuple of (test_cases, ground_truths)
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    test_cases = []
    ground_truths = []
    
    for item in data.get("test_cases", []):
        test_cases.append({
            "type": item.get("type", "text"),
            "input": item["input"]
        })
        ground_truths.append(item.get("ground_truth", ""))
    
    return test_cases, ground_truths


def get_default_test_cases() -> tuple:
    """Get default test cases for demonstration.
    
    Returns:
        Tuple of (test_cases, ground_truths)
    """
    test_cases = [
        {
            "type": "text",
            "input": "你好，请介绍一下你自己。"
        },
        {
            "type": "text",
            "input": "视频<|sid_begin|><s_a_8084><s_b_243><s_c_2535><|sid_end|>的类型是："
        },
        {
            "type": "chat",
            "input": [{"role": "user", "content": "写一首关于春天的短诗："}]
        },
        {
            "type": "chat",
            "input": [
                {"role": "system", "content": "你是一名视频描述生成器，请根据下面的视频token生成视频描述"},
                {"role": "user", "content": "这是一个视频：<|sid_begin|><s_a_3482><s_b_3606><s_c_3239><|sid_end|>，帮我总结一下这个视频讲述了什么内容"}
            ]
        },
    ]
    
    ground_truths = ["", "", "", ""]
    
    return test_cases, ground_truths


def main():
    parser = argparse.ArgumentParser(
        description="Test HuggingFace models with text generation or chat mode"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to HuggingFace model directory"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device mapping (default: auto)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["fp32", "fp16", "bf16"],
        help="Model data type (default: bf16)"
    )
    
    # Test case arguments
    parser.add_argument(
        "--test_file",
        type=str,
        default=None,
        help="Path to JSON file containing test cases (optional)"
    )
    parser.add_argument(
        "--use_default",
        action="store_true",
        help="Use default test cases if no test file provided"
    )
    
    # Generation arguments
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Maximum number of tokens to generate (default: 1024)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter (default: 0.9)"
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.2,
        help="Repetition penalty (default: 1.2)"
    )
    
    # Chat mode arguments
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Enable thinking mode for chat template"
    )
    parser.add_argument(
        "--no_generation_prompt",
        dest="add_generation_prompt",
        action="store_false",
        help="Disable generation prompt in chat template"
    )
    parser.add_argument(
        "--show_template",
        action="store_true",
        help="Show formatted chat template"
    )
    parser.add_argument(
        "--show_input_ids",
        action="store_true",
        help="Show input token IDs for text mode"
    )
    
    # Output arguments
    parser.add_argument(
        "--compare_ground_truth",
        action="store_true",
        help="Compare output with ground truth if available"
    )
    
    args = parser.parse_args()
    
    # Convert dtype string to torch.dtype
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    torch_dtype = dtype_map[args.dtype]
    
    # Load model
    model, tokenizer = load_model(args.model_path, args.device, torch_dtype)
    print_model_info(model)
    
    # Load test cases
    if args.test_file:
        logger.info(f"Loading test cases from: {args.test_file}")
        test_cases, ground_truths = load_test_cases_from_file(args.test_file)
    elif args.use_default:
        logger.info("Using default test cases")
        test_cases, ground_truths = get_default_test_cases()
    else:
        logger.error("Either --test_file or --use_default must be provided")
        sys.exit(1)
    
    logger.info(f"Loaded {len(test_cases)} test cases\n")
    
    # Run tests
    logger.info("Starting tests...\n")
    for i, (test_case, ground_truth) in enumerate(zip(test_cases, ground_truths), 1):
        logger.info("=" * 60)
        logger.info(f"Test {i}/{len(test_cases)}")
        logger.info("=" * 60)
        
        test_type = test_case["type"]
        test_input = test_case["input"]
        
        # Display input
        if test_type == "text":
            logger.info(f"Input (text): {test_input}\n")
        else:
            logger.info(f"Input (chat):")
            for msg in test_input:
                logger.info(f"  {msg['role']}: {msg['content'][:100]}...")
            logger.info("")
        
        try:
            # Generate
            if test_type == "text":
                generated = generate_text(
                    model,
                    tokenizer,
                    test_input,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                    show_input_ids=args.show_input_ids
                )
            else:  # chat mode
                generated = generate_chat(
                    model,
                    tokenizer,
                    test_input,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                    enable_thinking=args.enable_thinking,
                    add_generation_prompt=args.add_generation_prompt,
                    show_template=args.show_template
                )
            
            logger.info(f"Output: {generated}\n")
            
            # Compare with ground truth if available
            if args.compare_ground_truth and ground_truth:
                logger.info(f"Ground Truth: {ground_truth}\n")
                if generated.strip() == ground_truth.strip():
                    logger.info("✓ Match with ground truth")
                else:
                    logger.info("✗ Does not match ground truth")
            
        except Exception as e:
            logger.error(f"Generation failed: {e}", exc_info=True)
        
        logger.info("-" * 60 + "\n")
    
    logger.info("=" * 60)
    logger.info("All tests completed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

