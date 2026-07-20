"""Qwen3 Vocabulary Expansion Tool

Expand the standard Qwen3 HuggingFace checkpoint vocabulary to support post-training.
Add new tokens and adjust model vocabulary size (aligned to multiples of 256).
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import List

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def _align_vocab_size(vocab_size: int, alignment: int = 256) -> int:
    """Align vocabulary size to the nearest multiple of alignment.
    
    Args:
        vocab_size: Current vocabulary size
        alignment: Alignment value (default: 256)
        
    Returns:
        Aligned vocabulary size
    """
    return ((vocab_size + alignment - 1) // alignment) * alignment


def _fix_chat_template(reco_model_dir: str, hf_model_dir: str) -> None:
    """Fix chat template in tokenizer config by copying from original model.
    
    Args:
        reco_model_dir: Output model directory
        hf_model_dir: Original HuggingFace model directory
    """
    reco_tokenizer_config_path = os.path.join(reco_model_dir, "tokenizer_config.json")
    hf_tokenizer_config_path = os.path.join(hf_model_dir, "tokenizer_config.json")
    
    if not os.path.exists(hf_tokenizer_config_path):
        logger.warning(f"Original tokenizer_config.json not found: {hf_tokenizer_config_path}")
        return
    
    if not os.path.exists(reco_tokenizer_config_path):
        logger.warning(f"Output tokenizer_config.json not found: {reco_tokenizer_config_path}")
        return
    
    # Load configs
    with open(reco_tokenizer_config_path, "r", encoding="utf-8") as f:
        reco_config = json.load(f)
    
    with open(hf_tokenizer_config_path, "r", encoding="utf-8") as f:
        hf_config = json.load(f)
    
    # Copy chat template from original
    if "chat_template" in hf_config:
        reco_config["chat_template"] = hf_config["chat_template"]
        
        with open(reco_tokenizer_config_path, "w", encoding="utf-8") as f:
            json.dump(reco_config, f, indent=2, ensure_ascii=False)
        
        logger.info("Chat template copied from original model")


def _test_expanded_vocab(model, tokenizer, new_tokens: List[str]) -> None:
    """Test the expanded vocabulary with sample tokens.
    
    Args:
        model: Expanded model
        tokenizer: Expanded tokenizer
        new_tokens: List of newly added tokens
    """
    if not new_tokens:
        logger.info("No new tokens to test")
        return
    
    # Sample 3-5 tokens from new_tokens
    num_samples = min(random.randint(3, 5), len(new_tokens))
    sampled_tokens = random.sample(new_tokens, num_samples)
    input_text = " ".join(sampled_tokens) + " Hello world"
    
    try:
        input_ids = tokenizer.encode(input_text, return_tensors='pt')
        
        # Test generation (use eval mode to avoid training-specific behavior)
        model.eval()
        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=10, do_sample=False)
        
        logger.info("Vocabulary expansion test:")
        logger.info(f"  Input text: {input_text}")
        logger.info(f"  Decoded input: {tokenizer.decode(input_ids[0], skip_special_tokens=True)}")
        logger.info(f"  Input IDs shape: {input_ids.shape}")
        logger.info(f"  Generated: {tokenizer.decode(output[0], skip_special_tokens=True)}")
        
    except Exception as e:
        logger.warning(f"Vocabulary test failed: {e}")


def expand_qwen3_vocab_for_pretraining(
    hf_model_dir: str,
    output_model_dir: str,
    new_tokens: List[str]
) -> None:
    """Expand Qwen3 vocabulary for pretraining by adding new tokens.
    
    This function:
    1. Loads the original Qwen3 model and tokenizer
    2. Adds new tokens to the tokenizer
    3. Resizes model embeddings to aligned vocabulary size (multiple of 256)
    4. Updates model configuration
    5. Saves the expanded model, tokenizer, and config
    6. Fixes chat template from original model
    7. Tests the expanded vocabulary
    
    Args:
        hf_model_dir: Path to original HuggingFace model directory
        output_model_dir: Path to save expanded model
        new_tokens: List of new tokens to add
        
    Raises:
        FileNotFoundError: If model directory doesn't exist
        ValueError: If new_tokens is empty
    """
    if not new_tokens:
        raise ValueError("new_tokens list cannot be empty")
    
    if not os.path.exists(hf_model_dir):
        raise FileNotFoundError(f"Model directory does not exist: {hf_model_dir}")
    
    # Create output directory
    os.makedirs(output_model_dir, exist_ok=True)
    logger.info(f"Expanding vocabulary for pretraining")
    logger.info(f"  Input model: {hf_model_dir}")
    logger.info(f"  Output model: {output_model_dir}")
    logger.info(f"  New tokens: {len(new_tokens)}")
    
    # Step 1: Load original model components
    logger.info("Loading original model components...")
    config = AutoConfig.from_pretrained(hf_model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        hf_model_dir,
        torch_dtype=torch.float32,  # Use float32 for compatibility
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(hf_model_dir, trust_remote_code=True)
    
    original_vocab_size = len(tokenizer)
    logger.info(f"Original vocabulary size: {original_vocab_size}")
    
    # Step 2: Add new tokens
    logger.info(f"Adding {len(new_tokens)} new tokens...")
    num_added = tokenizer.add_tokens(new_tokens)
    logger.info(f"Successfully added {num_added} tokens")
    
    # Step 3: Calculate aligned vocabulary size
    new_vocab_size = len(tokenizer)
    target_vocab_size = _align_vocab_size(new_vocab_size, alignment=256)
    logger.info(f"New vocabulary size: {new_vocab_size}")
    logger.info(f"Target vocabulary size (aligned to 256): {target_vocab_size}")
    
    # Step 4: Resize model embeddings
    logger.info("Resizing model token embeddings...")
    model.resize_token_embeddings(target_vocab_size)
    
    # Step 5: Update configuration
    config.vocab_size = target_vocab_size
    logger.info(f"Updated config vocab_size to {target_vocab_size}")
    
    # Step 6: Save expanded components
    logger.info("Saving expanded model components...")
    tokenizer.save_pretrained(output_model_dir)
    model.save_pretrained(output_model_dir)
    config.save_pretrained(output_model_dir)
    logger.info("Model components saved successfully")
    
    # Step 7: Fix chat template
    logger.info("Fixing chat template...")
    _fix_chat_template(output_model_dir, hf_model_dir)
    
    # Step 8: Test expanded vocabulary
    logger.info("Testing expanded vocabulary...")
    _test_expanded_vocab(model, tokenizer, new_tokens)
    
    logger.info(f"✓ Vocabulary expansion completed! Final vocab size: {target_vocab_size}")


def generate_itemic_tokens(itemic_layer_n: int, vocab_size_per_layer: int) -> List[str]:
    """Generate itemic special tokens dynamically.
    
    IMPORTANT: Token order must strictly match gen_itemic_sp_tokens.py:
    1. All <s_a_{i}> tokens (i from 0 to vocab_size_per_layer-1)
    2. All <s_b_{i}> tokens (i from 0 to vocab_size_per_layer-1)
    3. All <s_c_{i}> tokens (i from 0 to vocab_size_per_layer-1)
    4. ... (for itemic_layer_n layers, in alphabetical order)
    5. <|sid_begin|>
    6. <|sid_end|>
    
    Args:
        itemic_layer_n: Number of itemic layers (determines s_a, s_b, s_c, ...)
        vocab_size_per_layer: Vocabulary size per layer (determines range of i)
        
    Returns:
        List of generated tokens in strict order
        
    Raises:
        ValueError: If itemic_layer_n or vocab_size_per_layer is invalid
    """
    if itemic_layer_n <= 0:
        raise ValueError(f"itemic_layer_n must be positive, got {itemic_layer_n}")
    if vocab_size_per_layer <= 0:
        raise ValueError(f"vocab_size_per_layer must be positive, got {vocab_size_per_layer}")
    
    # Generate layer names in alphabetical order: a, b, c, d, ...
    # This ensures the same order as gen_itemic_sp_tokens.py
    layer_names = [chr(ord('a') + i) for i in range(itemic_layer_n)]
    
    new_tokens = []
    
    # Generate tokens in strict order:
    # For each layer (a, b, c, ...), generate all tokens with i from 0 to vocab_size_per_layer-1
    # This matches the order: [*s_a_0..8191, *s_b_0..8191, *s_c_0..8191, ...]
    for layer_name in layer_names:
        for i in range(vocab_size_per_layer):
            new_tokens.append(f"<s_{layer_name}_{i}>")
    
    # Add special tokens at the end (must be in this exact order)
    new_tokens.append('<|sid_begin|>')
    new_tokens.append('<|sid_end|>')
    
    total_tokens = itemic_layer_n * vocab_size_per_layer + 2
    logger.info(f"Generated {total_tokens} itemic tokens in strict order:")
    logger.info(f"  Layers: {itemic_layer_n} ({', '.join([f's_{name}' for name in layer_names])})")
    logger.info(f"  Vocab size per layer: {vocab_size_per_layer}")
    logger.info(f"  Special tokens: <|sid_begin|>, <|sid_end|>")
    
    return new_tokens


def load_tokens_from_file(tokens_file: str) -> List[str]:
    """Load tokens from a text file (one token per line).
    
    Args:
        tokens_file: Path to text file containing tokens (one per line)
        
    Returns:
        List of tokens (empty lines are skipped)
        
    Raises:
        FileNotFoundError: If tokens file doesn't exist
    """
    if not os.path.exists(tokens_file):
        raise FileNotFoundError(f"Tokens file does not exist: {tokens_file}")
    
    new_tokens = []
    line_count = 0
    
    with open(tokens_file, "r", encoding="utf-8") as f:
        for line in f:
            line_count += 1
            token = line.strip()
            if token:  # Skip empty lines
                new_tokens.append(token)
    
    logger.info(f"Loaded {len(new_tokens)} tokens from {line_count} lines in {tokens_file}")
    return new_tokens


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Expand Qwen3 vocabulary for pretraining by adding new tokens. '
                    'Supports two modes: loading from file or generating itemic tokens dynamically.'
    )
    parser.add_argument(
        "--hf_model_dir",
        type=str,
        required=True,
        help="Path to original HuggingFace Qwen3 model directory"
    )
    parser.add_argument(
        "--output_model_dir",
        type=str,
        required=True,
        help="Path to save expanded model directory"
    )
    
    # Itemic token generation parameters
    parser.add_argument(
        "--itemic_layer_n",
        type=int,
        required=True,
        help="Number of itemic layers (e.g., 3 for s_a, s_b, s_c)"
    )
    parser.add_argument(
        "--vocab_size_per_layer",
        type=int,
        required=True,
        help="Vocabulary size per layer (e.g., 8192 for tokens from 0 to 8191)"
    )
    
    args = parser.parse_args()
    
    try:
        # Generate itemic tokens dynamically
        logger.info("Generating itemic tokens dynamically...")
        new_tokens = generate_itemic_tokens(
            itemic_layer_n=args.itemic_layer_n,
            vocab_size_per_layer=args.vocab_size_per_layer
        )
        
        if not new_tokens:
            logger.error("No tokens to add")
            sys.exit(1)
        
        # Expand vocabulary
        expand_qwen3_vocab_for_pretraining(
            hf_model_dir=args.hf_model_dir,
            output_model_dir=args.output_model_dir,
            new_tokens=new_tokens
        )
        
        logger.info("All operations completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Program execution failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
