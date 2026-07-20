"""
PT format model checkpoint loading tool

Supports loading PyTorch model checkpoints in non-safetensor format
"""

import torch
import hashlib
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from difflib import SequenceMatcher
from benchmark.console import console


def match_checkpoint_keys_to_model(
    checkpoint_keys: List[str],
    model_keys: List[str],
    similarity_threshold: float = 0.8
) -> Dict[str, str]:
    """
    Intelligently match checkpoint key names to model key names
    
    Args:
        checkpoint_keys: List of key names in checkpoint
        model_keys: List of key names in model
        similarity_threshold: Similarity threshold
    
    Returns:
        Mapping dictionary {checkpoint_key: model_key}
    """
    mapping = {}
    
    for ckpt_key in checkpoint_keys:
        # Try exact match first
        if ckpt_key in model_keys:
            mapping[ckpt_key] = ckpt_key
            continue
        
        # Try matching by removing "model." prefix
        if ckpt_key.startswith("model."):
            clean_key = ckpt_key[6:]  # Remove "model."
            if clean_key in model_keys:
                mapping[ckpt_key] = clean_key
                continue
        
        # Try matching by adding "model." prefix
        prefixed_key = f"model.{ckpt_key}"
        if prefixed_key in model_keys:
            mapping[ckpt_key] = prefixed_key
            continue
        
        # Use similarity matching
        best_match = None
        best_score = 0.0
        
        for model_key in model_keys:
            score = SequenceMatcher(None, ckpt_key, model_key).ratio()
            if score > best_score and score >= similarity_threshold:
                best_score = score
                best_match = model_key
        
        if best_match:
            mapping[ckpt_key] = best_match
            console.print(f"Similarity match: {ckpt_key} -> {best_match} (score: {best_score:.2f})")
    
    return mapping


def check_embedding_weight_sharing(
    state_dict: Dict[str, torch.Tensor],
    verbose: bool = True
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Check if embed_tokens and lm_head weights are shared
    
    Args:
        state_dict: Model state dictionary
        verbose: Whether to print detailed information
    
    Returns:
        (is_shared, embed_key, lm_head_key)
    """
    # Find embed_tokens and lm_head keys
    embed_key = None
    lm_head_key = None
    
    for key in state_dict.keys():
        if "embed_tokens.weight" in key:
            embed_key = key
        elif "lm_head.weight" in key:
            lm_head_key = key
    
    if not embed_key or not lm_head_key:
        if verbose:
            console.print(f"Complete weight pair not found: embed_tokens={embed_key}, lm_head={lm_head_key}")
        return False, embed_key, lm_head_key
    
    embed_tensor = state_dict[embed_key]
    lm_head_tensor = state_dict[lm_head_key]
    
    if verbose:
        console.print(f"embed_tokens.weight shape: {embed_tensor.shape}")
        console.print(f"lm_head.weight shape: {lm_head_tensor.shape}")
    
    # Check if completely identical
    is_shared = torch.equal(embed_tensor, lm_head_tensor)
    
    if verbose:
        if is_shared:
            console.print("✓ embed_tokens and lm_head weights are identical (shared weights)")
        else:
            console.print("✗ embed_tokens and lm_head weights are different")
            # Calculate difference statistics
            diff = (embed_tensor != lm_head_tensor).sum().item()
            total = embed_tensor.numel()
            console.print(f"  Different elements: {diff}/{total} ({diff/total*100:.2f}%)")
    
    return is_shared, embed_key, lm_head_key


def handle_weight_tying(
    state_dict: Dict[str, torch.Tensor],
    model_keys: List[str],
    new_state_dict: Dict[str, str]
) -> Dict[str, torch.Tensor]:
    """
    Handle weight tying situations
    
    In some models, embed_tokens and lm_head weights are tied
    
    Args:
        state_dict: Original state dictionary
        model_keys: List of model key names
        new_state_dict: Already mapped new state dictionary
    
    Returns:
        Updated state dictionary
    """
    # Scenario 1: checkpoint has embed_tokens but no lm_head
    if any("embed_tokens.weight" in k for k in state_dict.keys()):
        embed_key = next((k for k in state_dict.keys() if "embed_tokens.weight" in k), None)
        
        # Check if lm_head is missing in new_state_dict
        has_lm_head = any("lm_head.weight" in k for k in new_state_dict.keys())
        
        if not has_lm_head and embed_key:
            # Try to find lm_head key in model
            lm_head_candidates = ["lm_head.weight", "model.lm_head.weight"]
            for candidate in lm_head_candidates:
                if candidate in model_keys:
                    new_state_dict[candidate] = state_dict[embed_key]
                    console.print(f"✓ Weight tying: using {embed_key} to initialize {candidate}")
                    break
    
    # Scenario 2: checkpoint has lm_head but no embed_tokens
    if any("lm_head.weight" in k for k in state_dict.keys()):
        lm_head_key = next((k for k in state_dict.keys() if "lm_head.weight" in k), None)
        
        # Check if embed_tokens is missing in new_state_dict
        has_embed = any("embed_tokens.weight" in k for k in new_state_dict.keys())
        
        if not has_embed and lm_head_key:
            # Try to find embed_tokens key in model
            embed_candidates = ["embed_tokens.weight", "model.embed_tokens.weight"]
            for candidate in embed_candidates:
                if candidate in model_keys:
                    new_state_dict[candidate] = state_dict[lm_head_key]
                    console.print(f"✓ Weight tying: using {lm_head_key} to initialize {candidate}")
                    break
    
    return new_state_dict


def load_weights_from_pt(
    model: torch.nn.Module,
    checkpoint_path: str,
    device: str = "cpu",
    strict: bool = False,
    check_weight_sharing: bool = True,
    handle_weight_tying_flag: bool = True
) -> Tuple[List[str], List[str]]:
    """
    Load PT format checkpoint into model
    
    Args:
        model: Target model
        checkpoint_path: Checkpoint file path
        device: Loading device
        strict: Whether to load strictly (requires all keys to match)
        check_weight_sharing: Whether to check weight sharing
        handle_weight_tying_flag: Whether to handle weight tying
    
    Returns:
        (missing_keys, unexpected_keys) Missing keys and unexpected keys
    """
    console.print(f"Loading checkpoint: {checkpoint_path}")
    
    # 1. Load checkpoint
    try:
        state_dict = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        console.print(f"Failed to load checkpoint: {e}")
        raise
    
    # 2. Extract model state dictionary
    if 'model_state_dict' in state_dict:
        console.print("Detected 'model_state_dict' key, extracting nested state dictionary")
        state_dict = state_dict['model_state_dict']
    elif 'state_dict' in state_dict:
        console.print("Detected 'state_dict' key, extracting nested state dictionary")
        state_dict = state_dict['state_dict']

    checkpoint_keys = list(state_dict.keys())
    model_keys = list(model.state_dict().keys())

    console.print(f"Checkpoint key count: {len(checkpoint_keys)}")
    console.print(f"Model key count: {len(model_keys)}")

    if check_weight_sharing:
        check_embedding_weight_sharing(state_dict, verbose=True)

    console.print("Starting to match checkpoint key names to model key names...")
    key_mapping = match_checkpoint_keys_to_model(checkpoint_keys, model_keys)

    matched_count = len(key_mapping)
    console.print(f"Successfully matched: {matched_count}/{len(checkpoint_keys)} keys")

    new_state_dict = {}
    skipped_keys = []

    for ckpt_key in checkpoint_keys:
        target_key = key_mapping.get(ckpt_key)
        if target_key is None:
            skipped_keys.append(ckpt_key)
            continue
        new_state_dict[target_key] = state_dict[ckpt_key]

    if skipped_keys:
        console.print(f"Skipped {len(skipped_keys)} unmatched keys")
        if len(skipped_keys) <= 10:
            console.print(f"Skipped keys: {skipped_keys}")

    if handle_weight_tying_flag:
        new_state_dict = handle_weight_tying(state_dict, model_keys, new_state_dict)

    console.print("Loading state dictionary into model...")
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=strict)

    if missing_keys:
        console.print(f"Missing keys ({len(missing_keys)}): {missing_keys[:10]}{'...' if len(missing_keys) > 10 else ''}")
    else:
        console.print("✓ No missing keys")
    
    if unexpected_keys:
        console.print(f"Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:10]}{'...' if len(unexpected_keys) > 10 else ''}")
    else:
        console.print("✓ No unexpected keys")
    
    console.print(f"✓ Checkpoint loading completed")
    
    return missing_keys, unexpected_keys


def build_model_from_pt(
    config_path: str,
    checkpoint_path: str,
    device: str = "cuda",
    torch_dtype: Optional[torch.dtype] = None,
    trust_remote_code: bool = True
) -> torch.nn.Module:
    """
    Create model from config and load PT checkpoint
    
    This is the unified function used by both HfTransformersGenerator and RayHfTransformersGenerator.
    
    Args:
        config_path: Model configuration path
        checkpoint_path: PT checkpoint path
        device: Target device
        torch_dtype: Data type
        trust_remote_code: Whether to trust remote code
    
    Returns:
        Model with checkpoint loaded
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(
        config_path,
        trust_remote_code=trust_remote_code
    )

    model = AutoModelForCausalLM.from_config(
        config,
        trust_remote_code=trust_remote_code
    )

    if torch_dtype is not None:
        model = model.to(torch_dtype)
    if device != 'cpu':
        model = model.to(device)

    target_load_device = device if device != 'cpu' else 'cpu'
    load_weights_from_pt(
        model=model,
        checkpoint_path=checkpoint_path,
        device=target_load_device,
        strict=False,
        check_weight_sharing=True,
        handle_weight_tying_flag=True
    )
    
    return model


def build_model_from_hf(
    model_name_or_path: str,
    device: str = "cuda",
    torch_dtype: Optional[torch.dtype] = None,
    trust_remote_code: bool = True,
    use_device_map: bool = True
) -> torch.nn.Module:
    """
    Load pretrained model from HuggingFace
    
    This is the unified function used by both HfTransformersGenerator and RayHfTransformersGenerator.
    
    Args:
        model_name_or_path: Model name or path
        device: Target device
        torch_dtype: Data type
        trust_remote_code: Whether to trust remote code
        use_device_map: Whether to use device_map="auto" for multi-GPU
    
    Returns:
        Loaded model
    """
    from transformers import AutoModelForCausalLM

    should_use_device_map = use_device_map and device != "cpu" and "cuda" in device

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        device_map="auto" if should_use_device_map else None,
        trust_remote_code=trust_remote_code
    )

    if not should_use_device_map:
        model = model.to(device)

    return model


def export_pt_to_safetensor(
    config_path: str,
    checkpoint_path: str,
    output_dir: Optional[str] = None,
    trust_remote_code: bool = True,
    use_cache: bool = True
) -> str:
    """
    Convert PT checkpoint to HuggingFace format for vLLM compatibility

    Args:
        config_path: Model configuration path (HuggingFace model path or local config)
        checkpoint_path: PT checkpoint path
        output_dir: Output directory for converted model (optional, will use /tmp if not specified)
        trust_remote_code: Whether to trust remote code
        use_cache: Whether to use cached conversion (skip if already converted)

    Returns:
        Path to converted HuggingFace format model
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    hash_input = f"{config_path}_{checkpoint_path}".encode('utf-8')
    hash_suffix = hashlib.md5(hash_input).hexdigest()[:16]

    if output_dir is None:
        output_dir = f"/tmp/hf_checkpoint_{hash_suffix}"

    temp_model_path = Path(output_dir) / "converted_model"

    if use_cache and temp_model_path.exists():
        has_config = (temp_model_path / "config.json").exists()
        has_weights = (
            (temp_model_path / "model.safetensors").exists() or
            (temp_model_path / "pytorch_model.bin").exists() or
            any(temp_model_path.glob("*.safetensors")) or
            any(temp_model_path.glob("pytorch_model*.bin"))
        )

        if has_config and has_weights:
            console.print(
                f"✓ Found converted model, skipping conversion",
            )
            console.print(
                f"  Converted model path: {temp_model_path}",
            )
            return str(temp_model_path)

    # Create output directory
    temp_model_path.mkdir(parents=True, exist_ok=True)
    console.print(f"  Output directory: {temp_model_path}")

    try:
        # 1. Load configuration
        console.print("  [1/4] Loading model configuration...")
        config = AutoConfig.from_pretrained(
            config_path,
            trust_remote_code=trust_remote_code
        )

        # 2. Create model from config
        console.print("  [2/4] Initializing model...")
        model = AutoModelForCausalLM.from_config(
            config,
            trust_remote_code=trust_remote_code
        )

        # 3. Load checkpoint
        console.print("  [3/4] Loading PT checkpoint...")
        load_weights_from_pt(
            model=model,
            checkpoint_path=checkpoint_path,
            device='cpu',
            strict=False,
            check_weight_sharing=True,
            handle_weight_tying_flag=True
        )

        # 4. Save as HuggingFace format
        console.print("  [4/4] Saving as HuggingFace format...")
        model.save_pretrained(temp_model_path, safe_serialization=True)

        # Save tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            config_path,
            trust_remote_code=trust_remote_code
        )
        tokenizer.save_pretrained(temp_model_path)

        console.print(f"✓ Model conversion completed: {temp_model_path}")

        return str(temp_model_path)

    except Exception as e:
        console.print(f"✗ Conversion failed: {e}")
        # Clean up on failure
        import shutil
        if temp_model_path.exists():
            shutil.rmtree(temp_model_path)
        raise