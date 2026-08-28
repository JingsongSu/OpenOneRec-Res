#!/bin/bash

set -euo pipefail

# ============================================================
# Qwen3 vocab expansion for:
#   existing add_feature tokens + time tokens
#
# Goal:
#   1. Keep ALL existing add_feature token IDs unchanged:
#      SID / sid_begin / sid_end / ctype / MID / LS
#   2. Append time tokens strictly AFTER them:
#      <|time_0|> ... <|time_336|>
#   3. Keep model weight dtype consistent with the existing
#      add_feature model: FP32
#
# Important:
#   The existing expand_qwen3_vocab_add_feature.py generates:
#
#       SID -> time -> ctype -> MID -> LS
#
#   Therefore we intentionally call it with TIME_TOKEN_SIZE=0
#   first, then append time tokens in a second step.
#
#   CRITICAL:
#   The existing add_feature checkpoint stores actual weights
#   as FP32, even though config.json may contain:
#
#       "torch_dtype": "bfloat16"
#
#   Therefore Stage 2 MUST explicitly load with torch.float32.
#   DO NOT use torch_dtype="auto", otherwise the entire model
#   will be converted to BF16 when re-saved.
# ============================================================


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRETRAIN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"


# ============================================================
# Paths
# ============================================================

HF_MODEL_DIR=/home/jovyan/ceph-1/sujinsong/sujinsong/OpenOneRec-latent/pretrain/model_output/Qwen3-0.6B

# Existing add_feature model.
# This is the source whose token IDs and FP32 weights must remain unchanged.
ADD_FEATURE_MODEL_DIR=/home/jovyan/ceph-1/sujinsong/sujinsong/OpenOneRec-latent/pretrain/model_output/Qwen3-0.6B_itemic_add_feature

# Final model:
# existing add_feature vocab + time tokens appended at the end.
OUTPUT_MODEL_DIR=/home/jovyan/ceph-1/sujinsong/sujinsong/OpenOneRec-latent/pretrain/model_output/Qwen3-0.6B_itemic_add_feature_time


# ============================================================
# Existing add_feature vocabulary settings
# ============================================================

ITEMIC_LAYER_N=4
VOCAB_SIZE_PER_LAYER=1024
LAST_LAYER_VOCAB_SIZE=15000

CTYPE_TOKEN_SIZE=18
MID_VOCAB_SIZE_PER_LAYER=1024

# Keep exactly the same setting as existing add_feature.
LS_VOCAB_SIZE_PER_LAYER=0


# ============================================================
# Time vocabulary settings
# ============================================================

# Stage 1 MUST NOT add time tokens.
#
# Otherwise the existing converter would create:
#
#   SID -> time -> ctype -> MID
#
# which would shift ctype/MID token IDs.
BASE_TIME_TOKEN_SIZE=0

# time IDs:
#   0, 1, ..., 336
#
# total = 337 tokens.
TIME_TOKEN_SIZE=337


# ============================================================
# Stage 1
#
# Make sure the original add_feature expanded model exists.
#
# Result:
#
#   original Qwen vocab
#       +
#   SID
#       +
#   ctype
#       +
#   MID
#
# No time tokens are added here.
# ============================================================

if [[ ! -f "${ADD_FEATURE_MODEL_DIR}/config.json" || \
      ! -f "${ADD_FEATURE_MODEL_DIR}/tokenizer_config.json" ]]; then

    echo "============================================================"
    echo "[1/2] Existing add_feature model not found."
    echo "      Building add_feature model with TIME_TOKEN_SIZE=0 ..."
    echo "============================================================"

    python3 "${PRETRAIN_DIR}/tools/model_converter/expand_qwen3_vocab_add_feature.py" \
        --hf_model_dir "${HF_MODEL_DIR}" \
        --output_model_dir "${ADD_FEATURE_MODEL_DIR}" \
        --itemic_layer_n "${ITEMIC_LAYER_N}" \
        --vocab_size_per_layer "${VOCAB_SIZE_PER_LAYER}" \
        --last_layer_vocab_size "${LAST_LAYER_VOCAB_SIZE}" \
        --time_token_size "${BASE_TIME_TOKEN_SIZE}" \
        --ctype_token_size "${CTYPE_TOKEN_SIZE}" \
        --mid_vocab_size_per_layer "${MID_VOCAB_SIZE_PER_LAYER}" \
        --ls_vocab_size_per_layer "${LS_VOCAB_SIZE_PER_LAYER}"

else

    echo "============================================================"
    echo "[1/2] Reusing existing add_feature model:"
    echo "      ${ADD_FEATURE_MODEL_DIR}"
    echo "============================================================"

fi


# ============================================================
# Stage 2
#
# Append time tokens AFTER every existing add_feature token.
#
# Existing tokenizer IDs must not change.
#
# IMPORTANT:
#   Load model explicitly as FP32.
#
# DO NOT use:
#
#   torch_dtype="auto"
#
# because config.json may say bfloat16, which would convert
# all existing FP32 weights to BF16.
# ============================================================

echo ""
echo "============================================================"
echo "[2/2] Appending ${TIME_TOKEN_SIZE} time tokens at vocab end..."
echo "============================================================"

export ADD_FEATURE_MODEL_DIR
export OUTPUT_MODEL_DIR
export TIME_TOKEN_SIZE


python3 - <<'PY'
import glob
import json
import math
import os
import shutil
from collections import Counter

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# Helpers
# ============================================================

def align_vocab_size(vocab_size: int, alignment: int = 256) -> int:
    """Round vocab size upward to alignment boundary."""
    return ((vocab_size + alignment - 1) // alignment) * alignment


def inspect_safetensors(model_dir):
    """
    Read safetensors metadata without loading all tensors.

    Returns:
        shapes
        dtypes
        total_numel
        total_disk_bytes
    """
    tensor_files = sorted(
        glob.glob(os.path.join(model_dir, "*.safetensors"))
    )

    if not tensor_files:
        raise RuntimeError(
            f"No safetensors files found in: {model_dir}"
        )

    shapes = {}
    dtypes = {}
    total_numel = 0
    total_disk_bytes = 0

    for fn in tensor_files:
        total_disk_bytes += os.path.getsize(fn)

        with safe_open(fn, framework="pt", device="cpu") as f:
            for key in f.keys():
                sl = f.get_slice(key)

                shape = tuple(sl.get_shape())
                dtype = str(sl.get_dtype())
                numel = math.prod(shape)

                shapes[key] = shape
                dtypes[key] = dtype
                total_numel += numel

    return (
        shapes,
        dtypes,
        total_numel,
        total_disk_bytes,
    )


# ============================================================
# Configuration
# ============================================================

src_dir = os.environ["ADD_FEATURE_MODEL_DIR"]
out_dir = os.environ["OUTPUT_MODEL_DIR"]
time_token_size = int(os.environ["TIME_TOKEN_SIZE"])


if time_token_size != 337:
    raise ValueError(
        "TIME_TOKEN_SIZE must be 337 for "
        "<|time_0|> ... <|time_336|>, "
        f"got {time_token_size}"
    )


if os.path.realpath(src_dir) == os.path.realpath(out_dir):
    raise RuntimeError(
        "ADD_FEATURE_MODEL_DIR and OUTPUT_MODEL_DIR must "
        "not be the same directory."
    )


# ============================================================
# Inspect source model BEFORE doing anything.
#
# The current add_feature model is expected to be FP32.
# ============================================================

print("")
print("=" * 80)
print("SOURCE MODEL CHECK")
print("=" * 80)

(
    src_shapes,
    src_dtypes,
    src_numel,
    src_disk_size,
) = inspect_safetensors(src_dir)

src_dtype_counter = Counter(src_dtypes.values())

print(f"Source tensor count : {len(src_shapes)}")
print(f"Source total numel  : {src_numel:,}")
print(
    f"Source disk size    : "
    f"{src_disk_size / 1024**3:.4f} GiB"
)
print("Source dtype distribution:")

for dtype, count in src_dtype_counter.items():
    tensor_count = sum(
        1 for x in src_dtypes.values() if x == dtype
    )
    print(
        f"  {dtype:10s}: "
        f"{tensor_count} tensors"
    )


# For this specific add_feature model we require FP32.
non_fp32 = [
    key
    for key, dtype in src_dtypes.items()
    if dtype != "F32"
]

if non_fp32:
    raise RuntimeError(
        "Source add_feature checkpoint is not fully FP32. "
        "This script is designed to preserve the existing "
        "FP32 add_feature checkpoint exactly.\n"
        f"First non-FP32 tensor: "
        f"{non_fp32[0]} -> {src_dtypes[non_fp32[0]]}"
    )

print("Source checkpoint dtype: FP32 OK")


# Save original config torch_dtype so that we do not
# accidentally change unrelated config semantics.
src_config_path = os.path.join(src_dir, "config.json")

with open(src_config_path, "r") as f:
    src_config_json = json.load(f)

src_config_torch_dtype = src_config_json.get(
    "torch_dtype",
    None,
)

print(
    "Source config torch_dtype:",
    src_config_torch_dtype,
)


# ============================================================
# Load tokenizer
# ============================================================

print("")
print("=" * 80)
print("TOKENIZER EXPANSION")
print("=" * 80)

print(f"Loading tokenizer from: {src_dir}")

tokenizer = AutoTokenizer.from_pretrained(
    src_dir,
    trust_remote_code=True,
)

old_vocab = tokenizer.get_vocab()

old_tokenizer_size = len(tokenizer)
old_max_token_id = max(old_vocab.values())


# ============================================================
# Representative existing tokens whose IDs MUST remain stable.
# ============================================================

anchor_tokens = [
    "<s_a_0>",
    "<s_b_0>",
    "<s_c_0>",
    "<s_d_0>",
    "<|sid_begin|>",
    "<|sid_end|>",
    "<|ctype_0|>",
    "<|ctype_17|>",
    "<mid_a_0>",
    "<mid_b_0>",
    "<mid_c_0>",
]


missing_anchors = [
    tok
    for tok in anchor_tokens
    if tok not in old_vocab
]

if missing_anchors:
    raise RuntimeError(
        "Source model does not look like the expected "
        "add_feature model.\n"
        f"Missing tokens: {missing_anchors}"
    )


old_anchor_ids = {
    tok: old_vocab[tok]
    for tok in anchor_tokens
}


# ============================================================
# Construct time tokens.
# ============================================================

time_tokens = [
    f"<|time_{i}|>"
    for i in range(time_token_size)
]


# Source model must contain NO real time tokens.
already_existing = [
    tok
    for tok in time_tokens
    if tok in old_vocab
]

if already_existing:
    raise RuntimeError(
        "Source add_feature tokenizer already contains "
        "time tokens.\n"
        "To guarantee that old token IDs remain unchanged, "
        "ADD_FEATURE_MODEL_DIR must be the TIME_TOKEN_SIZE=0 "
        "add_feature model.\n"
        f"First existing time token: {already_existing[0]}"
    )


print(
    f"Tokenizer size before time tokens: "
    f"{old_tokenizer_size}"
)

print(
    f"Max token ID before time tokens:   "
    f"{old_max_token_id}"
)


# ============================================================
# Append time tokens.
# ============================================================

num_added = tokenizer.add_tokens(time_tokens)

if num_added != time_token_size:
    raise RuntimeError(
        f"Expected to add {time_token_size} time tokens, "
        f"but actually added {num_added}"
    )


new_vocab = tokenizer.get_vocab()

time_ids = [
    new_vocab[tok]
    for tok in time_tokens
]


# ============================================================
# Time token IDs must start immediately after old max token ID.
# ============================================================

expected_first_time_id = old_max_token_id + 1

if time_ids[0] != expected_first_time_id:
    raise RuntimeError(
        "<|time_0|> was not appended immediately after "
        "the existing vocabulary.\n"
        f"Expected ID: {expected_first_time_id}\n"
        f"Actual ID:   {time_ids[0]}"
    )


expected_time_ids = list(
    range(
        expected_first_time_id,
        expected_first_time_id + time_token_size,
    )
)


if time_ids != expected_time_ids:
    raise RuntimeError(
        "Time token IDs are not contiguous."
    )


# ============================================================
# Verify existing token IDs did not move.
# ============================================================

for tok, old_id in old_anchor_ids.items():
    new_id = new_vocab[tok]

    if new_id != old_id:
        raise RuntimeError(
            f"Existing token ID changed: "
            f"{tok}: {old_id} -> {new_id}"
        )


# ============================================================
# Verify every time token can be encoded as ONE token.
# ============================================================

bad_time_tokens = []

for i, tok in enumerate(time_tokens):
    encoded = tokenizer.encode(
        tok,
        add_special_tokens=False,
    )

    expected_id = expected_time_ids[i]

    if encoded != [expected_id]:
        bad_time_tokens.append(
            (tok, encoded, expected_id)
        )


if bad_time_tokens:
    tok, encoded, expected_id = bad_time_tokens[0]

    raise RuntimeError(
        "Time token is not encoded as exactly one token.\n"
        f"Token:       {tok}\n"
        f"Encoded:     {encoded}\n"
        f"Expected ID: {expected_id}"
    )


print(
    f"Tokenizer size after time tokens: "
    f"{len(tokenizer)}"
)

print(
    f"Time token ID range: "
    f"{time_ids[0]} .. {time_ids[-1]}"
)

print(
    "All existing SID / ctype / MID token IDs: unchanged"
)

print(
    "All 337 time tokens: single-token encoding OK"
)


# ============================================================
# Load model
#
# CRITICAL FIX:
#
# DO NOT USE:
#
#     torch_dtype="auto"
#
# The source config contains torch_dtype=bfloat16 even though
# the actual checkpoint weights are FP32.
#
# "auto" therefore converts every source tensor to BF16.
#
# We explicitly load FP32 so that all existing parameters keep
# the same dtype as the source add_feature checkpoint.
# ============================================================

print("")
print("=" * 80)
print("MODEL LOAD")
print("=" * 80)

print(f"Loading FP32 add_feature model from: {src_dir}")

model = AutoModelForCausalLM.from_pretrained(
    src_dir,

    # ========================================================
    # CRITICAL:
    # keep identical to original add_feature weight dtype.
    # ========================================================
    torch_dtype=torch.float32,

    trust_remote_code=True,
    low_cpu_mem_usage=True,
)


# Verify loaded model really is FP32.
loaded_non_fp32 = []

for name, param in model.named_parameters():
    if param.dtype != torch.float32:
        loaded_non_fp32.append(
            (name, str(param.dtype))
        )

if loaded_non_fp32:
    raise RuntimeError(
        "Model was not loaded fully as FP32.\n"
        f"First unexpected tensor: "
        f"{loaded_non_fp32[0]}"
    )

print("Loaded model dtype: FP32 OK")


# ============================================================
# Resize embedding.
# ============================================================

old_embedding_size = (
    model.get_input_embeddings().num_embeddings
)

new_tokenizer_size = len(tokenizer)


# Do not shrink an already padded embedding matrix.
#
# Example:
#
#   tokenizer old len  = 172833
#   embedding old rows = 173056
#
# After +337 tokens:
#
#   tokenizer new len  = 173170
#
# aligned:
#
#   173312
#
# Therefore:
#
#   embedding:
#   173056 -> 173312
#
# physical increase = 256 rows
#
# even though logical tokens increase by 337.
target_vocab_size = max(
    old_embedding_size,
    align_vocab_size(
        new_tokenizer_size,
        alignment=256,
    ),
)


print(
    f"Tokenizer size after time tokens: {new_tokenizer_size}"
)

print(
    f"Old embedding vocab size:         {old_embedding_size}"
)

print(
    f"Final embedding vocab size:       {target_vocab_size}"
)


if target_vocab_size != old_embedding_size:

    print(
        f"Resizing embeddings: "
        f"{old_embedding_size} -> {target_vocab_size}"
    )

    model.resize_token_embeddings(
        target_vocab_size
    )

else:

    print(
        "No embedding resize needed; "
        "existing aligned rows are sufficient."
    )


model.config.vocab_size = target_vocab_size


# ============================================================
# After resize, all parameters must STILL be FP32.
# ============================================================

post_resize_non_fp32 = []

for name, param in model.named_parameters():
    if param.dtype != torch.float32:
        post_resize_non_fp32.append(
            (name, str(param.dtype))
        )


if post_resize_non_fp32:
    raise RuntimeError(
        "Some model tensors changed dtype after resize.\n"
        f"First unexpected tensor: "
        f"{post_resize_non_fp32[0]}"
    )


print("Model dtype after embedding resize: FP32 OK")


# ============================================================
# Remove stale output from previous BF16 run.
# ============================================================

if os.path.exists(out_dir):
    print("")
    print(
        f"Removing existing output directory: "
        f"{out_dir}"
    )
    shutil.rmtree(out_dir)


os.makedirs(
    out_dir,
    exist_ok=True,
)


# ============================================================
# Save tokenizer and FP32 model.
# ============================================================

print("")
print("=" * 80)
print("SAVING MODEL")
print("=" * 80)

tokenizer.save_pretrained(out_dir)

model.save_pretrained(
    out_dir,
    safe_serialization=True,
)


# ============================================================
# Preserve source config torch_dtype semantics.
#
# The source add_feature config currently says bfloat16 even
# though its on-disk checkpoint is FP32.
#
# We only want to change vocab_size. Everything unrelated to
# time expansion should remain consistent with source.
# ============================================================

saved_config_path = os.path.join(
    out_dir,
    "config.json",
)

with open(saved_config_path, "r") as f:
    saved_config_json = json.load(f)


saved_config_json["vocab_size"] = target_vocab_size

if src_config_torch_dtype is not None:
    saved_config_json["torch_dtype"] = (
        src_config_torch_dtype
    )


with open(saved_config_path, "w") as f:
    json.dump(
        saved_config_json,
        f,
        ensure_ascii=False,
        indent=2,
    )
    f.write("\n")


# Release model before disk verification.
del model


# ============================================================
# Reload tokenizer from disk.
# ============================================================

print("")
print("=" * 80)
print("TOKENIZER DISK VERIFICATION")
print("=" * 80)

saved_tokenizer = AutoTokenizer.from_pretrained(
    out_dir,
    trust_remote_code=True,
)

saved_vocab = saved_tokenizer.get_vocab()


# Existing token IDs must still match.
for tok, old_id in old_anchor_ids.items():

    saved_id = saved_vocab[tok]

    if saved_id != old_id:
        raise RuntimeError(
            "Saved tokenizer changed existing token ID:\n"
            f"{tok}: {old_id} -> {saved_id}"
        )


# Time IDs must match.
for i, expected_id in enumerate(
    expected_time_ids
):

    tok = f"<|time_{i}|>"

    actual_id = saved_vocab[tok]

    if actual_id != expected_id:
        raise RuntimeError(
            f"Saved tokenizer has wrong ID for {tok}:\n"
            f"Expected: {expected_id}\n"
            f"Actual:   {actual_id}"
        )

    encoded = saved_tokenizer.encode(
        tok,
        add_special_tokens=False,
    )

    if encoded != [expected_id]:
        raise RuntimeError(
            f"Saved tokenizer does not encode "
            f"{tok} as one token:\n"
            f"{encoded}"
        )


print("Saved tokenizer IDs: OK")
print("Saved time-token encoding: OK")


# ============================================================
# Safetensors verification
#
# This catches exactly the previous problem:
#
#   OLD = FP32
#   NEW = BF16
#
# New output MUST have same dtype as source for every existing
# tensor.
# ============================================================

print("")
print("=" * 80)
print("MODEL DISK VERIFICATION")
print("=" * 80)

(
    out_shapes,
    out_dtypes,
    out_numel,
    out_disk_size,
) = inspect_safetensors(out_dir)


print(f"Output tensor count : {len(out_shapes)}")
print(f"Output total numel  : {out_numel:,}")
print(
    f"Output disk size    : "
    f"{out_disk_size / 1024**3:.4f} GiB"
)


out_dtype_counter = Counter(
    out_dtypes.values()
)

print("Output dtype distribution:")

for dtype in sorted(out_dtype_counter):

    tensor_count = sum(
        1
        for x in out_dtypes.values()
        if x == dtype
    )

    param_count = sum(
        math.prod(out_shapes[k])
        for k in out_shapes
        if out_dtypes[k] == dtype
    )

    print(
        f"  {dtype:10s}: "
        f"{tensor_count} tensors, "
        f"{param_count:,} params"
    )


# ============================================================
# Key sets must be identical.
# ============================================================

src_keys = set(src_shapes)
out_keys = set(out_shapes)

missing_in_output = src_keys - out_keys
extra_in_output = out_keys - src_keys


if missing_in_output:
    raise RuntimeError(
        "Output model is missing tensors.\n"
        f"First missing key: "
        f"{sorted(missing_in_output)[0]}"
    )


if extra_in_output:
    raise RuntimeError(
        "Output model has unexpected extra tensors.\n"
        f"First extra key: "
        f"{sorted(extra_in_output)[0]}"
    )


print("Tensor key set: unchanged")


# ============================================================
# Dtype must match source for EVERY tensor.
# ============================================================

dtype_differences = []

for key in sorted(src_keys):

    if src_dtypes[key] != out_dtypes[key]:
        dtype_differences.append(
            (
                key,
                src_dtypes[key],
                out_dtypes[key],
            )
        )


if dtype_differences:
    key, old_dtype, new_dtype = (
        dtype_differences[0]
    )

    raise RuntimeError(
        "Output tensor dtype differs from source.\n"
        f"Tensor: {key}\n"
        f"Source dtype: {old_dtype}\n"
        f"Output dtype: {new_dtype}"
    )


print("All tensor dtypes: unchanged (FP32)")


# ============================================================
# Shape differences are allowed ONLY for embeddings.
# ============================================================

allowed_shape_change_keys = {
    "model.embed_tokens.weight",
    "lm_head.weight",
}

shape_differences = []

for key in sorted(src_keys):

    if src_shapes[key] != out_shapes[key]:

        shape_differences.append(
            (
                key,
                src_shapes[key],
                out_shapes[key],
            )
        )

        if key not in allowed_shape_change_keys:
            raise RuntimeError(
                "Unexpected tensor shape changed.\n"
                f"Tensor: {key}\n"
                f"Source: {src_shapes[key]}\n"
                f"Output: {out_shapes[key]}"
            )


print("")
print("Shape differences:")

for key, old_shape, new_shape in shape_differences:
    print(
        f"  {key}: "
        f"{old_shape} -> {new_shape}"
    )


if "model.embed_tokens.weight" not in out_shapes:
    raise RuntimeError(
        "model.embed_tokens.weight missing "
        "from saved model."
    )


actual_embedding_rows = (
    out_shapes[
        "model.embed_tokens.weight"
    ][0]
)

if actual_embedding_rows != target_vocab_size:
    raise RuntimeError(
        "Saved embedding row count is wrong.\n"
        f"Expected: {target_vocab_size}\n"
        f"Actual:   {actual_embedding_rows}"
    )


# ============================================================
# Check config from disk.
# ============================================================

with open(saved_config_path, "r") as f:
    final_config = json.load(f)


if final_config.get("vocab_size") != target_vocab_size:
    raise RuntimeError(
        "Saved config vocab_size is incorrect.\n"
        f"Expected: {target_vocab_size}\n"
        f"Actual: "
        f"{final_config.get('vocab_size')}"
    )


print(
    "Final config vocab_size:",
    final_config.get("vocab_size"),
)

print(
    "Final config torch_dtype:",
    final_config.get("torch_dtype"),
)


# ============================================================
# Final summary
# ============================================================

print("")
print("=" * 80)
print("VOCABULARY EXPANSION COMPLETED SUCCESSFULLY")
print("=" * 80)

print(
    f"Source add_feature model:\n"
    f"  {src_dir}"
)

print(
    f"\nOutput model:\n"
    f"  {out_dir}"
)

print("")
print(
    f"Old tokenizer size:       "
    f"{old_tokenizer_size}"
)

print(
    f"New tokenizer size:       "
    f"{new_tokenizer_size}"
)

print(
    f"Added time tokens:        "
    f"{time_token_size}"
)

print(
    f"<|time_0|> ID:            "
    f"{expected_time_ids[0]}"
)

print(
    f"<|time_336|> ID:          "
    f"{expected_time_ids[-1]}"
)

print(
    f"Old embedding rows:       "
    f"{old_embedding_size}"
)

print(
    f"New embedding rows:       "
    f"{target_vocab_size}"
)

print(
    f"Source parameters:        "
    f"{src_numel:,}"
)

print(
    f"Output parameters:        "
    f"{out_numel:,}"
)

print(
    f"Source model size:        "
    f"{src_disk_size / 1024**3:.4f} GiB"
)

print(
    f"Output model size:        "
    f"{out_disk_size / 1024**3:.4f} GiB"
)

print("")
print("Checks passed:")
print("  [OK] existing token IDs unchanged")
print("  [OK] 337 time tokens appended at end")
print("  [OK] time token IDs contiguous")
print("  [OK] every time token is one token")
print("  [OK] tensor key set unchanged")
print("  [OK] all existing weight dtypes unchanged")
print("  [OK] model weights are FP32")
print("  [OK] only embedding shape expanded")
print("  [OK] embedding covers all time token IDs")

print("=" * 80)

PY
