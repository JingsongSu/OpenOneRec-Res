#!/usr/bin/env bash
set -euo pipefail
set -x

# Usage:
#   bash scripts/convert_residual_checkpoint_to_hf.sh \
#     <source_hf_model_dir> <training_output_dir> <step> [residual_config]
#
# This wrapper intentionally verifies chat-template preservation.
# Stage1/Stage2 pretraining may not call apply_chat_template(), so a missing
# template can otherwise stay hidden until SFT.  If the source tokenizer has
# a default chat template, the converted model must preserve it.

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: $0 <source_hf_model_dir> <training_output_dir> <step> [residual_config]" >&2
  exit 2
fi

SOURCE_HF_MODEL_DIR=$(readlink -f "$1")
MODEL_HOME=$(readlink -f "$2")
STEP=$3
RESIDUAL_CONFIG=${4:-${MODEL_HOME}/residual_sid_config.json}

PRETRAIN_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CHECKPOINT_DIR=${MODEL_HOME}/step${STEP}/global_step${STEP}
OUTPUT_DIR=${CHECKPOINT_DIR}/converted

for required_path in \
  "${SOURCE_HF_MODEL_DIR}/config.json" \
  "${SOURCE_HF_MODEL_DIR}/tokenizer_config.json" \
  "${CHECKPOINT_DIR}" \
  "${RESIDUAL_CONFIG}" \
  "${PRETRAIN_DIR}/scripts/convert_checkpoint_to_hf.sh" \
  "${PRETRAIN_DIR}/tools/model_converter/patch_residual_sid_hf_config.py"
do
  [[ -e "${required_path}" ]] || {
    echo "ERROR: missing ${required_path}" >&2
    exit 1
  }
done

cd "${PRETRAIN_DIR}"

bash scripts/convert_checkpoint_to_hf.sh \
  "${SOURCE_HF_MODEL_DIR}" \
  "${MODEL_HOME}" \
  "${STEP}"

python tools/model_converter/patch_residual_sid_hf_config.py \
  --hf_model_dir "${OUTPUT_DIR}" \
  --residual_config "${RESIDUAL_CONFIG}"

python - "${SOURCE_HF_MODEL_DIR}" "${OUTPUT_DIR}" <<'PY'
import json
import sys
from pathlib import Path


source_dir = Path(sys.argv[1])
model_dir = Path(sys.argv[2])

config_path = model_dir / "config.json"
index_path = model_dir / "model.safetensors.index.json"
residual_config_path = model_dir / "residual_sid_config.json"
tokenizer_config_path = model_dir / "tokenizer_config.json"

for path in (
    config_path,
    index_path,
    residual_config_path,
    tokenizer_config_path,
):
    if not path.is_file():
        raise FileNotFoundError(path)


def embedded_default_template(directory: Path) -> bool:
    path = directory / "tokenizer_config.json"
    if not path.is_file():
        return False

    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False

    value = cfg.get("chat_template")

    if isinstance(value, str):
        return bool(value.strip())

    if isinstance(value, dict):
        default = value.get("default")
        return (
            isinstance(default, str)
            and bool(default.strip())
        )

    if isinstance(value, list):
        for item in value:
            if (
                isinstance(item, dict)
                and item.get("name") == "default"
                and isinstance(item.get("template"), str)
                and item["template"].strip()
            ):
                return True

    return False


def has_default_chat_template(directory: Path) -> bool:
    standalone = directory / "chat_template.jinja"
    if standalone.is_file() and standalone.stat().st_size > 0:
        return True
    return embedded_default_template(directory)


source_has_chat_template = has_default_chat_template(source_dir)
output_has_chat_template = has_default_chat_template(model_dir)

# Preserve the template whenever it existed on the source model.
if source_has_chat_template and not output_has_chat_template:
    raise RuntimeError(
        "Chat template was lost during checkpoint -> HF conversion. "
        f"source={source_dir}, output={model_dir}"
    )

config = json.loads(
    config_path.read_text(encoding="utf-8")
)

required_keys = [
    "residual_sid_enabled",
    "residual_sid_layer_starts",
    "residual_sid_layer_sizes",
    "residual_sid_begin_token_id",
    "residual_sid_end_token_id",
]

missing = [
    key
    for key in required_keys
    if key not in config
]

if missing:
    raise RuntimeError(
        f"Residual config keys are missing: {missing}"
    )

weight_map = json.loads(
    index_path.read_text(encoding="utf-8")
)["weight_map"]

residual_keys = sorted(
    key
    for key in weight_map
    if key.startswith("sid_residual_blocks.")
)

if not residual_keys:
    raise RuntimeError(
        "No sid_residual_blocks.* tensors were found "
        "in the converted model."
    )

print(
    json.dumps(
        {
            "converted_model": str(model_dir),
            "architectures": config.get("architectures"),
            "residual_sid_layer_starts": config[
                "residual_sid_layer_starts"
            ],
            "residual_sid_layer_sizes": config[
                "residual_sid_layer_sizes"
            ],
            "num_residual_weight_tensors": len(residual_keys),
            "first_residual_weight_key": residual_keys[0],
            "source_has_chat_template": source_has_chat_template,
            "output_has_chat_template": output_has_chat_template,
            "chat_template_file": str(
                model_dir / "chat_template.jinja"
            ),
        },
        ensure_ascii=False,
        indent=2,
    )
)
PY
