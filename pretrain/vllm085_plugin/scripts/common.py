from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

from transformers import AutoConfig, AutoTokenizer
from vllm import LLM
from vllm.sampling_params import SamplingParams

from openonerec_vllm085_residual_sid.codec import (
    SIDCandidate,
    candidates_to_records,
    unpack_data,
)


def load_jsonl(path: str | Path, limit: int = 0) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}"
                ) from exc
            if limit and len(rows) >= limit:
                break
    return rows


def model_layout(model_dir: str | Path) -> dict:
    config = AutoConfig.from_pretrained(
        model_dir,
        trust_remote_code=True,
    )
    starts = [int(x) for x in config.residual_sid_layer_starts]
    sizes = [int(x) for x in config.residual_sid_layer_sizes]
    if len(starts) != 4 or len(sizes) != 4:
        raise ValueError("The exported model is not a four-layer SID model.")
    return {
        "beam_size": int(config.residual_sid_beam_size),
        "layer_starts": starts,
        "layer_sizes": sizes,
        "sid_begin_token_id": int(config.residual_sid_begin_token_id),
        "sid_end_token_id": int(config.residual_sid_end_token_id),
    }


def tokenizer_for(model_dir: str | Path):
    return AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=True,
    )


def tokenize_prompts(
    rows: Sequence[dict],
    tokenizer,
    sid_begin_token_id: int,
) -> list[dict]:
    prompts = []
    for row_index, row in enumerate(rows):
        if "prompt_token_ids" in row:
            token_ids = [int(x) for x in row["prompt_token_ids"]]
        elif "prompt" in row:
            token_ids = tokenizer.encode(
                str(row["prompt"]),
                add_special_tokens=False,
            )
        else:
            raise ValueError(
                f"Row {row_index} needs prompt or prompt_token_ids."
            )
        if not token_ids or token_ids[-1] != sid_begin_token_id:
            raise ValueError(
                f"Row {row_index} does not end with <|sid_begin|> "
                f"token ID {sid_begin_token_id}."
            )
        prompts.append({"prompt_token_ids": token_ids})
    return prompts


def target_global_ids(
    row: dict,
    tokenizer,
    num_layers: int = 4,
) -> list[int]:
    if "target_global_ids" in row:
        ids = [int(x) for x in row["target_global_ids"]]
    elif "target_tokens" in row:
        ids = [
            int(x)
            for x in tokenizer.convert_tokens_to_ids(
                row["target_tokens"]
            )
        ]
    elif "sid_tokens" in row:
        ids = [
            int(x)
            for x in tokenizer.convert_tokens_to_ids(
                row["sid_tokens"]
            )
        ]
    else:
        raise ValueError(
            "Evaluation rows need target_global_ids, target_tokens, "
            "or sid_tokens."
        )
    if len(ids) != num_layers:
        raise ValueError(
            f"Target SID depth is {len(ids)}; expected {num_layers}."
        )
    return ids


def parse_pooling_output(
    output: Any,
    *,
    beam_size: int,
    num_layers: int = 4,
) -> list[SIDCandidate]:
    return unpack_data(
        output.outputs.data,
        beam_size=beam_size,
        num_layers=num_layers,
    )


def records_with_tokens(
    candidates: Sequence[SIDCandidate],
    tokenizer,
    layer_starts: Sequence[int],
) -> list[dict]:
    records = candidates_to_records(candidates, layer_starts)
    for record in records:
        record["tokens"] = tokenizer.convert_ids_to_tokens(
            record["global_ids"]
        )
    return records


@dataclass(frozen=True)
class _BeamPath:
    prompt_token_ids: tuple[int, ...]
    sid_ids: tuple[int, ...]
    score: float


def layer_restricted_autoregressive_beam(
    llm: LLM,
    prompt_inputs: Sequence[dict],
    *,
    layer_starts: Sequence[int],
    layer_sizes: Sequence[int],
    beam_size: int,
) -> list[list[SIDCandidate]]:
    """Exact layer-constrained beam using vLLM 0.8.5 one-token generate calls.

    This follows the same search semantics as ordinary autoregressive beam:
    every layer expands every surviving beam, then globally keeps B paths.
    `allowed_token_ids` ensures layer l can only emit its own SID vocabulary.
    """
    if beam_size <= 0:
        raise ValueError("beam_size must be positive.")

    beams: list[list[_BeamPath]] = [
        [
            _BeamPath(
                prompt_token_ids=tuple(
                    int(x) for x in prompt["prompt_token_ids"]
                ),
                sid_ids=(),
                score=0.0,
            )
        ]
        for prompt in prompt_inputs
    ]

    for layer, (start, size) in enumerate(
        zip(layer_starts, layer_sizes)
    ):
        allowed = list(range(int(start), int(start) + int(size)))
        top_logprobs = min(int(size), max(2 * beam_size, beam_size))
        params = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            min_tokens=1,
            logprobs=top_logprobs,
            detokenize=False,
            ignore_eos=True,
            allowed_token_ids=allowed,
        )

        flat_paths: list[_BeamPath] = []
        owners: list[int] = []
        for owner, instance_beams in enumerate(beams):
            for path in instance_beams:
                owners.append(owner)
                flat_paths.append(path)

        generation_inputs = [
            {
                "prompt_token_ids": (
                    list(path.prompt_token_ids)
                    + list(path.sid_ids)
                )
            }
            for path in flat_paths
        ]
        outputs = llm.generate(
            generation_inputs,
            params,
            use_tqdm=False,
        )

        expanded: list[list[_BeamPath]] = [
            [] for _ in range(len(beams))
        ]
        for owner, path, output in zip(owners, flat_paths, outputs):
            completion = output.outputs[0]
            if not completion.logprobs:
                continue
            token_logprobs = completion.logprobs[0]
            ranked = sorted(
                (
                    (int(token_id), float(logprob.logprob))
                    for token_id, logprob in token_logprobs.items()
                    if int(start) <= int(token_id) < int(start) + int(size)
                ),
                key=lambda item: item[1],
                reverse=True,
            )
            for token_id, token_logprob in ranked[:top_logprobs]:
                expanded[owner].append(
                    _BeamPath(
                        prompt_token_ids=path.prompt_token_ids,
                        sid_ids=path.sid_ids + (token_id,),
                        score=path.score + token_logprob,
                    )
                )

        new_beams = []
        for owner, candidates in enumerate(expanded):
            if not candidates:
                raise RuntimeError(
                    f"No valid SID candidates survived at layer {layer} "
                    f"for request {owner}."
                )
            candidates.sort(key=lambda path: path.score, reverse=True)
            new_beams.append(candidates[:beam_size])
        beams = new_beams

    return [
        [
            SIDCandidate(global_ids=path.sid_ids, score=path.score)
            for path in instance_beams
        ]
        for instance_beams in beams
    ]
