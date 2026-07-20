"""Encode/decode the custom vLLM 0.8.5 pooling data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch


@dataclass(frozen=True)
class SIDCandidate:
    global_ids: tuple[int, ...]
    score: float

    def to_dict(self) -> dict:
        return {
            "global_ids": list(self.global_ids),
            "score": float(self.score),
        }


def pack_candidates(
    global_ids: torch.LongTensor,
    scores: torch.Tensor,
) -> torch.FloatTensor:
    """Return [batch, beam, num_layers + 1]."""
    if global_ids.ndim != 3:
        raise ValueError("global_ids must be [batch, beam, layers].")
    if scores.shape != global_ids.shape[:2]:
        raise ValueError("scores must be [batch, beam].")
    return torch.cat(
        [global_ids.to(torch.float32), scores[..., None].float()],
        dim=-1,
    )


def unpack_data(
    data: Sequence[Sequence[float]] | Sequence[float] | torch.Tensor,
    *,
    beam_size: int,
    num_layers: int,
) -> list[SIDCandidate]:
    tensor = torch.as_tensor(data, dtype=torch.float32)
    stride = num_layers + 1
    if tensor.ndim == 1:
        expected = beam_size * stride
        if tensor.numel() != expected:
            raise ValueError(
                f"Flat pooling output has {tensor.numel()} values; "
                f"expected {expected}."
            )
        tensor = tensor.reshape(beam_size, stride)
    elif tensor.ndim == 2:
        if tuple(tensor.shape) != (beam_size, stride):
            raise ValueError(
                f"Pooling output shape is {tuple(tensor.shape)}; "
                f"expected {(beam_size, stride)}."
            )
    else:
        raise ValueError(
            f"Pooling output must be 1-D or 2-D, got {tensor.ndim}-D."
        )

    candidates = []
    for row in tensor:
        ids = tuple(
            int(round(float(value)))
            for value in row[:num_layers]
        )
        candidates.append(
            SIDCandidate(
                global_ids=ids,
                score=float(row[num_layers]),
            )
        )
    return candidates


def candidates_to_records(
    candidates: Iterable[SIDCandidate],
    layer_starts: Sequence[int],
) -> list[dict]:
    starts = tuple(int(x) for x in layer_starts)
    records = []
    for candidate in candidates:
        if len(candidate.global_ids) != len(starts):
            raise ValueError("SID depth does not match layer_starts.")
        records.append(
            {
                "global_ids": list(candidate.global_ids),
                "local_ids": [
                    global_id - start
                    for global_id, start in zip(
                        candidate.global_ids, starts
                    )
                ],
                "score": float(candidate.score),
            }
        )
    return records
