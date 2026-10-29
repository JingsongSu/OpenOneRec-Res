#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPU sparse-CSR, catalog-constrained residual-SID beam search.

Installs against the ORIGINAL pooler class from the existing model.py.
The original model.py remains entirely unchanged. No alteration to latent/residual transitions,
weight names, vLLM interface, beam size or full-vocabulary score normalization.

Index layout:
 k1 = a
 k2 = a*size_b+b
 k3 = (a*size_b+b)*size_c+c
 k4 = ((a*size_b+b)*size_c+c)*size_d+d
Each list is sorted, unique, signed int64.

Important guarantee:
- If catalog has >=beam_size distinct four-layer SIDs AND per_layer_topk
  >=beam_size, the four-stage beam produces beam_size finite, distinct,
  catalog-valid paths. Invalid placeholders (-inf) may exist INTERNALLY
  while a prefix level has fewer than beam_size valid choices, but never
  reach the final output.
- Normalization occurs before masking to preserve original joint scores.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def install_constraints(pooler_cls: type) -> None:
    if getattr(pooler_cls, "_daily_sid_catalog_installed", False):
        return

    original_init = pooler_cls.__init__
    original_expand = pooler_cls._expand_one_layer

    def constrained_init(self: Any, config: Any) -> None:
        original_init(self, config)

        index_path = os.environ.get("RESIDUAL_SID_TRIE_PATH")
        if not index_path:
            raise RuntimeError(
                "Constrained SID decoding requires RESIDUAL_SID_TRIE_PATH. "
                "Run the daily online inference launcher, which builds the "
                "snapshot and index before starting vLLM workers."
            )
        path = Path(index_path)
        if not path.is_file():
            raise FileNotFoundError(f"Daily valid SID index missing: {path}")

        with np.load(path, allow_pickle=False) as archive:
            sizes = tuple(int(x) for x in archive["sizes"].tolist())
            if sizes != tuple(self.layer_sizes):
                raise ValueError(
                    f"SID index sizes {sizes} do not match model "
                    f"{tuple(self.layer_sizes)}"
                )
            keys = tuple(
                np.asarray(archive[f"k{i}"], dtype=np.int64).copy()
                for i in range(1, 5)
            )

        for i, arr in enumerate(keys, 1):
            if arr.ndim != 1 or not arr.size:
                raise ValueError(f"Empty/malformed valid SID index k{i}")
            if np.any(arr[1:] <= arr[:-1]):
                raise ValueError(f"Index k{i} is not sorted and unique")
        if len(keys[-1]) < int(self.beam_size):
            raise ValueError(
                "Cannot return 50 different valid SID combinations: "
                f"catalog has only {len(keys[-1])} unique SID"
            )
        # Each parent must be able to return up to beam_size valid children,
        # otherwise a catalog with sufficient total SIDs might be truncated.
        for layer in (1, 2, 3):
            if int(self.per_layer_topk[layer]) < int(self.beam_size):
                raise ValueError(
                    f"per_layer_topk[{layer}]={self.per_layer_topk[layer]} "
                    f"must be >= beam_size={self.beam_size} for the strict "
                    "50-valid-SID guarantee (no inference parameter changed)."
                )

        # CSR prefix -> legal child lists. Read-only CPU copy; transferred to
        # each rank-0 GPU lazily once. This avoids V-wide binary search masking
        # in each beam and is especially useful for large D vocabulary.
        self._catalog_cpu_keys = tuple(torch.from_numpy(x) for x in keys)
        csr = []
        for layer in (1, 2, 3):
            parents = keys[layer - 1]
            children_packed = keys[layer]
            vocab = int(sizes[layer])
            starts = np.searchsorted(
                children_packed, parents * vocab, side="left"
            ).astype(np.int64)
            pointers = np.concatenate([
                starts,
                np.asarray([len(children_packed)], dtype=np.int64)
            ])
            degrees = np.diff(pointers)
            if np.any(degrees <= 0):
                raise ValueError(
                    f"Broken daily SID CSR index at layer {layer}"
                )
            child_ids = (children_packed % vocab).astype(np.int64)
            csr.append((
                torch.from_numpy(parents.copy()),
                torch.from_numpy(pointers),
                torch.from_numpy(child_ids),
                int(degrees.max()),
            ))
        self._catalog_cpu_csr = tuple(csr)
        self._catalog_gpu_cache = {}
        self._catalog_first_masks = {}
        self._catalog_active_prefixes = None
        self._catalog_source_path = str(path)
        # Debug only. Avoid a GPU->CPU sync every pooling batch by default;
        # the unchanged online decode_pooling_outputs_batch() already
        # checks all returned scores with np.isfinite on CPU.
        self._catalog_strict_gpu_check = (
            os.environ.get("RESIDUAL_SID_STRICT_GPU_CHECK", "0") == "1"
        )

    def gpu_catalog(self: Any, device: torch.device):
        cache_key = (device.type, device.index)
        result = self._catalog_gpu_cache.get(cache_key)
        if result is None:
            root = self._catalog_cpu_keys[0].to(device=device)
            csr = tuple(
                (
                    parent.to(device=device),
                    starts.to(device=device),
                    child_ids.to(device=device),
                    degree,
                )
                for parent, starts, child_ids, degree in self._catalog_cpu_csr
            )
            result = (root, csr)
            self._catalog_gpu_cache[cache_key] = result
        return result

    def first_layer_mask(self: Any, device: torch.device) -> torch.Tensor:
        cache_key = (device.type, device.index)
        cached = self._catalog_first_masks.get(cache_key)
        if cached is not None:
            return cached
        keys = gpu_catalog(self, device)[0]
        ids = torch.arange(
            self.layer_sizes[0], device=device, dtype=torch.int64
        )
        locations = torch.searchsorted(keys, ids.contiguous())
        mask = (
            (locations < keys.numel()) &
            (keys[locations.clamp(max=keys.numel()-1)] == ids)
        )
        self._catalog_first_masks[cache_key] = mask
        return mask

    def legal_child_scores(
        self: Any,
        log_probs: torch.Tensor,
        parent_prefix: torch.Tensor,
        layer: int,
        child_topk: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sparse GPU CSR lookup + gather from FULL-vocabulary log_probs.

        Only searchsorted on one key per parent, NOT every candidate in V.
        Pad legal child lists with -inf (internal placeholders only).
        """
        if parent_prefix.shape[-1] != layer:
            raise ValueError("Unexpected parent SID depth")
        parent_ids = parent_prefix[..., 0].long()
        for j in range(1, layer):
            parent_ids = parent_ids * int(self.layer_sizes[j]) \
                + parent_prefix[..., j].long()

        parents, ptr, child_ids, max_degree = (
            gpu_catalog(self, log_probs.device)[1][layer - 1]
        )
        locations = torch.searchsorted(parents, parent_ids.contiguous())
        safe = locations.clamp(max=parents.numel() - 1)
        valid_parent = (
            (locations < parents.numel()) &
            (parents[safe] == parent_ids)
        )
        start = ptr[safe]
        end = ptr[safe + 1]
        count = torch.where(
            valid_parent, end - start, torch.zeros_like(start)
        )
        # A common real-world codebook has D=0 for every item. In that case
        # max_degree=1: gather ONE candidate per parent, not 50 padded ones.
        width = max_degree
        slot = torch.arange(
            width, device=log_probs.device, dtype=torch.long
        )
        indices = start.unsqueeze(-1) + slot.view(1, 1, -1)
        valid_child = slot.view(1, 1, -1) < count.unsqueeze(-1)
        indices = indices.clamp(max=child_ids.numel() - 1)
        code = child_ids[indices]
        legal_scores = torch.gather(log_probs, dim=-1, index=code)
        legal_scores = legal_scores.masked_fill(
            ~valid_child, float("-inf")
        )
        k = min(child_topk, width)
        scores, positions = torch.topk(
            legal_scores, k=k, dim=-1, largest=True, sorted=True
        )
        local_ids = torch.gather(code, dim=-1, index=positions)

        # The existing _expand_one_layer relies on a fixed child_topk width.
        # Use -inf placeholders ONLY internally; never return them as
        # final candidates. This padding is small when D is constant zero.
        if k < child_topk:
            extra = child_topk - k
            scores = torch.cat([
                scores,
                torch.full(
                    (*scores.shape[:-1], extra),
                    float("-inf"), device=scores.device, dtype=scores.dtype
                )
            ], dim=-1)
            local_ids = torch.cat([
                local_ids,
                torch.zeros(
                    (*local_ids.shape[:-1], extra),
                    device=local_ids.device, dtype=local_ids.dtype
                )
            ], dim=-1)
        return scores, local_ids

    def constrained_layer_topk(
        self: Any,
        layer_index: int,
        hidden_states: torch.Tensor,
        child_topk: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Original F.linear/log_softmax unchanged; Top-K from valid children."""
        parent_sequences = self._catalog_active_prefixes
        if parent_sequences is None:
            raise RuntimeError("SID prefix context missing during expansion")
        if parent_sequences.shape[:2] != hidden_states.shape[:2]:
            raise ValueError("Parent-prefix shape mismatch")

        output_weight = self.sid_output_weights[layer_index]
        parent_count = int(hidden_states.shape[1])
        chunk_scores, chunk_ids = [], []
        for start in range(0, parent_count, self.parent_chunk_size):
            end = min(start + self.parent_chunk_size, parent_count)
            hidden = hidden_states[:, start:end, :]
            logits = F.linear(hidden.to(output_weight.dtype), output_weight)
            log_probs = F.log_softmax(logits.float(), dim=-1)
            score, child_id = legal_child_scores(
                self, log_probs, parent_sequences[:, start:end, :],
                layer_index, child_topk,
            )
            chunk_scores.append(score)
            chunk_ids.append(child_id)
        return torch.cat(chunk_scores, dim=1), torch.cat(chunk_ids, dim=1)

    def constrained_expand(
        self: Any,
        *,
        layer_index: int,
        current_hidden: torch.Tensor,
        current_sequences: torch.Tensor,
        current_scores: torch.Tensor,
    ):
        # Delegate ALL original latent and formal residual computations to
        # the exact deployed implementation. Only scoring is monkeypatched.
        if self._catalog_active_prefixes is not None:
            raise RuntimeError("Concurrent/reentrant SID pooler decode")
        self._catalog_active_prefixes = current_sequences
        try:
            return original_expand(
                self,
                layer_index=layer_index,
                current_hidden=current_hidden,
                current_sequences=current_sequences,
                current_scores=current_scores,
            )
        finally:
            self._catalog_active_prefixes = None

    def constrained_decode(self: Any, raw_anchor: torch.Tensor) -> torch.Tensor:
        if raw_anchor.ndim != 2:
            raise ValueError("raw_anchor must have shape [batch, hidden]")
        if raw_anchor.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Anchor size {raw_anchor.shape[-1]} != {self.hidden_size}"
            )

        batch = int(raw_anchor.shape[0])
        first_weight = self.sid_output_weights[0]
        first_logits = F.linear(raw_anchor.to(first_weight.dtype), first_weight)
        first_log_probs = F.log_softmax(first_logits.float(), dim=-1)
        first_log_probs = first_log_probs.masked_fill(
            ~first_layer_mask(self, raw_anchor.device).unsqueeze(0),
            float("-inf"),
        )

        # Maintain exactly beam_size slots at ALL levels. The -inf slots
        # are harmless placeholders when the catalog has few early prefixes.
        # They cannot enter the final finite top-50 if >=50 full SIDs exist.
        current_scores, first_local = torch.topk(
            first_log_probs, k=self.beam_size, dim=-1,
            largest=True, sorted=True,
        )
        current_sequences = first_local.unsqueeze(-1)
        current_hidden = raw_anchor.unsqueeze(1).expand(
            batch, self.beam_size, self.hidden_size
        ).contiguous()
        for layer in range(1, 4):
            current_hidden, current_sequences, current_scores = (
                self._expand_one_layer(
                    layer_index=layer,
                    current_hidden=current_hidden,
                    current_sequences=current_sequences,
                    current_scores=current_scores,
                )
            )
        if current_sequences.shape != (batch, self.beam_size, 4):
            raise RuntimeError(
                f"Unexpected constrained SID shape {current_sequences.shape}"
            )

        # Optional strict check for debugging. In production the EXISTING
        # CPU output decoder already checks nonfinite scores; not calling
        # .item() here avoids a per-pooler-batch device synchronization.
        if self._catalog_strict_gpu_check:
            if not bool(torch.isfinite(current_scores).all().item()):
                raise RuntimeError(
                    "Constrained beam exhausted valid candidates; inspect "
                    "the daily catalog and layer top-k configuration."
                )
        global_ids = current_sequences + self._layer_starts.to(
            device=current_sequences.device
        ).view(1, 1, 4)
        return torch.cat(
            [
                global_ids.to(dtype=torch.float32),
                current_scores.to(dtype=torch.float32).unsqueeze(-1),
            ],
            dim=-1,
        )

    pooler_cls.__init__ = constrained_init
    pooler_cls._catalog_gpu_catalog = gpu_catalog
    pooler_cls._catalog_first_layer_mask = first_layer_mask
    pooler_cls._catalog_legal_child_scores = legal_child_scores
    pooler_cls._layer_log_probs_and_topk = constrained_layer_topk
    pooler_cls._expand_one_layer = constrained_expand
    pooler_cls.decode = constrained_decode
    pooler_cls._daily_sid_catalog_installed = True


# Import existing vLLM architecture and install the constraints during the
# ModelRegistry target module import. This happens inside the worker process,
# before the pooler instance is constructed. NO edits to original model.py.
from .model import (
    Qwen3ForResidualSIDPoolingV085,
    ResidualSIDHierarchicalBeamPoolerV085,
)

install_constraints(ResidualSIDHierarchicalBeamPoolerV085)
