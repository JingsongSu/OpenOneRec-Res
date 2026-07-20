# """vLLM 0.8.5 custom Qwen3 model for four-layer residual SID decoding."""

# from __future__ import annotations

# from collections.abc import Iterable, Iterator
# from typing import Optional, Set, Tuple

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from vllm.config import VllmConfig
# from vllm.model_executor.pooling_metadata import (
#     PoolingMetadata,
#     PoolingTensors,
# )
# from vllm.model_executor.models.interfaces import SupportsV0Only
# from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
# from vllm.model_executor.models.utils import AutoWeightsLoader
# from vllm.sequence import PoolerOutput, PoolingSequenceGroupOutput

# from .codec import pack_candidates


# class ResidualSIDBlock(nn.Module):
#     """Must remain identical to the block trained during SFT."""

#     def __init__(
#         self,
#         hidden_size: int,
#         dropout: float,
#         dtype: torch.dtype,
#     ) -> None:
#         super().__init__()
#         self.linear = nn.Linear(
#             hidden_size * 2,
#             hidden_size,
#             dtype=dtype,
#         )
#         self.layer_norm = nn.LayerNorm(
#             hidden_size,
#             dtype=dtype,
#         )
#         self.activation = nn.ReLU()
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, fused: torch.Tensor) -> torch.Tensor:
#         return self.dropout(
#             self.activation(self.layer_norm(self.linear(fused)))
#         )


# class ResidualSIDBeamOncePoolerV085(nn.Module):
#     """One Top-B at layer a, then greedy residual b/c/d."""

#     def __init__(self, vllm_config: VllmConfig) -> None:
#         super().__init__()
#         model_config = vllm_config.model_config
#         config = model_config.hf_config
#         dtype = model_config.dtype
#         if not isinstance(dtype, torch.dtype):
#             raise TypeError(f"Expected torch.dtype, got {dtype!r}.")

#         self.layer_starts = tuple(
#             int(x) for x in config.residual_sid_layer_starts
#         )
#         self.layer_sizes = tuple(
#             int(x) for x in config.residual_sid_layer_sizes
#         )
#         if len(self.layer_starts) != 4 or len(self.layer_sizes) != 4:
#             raise ValueError(
#                 "Qwen3ForResidualSIDPoolingV085 requires four SID layers."
#             )

#         self.num_layers = 4
#         self.hidden_size = int(config.hidden_size)
#         self.sid_begin_token_id = int(
#             config.residual_sid_begin_token_id
#         )
#         self.sid_end_token_id = int(
#             config.residual_sid_end_token_id
#         )
#         self.beam_size = int(config.residual_sid_beam_size)
#         if not 0 < self.beam_size <= self.layer_sizes[0]:
#             raise ValueError(
#                 "residual_sid_beam_size must be in the first-layer range."
#             )

#         dropout = float(getattr(config, "residual_sid_dropout", 0.1))
#         # Four SID levels require three learned state transitions.
#         self.sid_residual_blocks = nn.ModuleList(
#             ResidualSIDBlock(
#                 hidden_size=self.hidden_size,
#                 dropout=dropout,
#                 dtype=dtype,
#             )
#             for _ in range(3)
#         )

#         # Complete replicated classifiers, exported for rank-0 pooling.
#         self.sid_output_weights = nn.ParameterList(
#             [
#                 nn.Parameter(
#                     torch.empty(
#                         size,
#                         self.hidden_size,
#                         dtype=dtype,
#                     ),
#                     requires_grad=False,
#                 )
#                 for size in self.layer_sizes
#             ]
#         )

#         self.embeddings_are_tied = bool(config.tie_word_embeddings)
#         if self.embeddings_are_tied:
#             self.sid_input_embeddings = nn.ParameterList()
#         else:
#             # a/b/c embeddings are consumed by transitions; d is not.
#             self.sid_input_embeddings = nn.ParameterList(
#                 [
#                     nn.Parameter(
#                         torch.empty(
#                             size,
#                             self.hidden_size,
#                             dtype=dtype,
#                         ),
#                         requires_grad=False,
#                     )
#                     for size in self.layer_sizes[:-1]
#                 ]
#             )

#     def _validate_prompt_endings(
#         self,
#         pooling_metadata: PoolingMetadata,
#     ) -> None:
#         for sequence_index, seq_data in enumerate(
#             pooling_metadata.seq_data.values()
#         ):
#             token_ids = seq_data.prompt_token_ids
#             if not token_ids:
#                 raise ValueError(f"Prompt {sequence_index} is empty.")
#             if int(token_ids[-1]) != self.sid_begin_token_id:
#                 raise ValueError(
#                     "Every residual-SID prompt must end with "
#                     f"<|sid_begin|> token ID {self.sid_begin_token_id}; "
#                     f"prompt {sequence_index} ended with {int(token_ids[-1])}."
#                 )

#     def _previous_embedding(
#         self,
#         previous_layer: int,
#         local_ids: torch.LongTensor,
#     ) -> torch.Tensor:
#         if self.embeddings_are_tied:
#             weight = self.sid_output_weights[previous_layer]
#         else:
#             weight = self.sid_input_embeddings[previous_layer]
#         return F.embedding(local_ids, weight)

#     def _layer_log_probs(
#         self,
#         hidden: torch.Tensor,
#         layer: int,
#     ) -> torch.Tensor:
#         weight = self.sid_output_weights[layer]
#         logits = F.linear(hidden.to(weight.dtype), weight)
#         return F.log_softmax(logits.float(), dim=-1)

#     def decode(
#         self,
#         anchors: torch.Tensor,
#     ) -> tuple[torch.LongTensor, torch.FloatTensor]:
#         batch_size, hidden_size = anchors.shape

#         # The only beam expansion in the improved inference method.
#         first_log_probs = self._layer_log_probs(anchors, layer=0)
#         scores, first_local_ids = torch.topk(
#             first_log_probs,
#             k=self.beam_size,
#             dim=-1,
#             sorted=True,
#         )

#         local_ids = torch.empty(
#             (batch_size, self.beam_size, self.num_layers),
#             dtype=torch.long,
#             device=anchors.device,
#         )
#         local_ids[:, :, 0] = first_local_ids

#         anchor_expanded = (
#             anchors[:, None, :]
#             .expand(batch_size, self.beam_size, hidden_size)
#             .reshape(batch_size * self.beam_size, hidden_size)
#         )
#         current = anchor_expanded
#         flat_scores = scores.reshape(-1)

#         for layer in range(1, self.num_layers):
#             previous_local_ids = local_ids[
#                 :, :, layer - 1
#             ].reshape(-1)
#             previous_embedding = self._previous_embedding(
#                 layer - 1,
#                 previous_local_ids,
#             )
#             fused = torch.cat(
#                 [anchor_expanded, previous_embedding],
#                 dim=-1,
#             )
#             current = (
#                 current
#                 + self.sid_residual_blocks[layer - 1](fused)
#             )

#             layer_log_probs = self._layer_log_probs(
#                 current,
#                 layer=layer,
#             )
#             # b/c/d each keep exactly one continuation per first-layer beam.
#             next_score, next_local_id = layer_log_probs.max(dim=-1)
#             local_ids[:, :, layer] = next_local_id.view(
#                 batch_size,
#                 self.beam_size,
#             )
#             flat_scores = flat_scores + next_score

#         scores = flat_scores.view(batch_size, self.beam_size)
#         sorted_scores, order = torch.sort(
#             scores,
#             dim=-1,
#             descending=True,
#         )
#         local_ids = torch.gather(
#             local_ids,
#             dim=1,
#             index=order[:, :, None].expand(
#                 -1,
#                 -1,
#                 self.num_layers,
#             ),
#         )
#         starts = torch.tensor(
#             self.layer_starts,
#             dtype=torch.long,
#             device=local_ids.device,
#         )
#         global_ids = local_ids + starts[None, None, :]
#         return global_ids, sorted_scores

#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         pooling_metadata: PoolingMetadata,
#     ) -> PoolerOutput:
#         self._validate_prompt_endings(pooling_metadata)

#         prompt_lens = PoolingTensors.from_pooling_metadata(
#             pooling_metadata,
#             hidden_states.device,
#         ).prompt_lens
#         last_token_flat_indices = torch.cumsum(
#             prompt_lens,
#             dim=0,
#         ) - 1
#         anchors = hidden_states[last_token_flat_indices]

#         global_ids, scores = self.decode(anchors)
#         packed = pack_candidates(global_ids, scores)
#         return PoolerOutput(
#             outputs=[
#                 PoolingSequenceGroupOutput(data=row)
#                 for row in packed
#             ]
#         )


# class Qwen3ForResidualSIDPoolingV085(
#     Qwen3ForCausalLM,
#     SupportsV0Only,
# ):
#     """Qwen3 prefill once; custom vLLM 0.8.5 pooler decodes all SID levels."""

#     def __init__(
#         self,
#         *,
#         vllm_config: VllmConfig,
#         prefix: str = "",
#         **kwargs,
#     ) -> None:
#         if vllm_config.model_config.runner_type != "pooling":
#             raise ValueError(
#                 "Initialize this architecture with task='embed' / --task embed."
#             )
#         if vllm_config.parallel_config.pipeline_parallel_size != 1:
#             raise ValueError(
#                 "Tensor parallelism is supported; pipeline parallelism must be 1."
#             )
#         super().__init__(
#             vllm_config=vllm_config,
#             prefix=prefix,
#             **kwargs,
#         )
#         self._pooler = ResidualSIDBeamOncePoolerV085(vllm_config)

#     def pooler(
#         self,
#         hidden_states: torch.Tensor,
#         pooling_metadata: PoolingMetadata,
#     ) -> Optional[PoolerOutput]:
#         return self._pooler(hidden_states, pooling_metadata)

#     def load_weights(
#         self,
#         weights: Iterable[Tuple[str, torch.Tensor]],
#     ) -> Set[str]:
#         def remap() -> Iterator[Tuple[str, torch.Tensor]]:
#             for name, tensor in weights:
#                 if name.startswith("sid_residual_blocks."):
#                     yield "_pooler." + name, tensor
#                 elif name.startswith("pooler."):
#                     yield "_pooler." + name[len("pooler."):], tensor
#                 else:
#                     yield name, tensor

#         loader = AutoWeightsLoader(
#             self,
#             skip_prefixes=(
#                 ["lm_head."]
#                 if self.config.tie_word_embeddings
#                 else None
#             ),
#         )
#         return loader.load_weights(remap())


# class Qwen3ForCausalLMIgnoreResidualSIDV085(Qwen3ForCausalLM):
#     """Generation baseline using the same HF directory but ignoring pooler weights."""

#     def load_weights(
#         self,
#         weights: Iterable[Tuple[str, torch.Tensor]],
#     ) -> Set[str]:
#         ignored_prefixes = (
#             "sid_residual_blocks.",
#             "pooler.",
#             "_pooler.",
#         )
#         filtered = (
#             (name, tensor)
#             for name, tensor in weights
#             if not name.startswith(ignored_prefixes)
#         )
#         return super().load_weights(filtered)





# SPDX-License-Identifier: Apache-2.0
"""
OpenOneRec four-layer residual SID model for vLLM 0.8.5.

Decode path:
    Qwen3 backbone/prefill: exactly once
    A layer: Top-B
    B/C/D layers:
        every surviving parent beam expands Top-K children
        accumulated log-probabilities are compared globally
        only the global best B paths survive

The returned pooling tensor keeps the existing format:
    [beam_size, 5]
    columns = [a_global_id, b_global_id, c_global_id, d_global_id, score]

This file is intended to replace:
    vllm085_plugin/openonerec_vllm085_residual_sid/model.py
"""

from __future__ import annotations

from typing import Iterable, Optional, Set, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.model_executor.pooling_metadata import (
    PoolingMetadata,
    PoolingTensors,
)
from vllm.sequence import (
    IntermediateTensors,
    PoolerOutput,
    PoolingSequenceGroupOutput,
)


NUM_SID_LAYERS = 4


class ResidualSIDBlockV085(nn.Module):
    """
    Training-compatible residual block:

        Linear(2H, H)
        LayerNorm(H)
        ReLU
        Dropout

    The block returns a delta. The caller performs:

        next_state = current_state + delta
    """

    def __init__(
        self,
        hidden_size: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.proj = nn.Linear(
            hidden_size * 2,
            hidden_size,
        )
        self.norm = nn.LayerNorm(
            hidden_size,
        )
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(
            dropout,
        )

    def forward(
        self,
        anchor_hidden: torch.Tensor,
        previous_token_embedding: torch.Tensor,
    ) -> torch.Tensor:
        if anchor_hidden.shape != previous_token_embedding.shape:
            raise ValueError(
                "Residual block input shapes do not match: "
                f"anchor={tuple(anchor_hidden.shape)}, "
                f"embedding={tuple(previous_token_embedding.shape)}"
            )

        fused = torch.cat(
            [
                anchor_hidden,
                previous_token_embedding,
            ],
            dim=-1,
        )

        delta = self.proj(
            fused
        )
        delta = self.norm(
            delta
        )
        delta = self.activation(
            delta
        )
        delta = self.dropout(
            delta
        )

        return delta


class ResidualSIDHierarchicalBeamPoolerV085(nn.Module):
    """
    Four-layer residual SID hierarchical beam decoder.

    Important properties:

    1. The Qwen backbone is not called here. This pooler receives the flattened
       hidden states produced by the single vLLM prefill.

    2. The anchor hidden state is the hidden state of the final prompt token.
       The evaluation prompt must therefore end with residual_sid_begin_token_id.

    3. Every SID layer performs beam expansion. For B/C/D, each surviving
       parent expands child_topk candidates and the global best beam_size paths
       are retained.

    4. Log-probabilities are normalized inside each SID layer before scores are
       accumulated. Raw logits are never accumulated across layers.
    """

    def __init__(
        self,
        config,
    ) -> None:
        super().__init__()

        self.hidden_size = int(
            config.hidden_size
        )

        self.layer_starts = [
            int(value)
            for value in getattr(
                config,
                "residual_sid_layer_starts",
                [],
            )
        ]

        self.layer_sizes = [
            int(value)
            for value in getattr(
                config,
                "residual_sid_layer_sizes",
                [],
            )
        ]

        self.sid_begin_token_id = int(
            getattr(
                config,
                "residual_sid_begin_token_id",
                -1,
            )
        )

        self.beam_size = int(
            getattr(
                config,
                "residual_sid_beam_size",
                0,
            )
        )

        self.dropout_probability = float(
            getattr(
                config,
                "residual_sid_dropout",
                0.0,
            )
        )

        self.tie_word_embeddings = bool(
            getattr(
                config,
                "tie_word_embeddings",
                False,
            )
        )

        self.validate_prompt_ends_with_sid_begin = bool(
            getattr(
                config,
                "residual_sid_validate_prompt_end",
                True,
            )
        )

        # Controls only temporary logits memory. This does not change results.
        # With beam=100 and D vocab=15000, 8 or 16 is usually much safer than
        # materializing [batch, 100, 15000] float32 log-probabilities at once.
        self.parent_chunk_size = int(
            getattr(
                config,
                "residual_sid_parent_chunk_size",
                16,
            )
        )

        raw_per_layer_topk = getattr(
            config,
            "residual_sid_per_layer_topk",
            None,
        )

        if raw_per_layer_topk is None:
            # Exact reference behavior: Top-B expansion at every SID layer.
            self.per_layer_topk = [
                self.beam_size
                for _ in range(
                    NUM_SID_LAYERS
                )
            ]

        elif isinstance(
            raw_per_layer_topk,
            int,
        ):
            self.per_layer_topk = [
                int(
                    raw_per_layer_topk
                )
                for _ in range(
                    NUM_SID_LAYERS
                )
            ]

        else:
            self.per_layer_topk = [
                int(value)
                for value in list(
                    raw_per_layer_topk
                )
            ]

        self._validate_config()

        # Full, unsharded output matrices exported specifically for the rank-0
        # vLLM pooler. Shape of each parameter: [layer_vocab_size, hidden_size].
        self.sid_output_weights = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(
                        layer_size,
                        self.hidden_size,
                    ),
                    requires_grad=False,
                )
                for layer_size in self.layer_sizes
            ]
        )

        # If embeddings are tied, the preceding layer's output matrix is also
        # its input embedding table. Otherwise the exporter supplies three
        # explicit input embedding slices for A, B and C.
        if self.tie_word_embeddings:
            self.sid_input_embeddings = nn.ParameterList()

        else:
            self.sid_input_embeddings = nn.ParameterList(
                [
                    nn.Parameter(
                        torch.empty(
                            self.layer_sizes[
                                transition_index
                            ],
                            self.hidden_size,
                        ),
                        requires_grad=False,
                    )
                    for transition_index in range(
                        NUM_SID_LAYERS - 1
                    )
                ]
            )

        self.sid_residual_blocks = nn.ModuleList(
            [
                ResidualSIDBlockV085(
                    hidden_size=self.hidden_size,
                    dropout=self.dropout_probability,
                )
                for _ in range(
                    NUM_SID_LAYERS - 1
                )
            ]
        )

        self.register_buffer(
            "_layer_starts",
            torch.tensor(
                self.layer_starts,
                dtype=torch.long,
            ),
            persistent=False,
        )

    def _validate_config(
        self,
    ) -> None:
        if len(
            self.layer_starts
        ) != NUM_SID_LAYERS:
            raise ValueError(
                "Expected four residual SID layer starts, "
                f"got {self.layer_starts}"
            )

        if len(
            self.layer_sizes
        ) != NUM_SID_LAYERS:
            raise ValueError(
                "Expected four residual SID layer sizes, "
                f"got {self.layer_sizes}"
            )

        if any(
            size <= 0
            for size in self.layer_sizes
        ):
            raise ValueError(
                "Every residual SID layer size must be positive: "
                f"{self.layer_sizes}"
            )

        if self.sid_begin_token_id < 0:
            raise ValueError(
                "residual_sid_begin_token_id is missing or invalid."
            )

        if self.beam_size <= 0:
            raise ValueError(
                "residual_sid_beam_size is missing or invalid."
            )

        # The output codec expects exactly beam_size candidates.
        if self.beam_size > min(
            self.layer_sizes
        ):
            raise ValueError(
                "beam_size must not exceed the smallest SID vocabulary: "
                f"beam={self.beam_size}, "
                f"layer_sizes={self.layer_sizes}"
            )

        if len(
            self.per_layer_topk
        ) != NUM_SID_LAYERS:
            raise ValueError(
                "residual_sid_per_layer_topk must contain four values, "
                f"got {self.per_layer_topk}"
            )

        if any(
            value <= 0
            for value in self.per_layer_topk
        ):
            raise ValueError(
                "Every per-layer top-k value must be positive: "
                f"{self.per_layer_topk}"
            )

        if self.parent_chunk_size <= 0:
            raise ValueError(
                "residual_sid_parent_chunk_size must be positive."
            )

    def _validate_prompt_end(
        self,
        pooling_metadata: PoolingMetadata,
    ) -> None:
        if not self.validate_prompt_ends_with_sid_begin:
            return

        invalid_sequences = []

        for sequence_id, sequence_data in (
            pooling_metadata
            .seq_data
            .items()
        ):
            prompt_token_ids = getattr(
                sequence_data,
                "prompt_token_ids",
                None,
            )

            if not prompt_token_ids:
                invalid_sequences.append(
                    (
                        int(sequence_id),
                        None,
                    )
                )
                continue

            final_token_id = int(
                prompt_token_ids[-1]
            )

            if final_token_id != (
                self.sid_begin_token_id
            ):
                invalid_sequences.append(
                    (
                        int(sequence_id),
                        final_token_id,
                    )
                )

        if invalid_sequences:
            preview = (
                invalid_sequences[:8]
            )

            raise ValueError(
                "Residual SID pooling requires every prompt to end with "
                "the sid_begin token.\n"
                f"Expected token ID: {self.sid_begin_token_id}\n"
                f"Invalid sequence preview: {preview}"
            )

    def _get_previous_embedding(
        self,
        transition_index: int,
        previous_local_ids: torch.Tensor,
    ) -> torch.Tensor:
        if self.tie_word_embeddings:
            embedding_weight = (
                self.sid_output_weights[
                    transition_index
                ]
            )
        else:
            embedding_weight = (
                self.sid_input_embeddings[
                    transition_index
                ]
            )

        return F.embedding(
            previous_local_ids,
            embedding_weight,
        )

    def _layer_log_probs_and_topk(
        self,
        layer_index: int,
        hidden_states: torch.Tensor,
        child_topk: int,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Compute one SID layer's normalized Top-K.

        hidden_states:
            [batch, parent_count, hidden_size]

        returns:
            child_log_probs: [batch, parent_count, child_topk]
            child_local_ids: [batch, parent_count, child_topk]
        """
        output_weight = (
            self.sid_output_weights[
                layer_index
            ]
        )

        parent_count = int(
            hidden_states.shape[1]
        )

        chunk_scores = []
        chunk_ids = []

        for parent_start in range(
            0,
            parent_count,
            self.parent_chunk_size,
        ):
            parent_end = min(
                parent_start
                + self.parent_chunk_size,
                parent_count,
            )

            hidden_chunk = hidden_states[
                :,
                parent_start:parent_end,
                :,
            ]

            # [batch, parent_chunk, layer_vocab]
            logits = F.linear(
                hidden_chunk,
                output_weight,
            )

            # Scores from layers with different vocabulary sizes are only
            # comparable after layer-local normalization.
            log_probs = F.log_softmax(
                logits.float(),
                dim=-1,
            )

            top_log_probs, top_local_ids = (
                torch.topk(
                    log_probs,
                    k=child_topk,
                    dim=-1,
                    largest=True,
                    sorted=True,
                )
            )

            chunk_scores.append(
                top_log_probs
            )
            chunk_ids.append(
                top_local_ids
            )

        return (
            torch.cat(
                chunk_scores,
                dim=1,
            ),
            torch.cat(
                chunk_ids,
                dim=1,
            ),
        )

    def _expand_one_layer(
        self,
        *,
        layer_index: int,
        anchor_hidden: torch.Tensor,
        current_hidden: torch.Tensor,
        current_sequences: torch.Tensor,
        current_scores: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Expand all surviving beams through one residual layer, then globally
        retain beam_size paths.

        current_hidden:
            state used to predict the preceding SID layer,
            [batch, beam, hidden]

        current_sequences:
            local IDs selected so far,
            [batch, beam, layer_index]

        current_scores:
            accumulated layer-normalized log-probabilities,
            [batch, beam]
        """
        previous_local_ids = (
            current_sequences[
                :,
                :,
                -1,
            ]
        )

        previous_embedding = (
            self._get_previous_embedding(
                transition_index=(
                    layer_index - 1
                ),
                previous_local_ids=(
                    previous_local_ids
                ),
            )
        )

        expanded_anchor = (
            anchor_hidden
            .unsqueeze(1)
            .expand_as(
                current_hidden
            )
        )

        residual_delta = (
            self.sid_residual_blocks[
                layer_index - 1
            ](
                expanded_anchor,
                previous_embedding,
            )
        )

        next_hidden_all_parents = (
            current_hidden
            + residual_delta
        )

        child_topk = min(
            int(
                self.per_layer_topk[
                    layer_index
                ]
            ),
            int(
                self.layer_sizes[
                    layer_index
                ]
            ),
        )

        child_log_probs, child_local_ids = (
            self._layer_log_probs_and_topk(
                layer_index=layer_index,
                hidden_states=(
                    next_hidden_all_parents
                ),
                child_topk=child_topk,
            )
        )

        total_child_scores = (
            current_scores
            .unsqueeze(-1)
            + child_log_probs
        )

        batch_size = int(
            total_child_scores.shape[0]
        )

        flattened_scores = (
            total_child_scores
            .reshape(
                batch_size,
                -1,
            )
        )

        next_scores, flattened_indices = (
            torch.topk(
                flattened_scores,
                k=self.beam_size,
                dim=-1,
                largest=True,
                sorted=True,
            )
        )

        parent_indices = (
            flattened_indices
            // child_topk
        )

        child_slots = (
            flattened_indices
            % child_topk
        )

        batch_indices = (
            torch.arange(
                batch_size,
                device=(
                    current_hidden.device
                ),
            )
            .unsqueeze(1)
        )

        selected_child_local_ids = (
            child_local_ids[
                batch_indices,
                parent_indices,
                child_slots,
            ]
        )

        selected_parent_hidden = (
            next_hidden_all_parents[
                batch_indices,
                parent_indices,
                :,
            ]
        )

        selected_parent_sequences = (
            current_sequences[
                batch_indices,
                parent_indices,
                :,
            ]
        )

        next_sequences = torch.cat(
            [
                selected_parent_sequences,
                selected_child_local_ids
                .unsqueeze(-1),
            ],
            dim=-1,
        )

        return (
            selected_parent_hidden,
            next_sequences,
            next_scores,
        )

    def decode(
        self,
        anchor_hidden: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode a batch of anchor hidden states.

        returns:
            [batch, beam_size, 5]
        """
        if anchor_hidden.ndim != 2:
            raise ValueError(
                "anchor_hidden must have shape [batch, hidden], "
                f"got {tuple(anchor_hidden.shape)}"
            )

        if anchor_hidden.shape[-1] != (
            self.hidden_size
        ):
            raise ValueError(
                "Unexpected anchor hidden size: "
                f"{anchor_hidden.shape[-1]} "
                f"!= {self.hidden_size}"
            )

        batch_size = int(
            anchor_hidden.shape[0]
        )

        # Layer A: one parent per request, Top-B candidates.
        first_logits = F.linear(
            anchor_hidden,
            self.sid_output_weights[0],
        )

        first_log_probs = F.log_softmax(
            first_logits.float(),
            dim=-1,
        )

        first_scores, first_local_ids = (
            torch.topk(
                first_log_probs,
                k=self.beam_size,
                dim=-1,
                largest=True,
                sorted=True,
            )
        )

        current_sequences = (
            first_local_ids
            .unsqueeze(-1)
        )

        # state_a = anchor. Each A candidate starts from the same anchor state.
        current_hidden = (
            anchor_hidden
            .unsqueeze(1)
            .expand(
                batch_size,
                self.beam_size,
                self.hidden_size,
            )
            .contiguous()
        )

        current_scores = (
            first_scores
        )

        # B, C and D all continue hierarchical beam search.
        for layer_index in range(
            1,
            NUM_SID_LAYERS,
        ):
            (
                current_hidden,
                current_sequences,
                current_scores,
            ) = self._expand_one_layer(
                layer_index=layer_index,
                anchor_hidden=anchor_hidden,
                current_hidden=current_hidden,
                current_sequences=(
                    current_sequences
                ),
                current_scores=current_scores,
            )

        if current_sequences.shape != (
            batch_size,
            self.beam_size,
            NUM_SID_LAYERS,
        ):
            raise RuntimeError(
                "Unexpected residual SID sequence shape: "
                f"{tuple(current_sequences.shape)}"
            )

        global_ids = (
            current_sequences
            + self._layer_starts
            .to(
                device=(
                    current_sequences.device
                )
            )
            .view(
                1,
                1,
                NUM_SID_LAYERS,
            )
        )

        # Float32 exactly represents these token IDs and is compatible with
        # PoolingSequenceGroupOutput data plus the existing NumPy decoder.
        packed_output = torch.cat(
            [
                global_ids.to(
                    dtype=torch.float32
                ),
                current_scores
                .to(
                    dtype=torch.float32
                )
                .unsqueeze(-1),
            ],
            dim=-1,
        )

        return packed_output

    @torch.inference_mode()
    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> PoolerOutput:
        self._validate_prompt_end(
            pooling_metadata
        )

        prompt_lens = (
            PoolingTensors
            .from_pooling_metadata(
                pooling_metadata,
                hidden_states.device,
            )
            .prompt_lens
        )

        if prompt_lens.numel() == 0:
            return PoolerOutput(
                outputs=[]
            )

        last_token_flat_indices = (
            torch.cumsum(
                prompt_lens,
                dim=0,
            )
            - 1
        )

        if int(
            last_token_flat_indices[-1]
        ) >= int(
            hidden_states.shape[0]
        ):
            raise RuntimeError(
                "Pooling metadata points beyond flattened hidden states: "
                f"last_index={int(last_token_flat_indices[-1])}, "
                f"hidden_rows={hidden_states.shape[0]}"
            )

        anchor_hidden = (
            hidden_states
            .index_select(
                0,
                last_token_flat_indices,
            )
        )

        decoded = self.decode(
            anchor_hidden
        )

        outputs = [
            PoolingSequenceGroupOutput(
                request_data
            )
            for request_data in decoded
        ]

        return PoolerOutput(
            outputs=outputs
        )


def _canonicalize_residual_block_suffix(
    suffix: str,
) -> str:
    """
    Accept common checkpoint naming variants while keeping this implementation's
    canonical module names: proj.* and norm.*.
    """
    aliases = {
        "proj.weight": "proj.weight",
        "proj.bias": "proj.bias",
        "linear.weight": "proj.weight",
        "linear.bias": "proj.bias",
        "projection.weight": "proj.weight",
        "projection.bias": "proj.bias",
        "dense.weight": "proj.weight",
        "dense.bias": "proj.bias",
        "fc.weight": "proj.weight",
        "fc.bias": "proj.bias",
        "0.weight": "proj.weight",
        "0.bias": "proj.bias",
        "net.0.weight": "proj.weight",
        "net.0.bias": "proj.bias",
        "network.0.weight": "proj.weight",
        "network.0.bias": "proj.bias",
        "layers.0.weight": "proj.weight",
        "layers.0.bias": "proj.bias",
        "block.0.weight": "proj.weight",
        "block.0.bias": "proj.bias",
        "mlp.0.weight": "proj.weight",
        "mlp.0.bias": "proj.bias",
        "fusion.0.weight": "proj.weight",
        "fusion.0.bias": "proj.bias",
        "fusion_layer.0.weight": "proj.weight",
        "fusion_layer.0.bias": "proj.bias",
        "norm.weight": "norm.weight",
        "norm.bias": "norm.bias",
        "layer_norm.weight": "norm.weight",
        "layer_norm.bias": "norm.bias",
        "layernorm.weight": "norm.weight",
        "layernorm.bias": "norm.bias",
        "ln.weight": "norm.weight",
        "ln.bias": "norm.bias",
        "1.weight": "norm.weight",
        "1.bias": "norm.bias",
        "net.1.weight": "norm.weight",
        "net.1.bias": "norm.bias",
        "network.1.weight": "norm.weight",
        "network.1.bias": "norm.bias",
        "layers.1.weight": "norm.weight",
        "layers.1.bias": "norm.bias",
        "block.1.weight": "norm.weight",
        "block.1.bias": "norm.bias",
        "mlp.1.weight": "norm.weight",
        "mlp.1.bias": "norm.bias",
        "fusion.1.weight": "norm.weight",
        "fusion.1.bias": "norm.bias",
        "fusion_layer.1.weight": "norm.weight",
        "fusion_layer.1.bias": "norm.bias",
    }

    return aliases.get(
        suffix,
        suffix,
    )


def _canonicalize_pooler_weight_name(
    name: str,
    *,
    tie_word_embeddings: bool,
) -> Optional[str]:
    """
    Map both the original SFT checkpoint names and the exported sidecar names
    into this model's `_residual_pooler.*` module.

    Existing exporter conventions accepted here include:
        pooler.sid_output_weights.*
        pooler.sid_input_embeddings.*
        pooler.sid_residual_blocks.*
        sid_residual_blocks.*
    """
    mapped = name

    prefix_aliases = (
        (
            "model.sid_residual_blocks.",
            "_residual_pooler.sid_residual_blocks.",
        ),
        (
            "sid_residual_blocks.",
            "_residual_pooler.sid_residual_blocks.",
        ),
        (
            "pooler.sid_residual_blocks.",
            "_residual_pooler.sid_residual_blocks.",
        ),
        (
            "pooler.residual_blocks.",
            "_residual_pooler.sid_residual_blocks.",
        ),
        (
            "residual_sid_pooler.sid_residual_blocks.",
            "_residual_pooler.sid_residual_blocks.",
        ),
        (
            "pooler.sid_output_weights.",
            "_residual_pooler.sid_output_weights.",
        ),
        (
            "pooler.sid_output_weight.",
            "_residual_pooler.sid_output_weights.",
        ),
        (
            "sid_output_weights.",
            "_residual_pooler.sid_output_weights.",
        ),
        (
            "pooler.sid_input_embeddings.",
            "_residual_pooler.sid_input_embeddings.",
        ),
        (
            "pooler.sid_input_embedding.",
            "_residual_pooler.sid_input_embeddings.",
        ),
        (
            "sid_input_embeddings.",
            "_residual_pooler.sid_input_embeddings.",
        ),
        (
            "_residual_pooler.",
            "_residual_pooler.",
        ),
    )

    for old_prefix, new_prefix in prefix_aliases:
        if mapped.startswith(
            old_prefix
        ):
            mapped = (
                new_prefix
                + mapped[
                    len(
                        old_prefix
                    ):
                ]
            )
            break

    if (
        tie_word_embeddings
        and mapped.startswith(
            "_residual_pooler."
            "sid_input_embeddings."
        )
    ):
        # Tied models reuse the preceding layer's output matrix.
        return None

    residual_prefix = (
        "_residual_pooler."
        "sid_residual_blocks."
    )

    if mapped.startswith(
        residual_prefix
    ):
        remaining = mapped[
            len(
                residual_prefix
            ):
        ]

        if "." not in remaining:
            return mapped

        block_index, suffix = (
            remaining.split(
                ".",
                1,
            )
        )

        suffix = (
            _canonicalize_residual_block_suffix(
                suffix
            )
        )

        mapped = (
            residual_prefix
            + block_index
            + "."
            + suffix
        )

    return mapped


def _is_residual_sidecar_weight(
    name: str,
) -> bool:
    prefixes = (
        "sid_residual_blocks.",
        "model.sid_residual_blocks.",
        "pooler.",
        "residual_sid_pooler.",
        "_residual_pooler.",
        "sid_output_weights.",
        "sid_input_embeddings.",
    )

    return name.startswith(
        prefixes
    )


class Qwen3ForResidualSIDPoolingV085(
    Qwen3ForCausalLM
):
    """
    Qwen3 backbone plus residual SID hierarchical beam pooling.

    The inherited Qwen3 forward performs the ordinary vLLM prefill. vLLM then
    calls `pooler(...)` on TP rank 0.
    """

    supports_v0_only = True

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
        )

        self._residual_pooler = (
            ResidualSIDHierarchicalBeamPoolerV085(
                self.config
            )
        )

    def pooler(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> PoolerOutput:
        return self._residual_pooler(
            hidden_states,
            pooling_metadata,
        )

    def load_weights(
        self,
        weights: Iterable[
            Tuple[
                str,
                torch.Tensor,
            ]
        ],
    ) -> Set[str]:
        tie_word_embeddings = bool(
            self.config.tie_word_embeddings
        )

        def mapped_weights():
            for name, loaded_weight in weights:
                mapped_name = (
                    _canonicalize_pooler_weight_name(
                        name,
                        tie_word_embeddings=(
                            tie_word_embeddings
                        ),
                    )
                )

                if mapped_name is None:
                    continue

                yield (
                    mapped_name,
                    loaded_weight,
                )

        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(
                ["lm_head."]
                if tie_word_embeddings
                else None
            ),
        )

        return loader.load_weights(
            mapped_weights()
        )


class Qwen3ForCausalLMIgnoreResidualSIDV085(
    Qwen3ForCausalLM
):
    """
    Normal autoregressive Qwen3 loader that ignores residual sidecar tensors.

    This preserves the architecture name already registered by the plugin and
    allows the same exported directory/checkpoint to be used for an AR baseline.
    """

    supports_v0_only = True

    def load_weights(
        self,
        weights: Iterable[
            Tuple[
                str,
                torch.Tensor,
            ]
        ],
    ) -> Set[str]:
        filtered_weights = (
            (
                name,
                loaded_weight,
            )
            for name, loaded_weight in weights
            if not _is_residual_sidecar_weight(
                name
            )
        )

        return super().load_weights(
            filtered_weights
        )
