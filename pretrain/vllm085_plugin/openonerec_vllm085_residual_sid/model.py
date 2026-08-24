"""OpenOneRec branch-conditioned interleaved latent residual SID pooling.

Inference:
    raw SID_BEGIN hidden -> formal A beam
    each hard-A branch: latent thought B -> formal residual B
    each hard-AB branch: latent thought C -> formal residual C
    each hard-ABC branch: latent thought D -> formal residual D

For transition l (A->B, B->C, C->D):
    thought = current + L_l([current, E(hard previous SID)])
    next    = thought + R_l([thought, E(hard previous SID)])

Thus each latent thought sees the actual previous SID selected on that beam
branch. Training uses the identical graph with teacher-forced previous SIDs.
"""

from __future__ import annotations

from typing import Iterable, Optional, Set, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.models.qwen3 import (
    Qwen3ForCausalLM,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
)
from vllm.model_executor.pooling_metadata import (
    PoolingMetadata,
    PoolingTensors,
)
from vllm.sequence import (
    PoolerOutput,
    PoolingSequenceGroupOutput,
)


NUM_SID_LAYERS = 4


class ResidualSIDBlockV085(nn.Module):
    """Training-compatible residual SID transition."""

    def __init__(
        self,
        hidden_size: int,
        dropout: float,
    ) -> None:
        super().__init__()

        # Keep parameter names identical to training.
        self.linear = nn.Linear(
            hidden_size * 2,
            hidden_size,
        )
        self.layer_norm = nn.LayerNorm(
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
        if (
            anchor_hidden.shape
            != previous_token_embedding.shape
        ):
            raise ValueError(
                "Residual block input shapes do not match: "
                f"anchor={tuple(anchor_hidden.shape)}, "
                "embedding="
                f"{tuple(previous_token_embedding.shape)}"
            )

        fused = torch.cat(
            [
                anchor_hidden,
                previous_token_embedding,
            ],
            dim=-1,
        )

        return self.dropout(
            self.activation(
                self.layer_norm(
                    self.linear(fused)
                )
            )
        )



class ResidualSIDHierarchicalBeamPoolerV085(
    nn.Module
):
    """Branch-conditioned interleaved latent residual SID beam decoder."""

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

        # Temporary-logit memory control only.
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
                int(raw_per_layer_topk)
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

        # Dynamic personalized latent configuration.
        self.latent_reasoning_enabled = bool(
            getattr(
                config,
                "latent_reasoning_enabled",
                False,
            )
        )
        self.latent_reasoning_mode = str(
            getattr(
                config,
                "latent_reasoning_mode",
                "",
            )
        )
        self.latent_reasoning_num_steps = int(
            getattr(
                config,
                "latent_reasoning_num_steps",
                0,
            )
        )
        self.latent_reasoning_num_transitions = int(
            getattr(
                config,
                "latent_reasoning_num_transitions",
                self.latent_reasoning_num_steps,
            )
        )
        self.latent_reasoning_dropout = float(
            getattr(
                config,
                "latent_reasoning_dropout",
                0.0,
            )
        )
        self.latent_reasoning_conditioning = str(
            getattr(config, "latent_reasoning_conditioning", "")
        )
        self.latent_reasoning_update = str(
            getattr(config, "latent_reasoning_update", "")
        )

        self._validate_config()

        # Full unsharded SID output matrices exported for rank-0 pooling.
        self.sid_output_weights = (
            nn.ParameterList(
                [
                    nn.Parameter(
                        torch.empty(
                            layer_size,
                            self.hidden_size,
                        ),
                        requires_grad=False,
                    )
                    for layer_size
                    in self.layer_sizes
                ]
            )
        )

        # If embeddings are tied, output matrices are also input embeddings.
        # Otherwise the exporter supplies only A/B/C input slices because D is
        # terminal and has no latent/formal transition after it.
        if self.tie_word_embeddings:
            self.sid_input_embeddings = (
                nn.ParameterList()
            )
        else:
            self.sid_input_embeddings = (
                nn.ParameterList(
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
                        for transition_index
                        in range(
                            NUM_SID_LAYERS - 1
                        )
                    ]
                )
            )

        self.sid_residual_blocks = (
            nn.ModuleList(
                [
                    ResidualSIDBlockV085(
                        hidden_size=(
                            self.hidden_size
                        ),
                        dropout=(
                            self.dropout_probability
                        ),
                    )
                    for _ in range(
                        NUM_SID_LAYERS - 1
                    )
                ]
            )
        )

        self.latent_reasoning_blocks = (
            nn.ModuleList(
                [
                    ResidualSIDBlockV085(
                        hidden_size=(
                            self.hidden_size
                        ),
                        dropout=(
                            self.latent_reasoning_dropout
                        ),
                    )
                    for _ in range(
                        self.latent_reasoning_num_transitions
                    )
                ]
            )
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
        if (
            len(self.layer_starts)
            != NUM_SID_LAYERS
        ):
            raise ValueError(
                "Expected four residual SID "
                "layer starts, got "
                f"{self.layer_starts}"
            )

        if (
            len(self.layer_sizes)
            != NUM_SID_LAYERS
        ):
            raise ValueError(
                "Expected four residual SID "
                "layer sizes, got "
                f"{self.layer_sizes}"
            )

        if any(
            size <= 0
            for size in self.layer_sizes
        ):
            raise ValueError(
                "Every residual SID layer size "
                "must be positive: "
                f"{self.layer_sizes}"
            )

        if self.sid_begin_token_id < 0:
            raise ValueError(
                "residual_sid_begin_token_id "
                "is missing or invalid."
            )

        if self.beam_size <= 0:
            raise ValueError(
                "residual_sid_beam_size "
                "is missing or invalid."
            )

        if self.beam_size > min(
            self.layer_sizes
        ):
            raise ValueError(
                "beam_size must not exceed "
                "the smallest SID vocabulary: "
                f"beam={self.beam_size}, "
                f"layer_sizes={self.layer_sizes}"
            )

        if (
            len(self.per_layer_topk)
            != NUM_SID_LAYERS
        ):
            raise ValueError(
                "residual_sid_per_layer_topk "
                "must contain four values, got "
                f"{self.per_layer_topk}"
            )

        if any(
            value <= 0
            for value in self.per_layer_topk
        ):
            raise ValueError(
                "Every per-layer top-k value "
                "must be positive: "
                f"{self.per_layer_topk}"
            )

        if self.parent_chunk_size <= 0:
            raise ValueError(
                "residual_sid_parent_chunk_size "
                "must be positive."
            )

        if self.latent_reasoning_enabled:
            if (
                self.latent_reasoning_mode
                != "branch_conditioned_interleaved"
            ):
                raise ValueError(
                    "Expected latent_reasoning_mode="
                    "'branch_conditioned_interleaved', got "
                    f"{self.latent_reasoning_mode!r}"
                )

            if self.latent_reasoning_num_steps != NUM_SID_LAYERS - 1:
                raise ValueError(
                    "Branch-conditioned interleaved reasoning requires three "
                    "latent steps before B/C/D."
                )

            if (
                self.latent_reasoning_num_transitions
                != NUM_SID_LAYERS - 1
            ):
                raise ValueError(
                    "Four SID layers require exactly three latent transitions."
                )

            if self.latent_reasoning_conditioning != "hard_previous_sid":
                raise ValueError(
                    "latent_reasoning_conditioning must be 'hard_previous_sid'."
                )
            if self.latent_reasoning_update != "thought_then_formal_residual":
                raise ValueError(
                    "latent_reasoning_update must be "
                    "'thought_then_formal_residual'."
                )

    def _validate_prompt_end(
        self,
        pooling_metadata: PoolingMetadata,
    ) -> None:
        if not (
            self.validate_prompt_ends_with_sid_begin
        ):
            return

        invalid_sequences = []

        for (
            sequence_id,
            sequence_data,
        ) in pooling_metadata.seq_data.items():
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

            if (
                final_token_id
                != self.sid_begin_token_id
            ):
                invalid_sequences.append(
                    (
                        int(sequence_id),
                        final_token_id,
                    )
                )

        if invalid_sequences:
            raise ValueError(
                "Residual SID pooling requires "
                "every prompt to end with "
                "the sid_begin token.\n"
                "Expected token ID: "
                f"{self.sid_begin_token_id}\n"
                "Invalid sequence preview: "
                f"{invalid_sequences[:8]}"
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
        """Compute one SID layer's normalized Top-K."""
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

            logits = F.linear(
                hidden_chunk.to(
                    output_weight.dtype
                ),
                output_weight,
            )

            log_probs = F.log_softmax(
                logits.float(),
                dim=-1,
            )

            (
                top_log_probs,
                top_local_ids,
            ) = torch.topk(
                log_probs,
                k=child_topk,
                dim=-1,
                largest=True,
                sorted=True,
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
        current_hidden: torch.Tensor,
        current_sequences: torch.Tensor,
        current_scores: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Interleave branch-conditioned latent thought and formal residual."""
        previous_local_ids = current_sequences[:, :, -1]
        previous_embedding = self._get_previous_embedding(
            transition_index=layer_index - 1,
            previous_local_ids=previous_local_ids,
        )

        # Every parent has its own current_hidden and hard previous SID.  The
        # latent thought therefore sees the actual prefix represented by this
        # beam branch rather than a global soft lookahead distribution.
        latent_delta = self.latent_reasoning_blocks[
            layer_index - 1
        ](
            current_hidden,
            previous_embedding,
        )
        thought_hidden = current_hidden + latent_delta

        formal_delta = self.sid_residual_blocks[
            layer_index - 1
        ](
            thought_hidden,
            previous_embedding,
        )
        next_hidden_all_parents = thought_hidden + formal_delta

        child_topk = min(
            int(self.per_layer_topk[layer_index]),
            int(self.layer_sizes[layer_index]),
        )
        child_log_probs, child_local_ids = self._layer_log_probs_and_topk(
            layer_index=layer_index,
            hidden_states=next_hidden_all_parents,
            child_topk=child_topk,
        )

        total_child_scores = current_scores.unsqueeze(-1) + child_log_probs
        batch_size = int(total_child_scores.shape[0])
        flattened_scores = total_child_scores.reshape(batch_size, -1)
        next_scores, flattened_indices = torch.topk(
            flattened_scores,
            k=self.beam_size,
            dim=-1,
            largest=True,
            sorted=True,
        )
        parent_indices = flattened_indices // child_topk
        child_slots = flattened_indices % child_topk
        batch_indices = torch.arange(
            batch_size,
            device=current_hidden.device,
        ).unsqueeze(1)

        selected_child_local_ids = child_local_ids[
            batch_indices,
            parent_indices,
            child_slots,
        ]
        selected_parent_hidden = next_hidden_all_parents[
            batch_indices,
            parent_indices,
            :,
        ]
        selected_parent_sequences = current_sequences[
            batch_indices,
            parent_indices,
            :,
        ]
        next_sequences = torch.cat(
            [
                selected_parent_sequences,
                selected_child_local_ids.unsqueeze(-1),
            ],
            dim=-1,
        )
        return selected_parent_hidden, next_sequences, next_scores

    def decode(
        self,
        raw_anchor: torch.Tensor,
    ) -> torch.Tensor:
        """Hierarchical beam with a branch-conditioned thought before B/C/D."""
        if raw_anchor.ndim != 2:
            raise ValueError(
                "raw_anchor must have shape [batch, hidden]."
            )
        if raw_anchor.shape[-1] != self.hidden_size:
            raise ValueError(
                "Unexpected anchor hidden size: "
                f"{raw_anchor.shape[-1]} != {self.hidden_size}"
            )

        batch_size = int(raw_anchor.shape[0])

        # A is unchanged from the residual baseline.
        first_weight = self.sid_output_weights[0]
        first_logits = F.linear(
            raw_anchor.to(first_weight.dtype),
            first_weight,
        )
        first_log_probs = F.log_softmax(first_logits.float(), dim=-1)
        first_scores, first_local_ids = torch.topk(
            first_log_probs,
            k=self.beam_size,
            dim=-1,
            largest=True,
            sorted=True,
        )

        current_sequences = first_local_ids.unsqueeze(-1)
        current_hidden = raw_anchor.unsqueeze(1).expand(
            batch_size,
            self.beam_size,
            self.hidden_size,
        ).contiguous()
        current_scores = first_scores

        # After each hard SID selection, the next layer's thought is computed
        # separately for every surviving branch.
        for layer_index in range(1, NUM_SID_LAYERS):
            (
                current_hidden,
                current_sequences,
                current_scores,
            ) = self._expand_one_layer(
                layer_index=layer_index,
                current_hidden=current_hidden,
                current_sequences=current_sequences,
                current_scores=current_scores,
            )

        expected_shape = (
            batch_size,
            self.beam_size,
            NUM_SID_LAYERS,
        )
        if current_sequences.shape != expected_shape:
            raise RuntimeError(
                "Unexpected residual SID sequence shape: "
                f"{tuple(current_sequences.shape)} != {expected_shape}"
            )

        global_ids = current_sequences + self._layer_starts.to(
            device=current_sequences.device
        ).view(1, 1, NUM_SID_LAYERS)
        return torch.cat(
            [
                global_ids.to(dtype=torch.float32),
                current_scores.to(dtype=torch.float32).unsqueeze(-1),
            ],
            dim=-1,
        )

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

        if (
            int(
                last_token_flat_indices[-1]
            )
            >= int(
                hidden_states.shape[0]
            )
        ):
            raise RuntimeError(
                "Pooling metadata points beyond "
                "flattened hidden states: "
                "last_index="
                f"{int(last_token_flat_indices[-1])}, "
                "hidden_rows="
                f"{hidden_states.shape[0]}"
            )

        raw_anchor = (
            hidden_states.index_select(
                0,
                last_token_flat_indices,
            )
        )

        # Exact train/inference match. A is scored first. Then every surviving
        # hard branch performs one latent thought followed by one formal
        # residual transition before scoring B/C/D.
        decoded = self.decode(
            raw_anchor
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


def _canonicalize_block_suffix(
    suffix: str,
) -> str:
    """Map common block parameter aliases to training names."""
    aliases = {
        "linear.weight":
            "linear.weight",
        "linear.bias":
            "linear.bias",
        "proj.weight":
            "linear.weight",
        "proj.bias":
            "linear.bias",
        "projection.weight":
            "linear.weight",
        "projection.bias":
            "linear.bias",
        "dense.weight":
            "linear.weight",
        "dense.bias":
            "linear.bias",
        "fc.weight":
            "linear.weight",
        "fc.bias":
            "linear.bias",
        "0.weight":
            "linear.weight",
        "0.bias":
            "linear.bias",

        "layer_norm.weight":
            "layer_norm.weight",
        "layer_norm.bias":
            "layer_norm.bias",
        "norm.weight":
            "layer_norm.weight",
        "norm.bias":
            "layer_norm.bias",
        "layernorm.weight":
            "layer_norm.weight",
        "layernorm.bias":
            "layer_norm.bias",
        "ln.weight":
            "layer_norm.weight",
        "ln.bias":
            "layer_norm.bias",
        "1.weight":
            "layer_norm.weight",
        "1.bias":
            "layer_norm.bias",
    }

    return aliases.get(
        suffix,
        suffix,
    )


def _canonicalize_module_block_name(
    mapped: str,
    block_prefix: str,
) -> str:
    if not mapped.startswith(
        block_prefix
    ):
        return mapped

    remaining = mapped[
        len(block_prefix):
    ]

    if "." not in remaining:
        return mapped

    (
        block_index,
        suffix,
    ) = remaining.split(
        ".",
        1,
    )

    suffix = (
        _canonicalize_block_suffix(
            suffix
        )
    )

    return (
        block_prefix
        + block_index
        + "."
        + suffix
    )


def _canonicalize_pooler_weight_name(
    name: str,
    *,
    tie_word_embeddings: bool,
) -> Optional[str]:
    """Map training/exporter weights into ``_residual_pooler``."""

    mapped = name

    prefix_aliases = (
        # Dynamic latent reasoning blocks.
        (
            "model.latent_reasoning_blocks.",
            "_residual_pooler."
            "latent_reasoning_blocks.",
        ),
        (
            "latent_reasoning_blocks.",
            "_residual_pooler."
            "latent_reasoning_blocks.",
        ),
        (
            "pooler.latent_reasoning_blocks.",
            "_residual_pooler."
            "latent_reasoning_blocks.",
        ),

        # Residual SID blocks.
        (
            "model.sid_residual_blocks.",
            "_residual_pooler."
            "sid_residual_blocks.",
        ),
        (
            "sid_residual_blocks.",
            "_residual_pooler."
            "sid_residual_blocks.",
        ),
        (
            "pooler.sid_residual_blocks.",
            "_residual_pooler."
            "sid_residual_blocks.",
        ),
        (
            "pooler.residual_blocks.",
            "_residual_pooler."
            "sid_residual_blocks.",
        ),
        (
            "residual_sid_pooler."
            "sid_residual_blocks.",
            "_residual_pooler."
            "sid_residual_blocks.",
        ),

        # Exported complete classifier slices.
        (
            "pooler.sid_output_weights.",
            "_residual_pooler."
            "sid_output_weights.",
        ),
        (
            "pooler.sid_output_weight.",
            "_residual_pooler."
            "sid_output_weights.",
        ),
        (
            "sid_output_weights.",
            "_residual_pooler."
            "sid_output_weights.",
        ),
        (
            "pooler.sid_input_embeddings.",
            "_residual_pooler."
            "sid_input_embeddings.",
        ),
        (
            "pooler.sid_input_embedding.",
            "_residual_pooler."
            "sid_input_embeddings.",
        ),
        (
            "sid_input_embeddings.",
            "_residual_pooler."
            "sid_input_embeddings.",
        ),
        (
            "_residual_pooler.",
            "_residual_pooler.",
        ),
    )

    for (
        old_prefix,
        new_prefix,
    ) in prefix_aliases:
        if mapped.startswith(
            old_prefix
        ):
            mapped = (
                new_prefix
                + mapped[
                    len(old_prefix):
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
        # Tied models reuse the previous layer's output matrix.
        return None

    mapped = (
        _canonicalize_module_block_name(
            mapped,
            "_residual_pooler."
            "sid_residual_blocks.",
        )
    )
    mapped = (
        _canonicalize_module_block_name(
            mapped,
            "_residual_pooler."
            "latent_reasoning_blocks.",
        )
    )

    return mapped


def _is_custom_sidecar_weight(
    name: str,
) -> bool:
    prefixes = (
        "sid_residual_blocks.",
        "model.sid_residual_blocks.",
        "latent_reasoning_blocks.",
        "model.latent_reasoning_blocks.",
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
    """Qwen3 backbone + branch-conditioned interleaved latent SID pooling."""

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
            for (
                name,
                loaded_weight,
            ) in weights:
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
    """Normal AR Qwen3 loader that ignores custom residual/latent weights."""

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
            for (
                name,
                loaded_weight,
            ) in weights
            if not _is_custom_sidecar_weight(
                name
            )
        )

        return super().load_weights(
            filtered_weights
        )
