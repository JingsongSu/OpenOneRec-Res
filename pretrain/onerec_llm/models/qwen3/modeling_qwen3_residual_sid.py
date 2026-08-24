"""Qwen3 causal LM with branch-conditioned interleaved latent reasoning.

This variant interleaves one latent "thinking" transition with every formal
residual SID transition after SID-A.  The latent step is conditioned on the
*actual previous hard SID* and the current branch-specific hidden state.

Training (teacher forced):

    hA = backbone_hidden_at_sid_begin
    A  = ordinary LM target

    # Before predicting B, think with the gold hard A.
    tB = hA + L0([hA, E(A)])
    hB = tB + R0([tB, E(A)])
    B  = head_B(hB)

    # Before predicting C, think with the gold hard B.  hB already contains A.
    tC = hB + L1([hB, E(B)])
    hC = tC + R1([tC, E(B)])
    C  = head_C(hC)

    # Before predicting D, think with the gold hard C.  hC contains A/B.
    tD = hC + L2([hC, E(C)])
    hD = tD + R2([tD, E(C)])
    D  = head_D(hD)

Inference uses the exact same graph, except A/B/C are the hard SID values on
each surviving beam branch.  Consequently each latent step sees what that
branch actually generated before it reasons about the next SID layer.

There are three latent reasoning blocks for a four-layer A/B/C/D SID.  There
is no soft-SID lookahead, no latent token, no auxiliary latent CE, and no
inference-only score fusion.  The original formal A CE and residual B/C/D CE
are the only training objectives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast

from .modeling_qwen3 import Qwen3ForCausalLM


class ResidualSIDBlock(nn.Module):
    """Reference-compatible residual fusion block."""

    def __init__(self, hidden_size: int, dropout: float):
        super().__init__()
        self.linear = nn.Linear(hidden_size * 2, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        return self.dropout(
            self.activation(self.layer_norm(self.linear(fused)))
        )


@dataclass
class ResidualSIDCausalLMOutput(CausalLMOutputWithPast):
    """Causal-LM output plus residual-SID loss fields."""

    residual_sid_loss: Optional[torch.FloatTensor] = None
    residual_sid_token_losses: Optional[torch.FloatTensor] = None
    residual_sid_count: Optional[torch.LongTensor] = None

    # Kept only for compatibility with the training loop API.  Branch-
    # conditioned latent reasoning uses no separate auxiliary latent CE.
    latent_reasoning_loss: Optional[torch.FloatTensor] = None
    latent_reasoning_count: Optional[torch.LongTensor] = None


class Qwen3ForCausalLMResidualSID(Qwen3ForCausalLM):
    """Qwen3 with formal residual SID blocks and interleaved latent blocks."""

    def __init__(self, config):
        super().__init__(config)

        starts = tuple(
            int(x)
            for x in getattr(config, "residual_sid_layer_starts", ())
        )
        sizes = tuple(
            int(x)
            for x in getattr(config, "residual_sid_layer_sizes", ())
        )
        if len(starts) < 2 or len(starts) != len(sizes):
            raise ValueError(
                "config.residual_sid_layer_starts and "
                "config.residual_sid_layer_sizes must have equal length >= 2."
            )
        if any(size <= 0 for size in sizes):
            raise ValueError("Every residual SID layer size must be positive.")

        self.residual_sid_layer_starts: Tuple[int, ...] = starts
        self.residual_sid_layer_sizes: Tuple[int, ...] = sizes
        self.residual_sid_num_layers = len(starts)
        self.residual_sid_begin_token_id = int(
            getattr(config, "residual_sid_begin_token_id")
        )
        self.residual_sid_end_token_id = int(
            getattr(config, "residual_sid_end_token_id")
        )
        self.residual_sid_dropout = float(
            getattr(config, "residual_sid_dropout", 0.1)
        )

        # Formal residual transitions: A->B, B->C, C->D.
        self.sid_residual_blocks = nn.ModuleList(
            ResidualSIDBlock(config.hidden_size, self.residual_sid_dropout)
            for _ in range(self.residual_sid_num_layers - 1)
        )

        self.latent_reasoning_enabled = bool(
            getattr(config, "latent_reasoning_enabled", True)
        )
        self.latent_reasoning_mode = str(
            getattr(
                config,
                "latent_reasoning_mode",
                "branch_conditioned_interleaved",
            )
        )
        # Here num_steps means the number of interleaved latent thoughts,
        # namely one before B/C/D for four SID layers.
        self.latent_reasoning_num_steps = int(
            getattr(
                config,
                "latent_reasoning_num_steps",
                self.residual_sid_num_layers - 1,
            )
        )
        self.latent_reasoning_dropout = float(
            getattr(config, "latent_reasoning_dropout", 0.1)
        )
        self.latent_reasoning_loss_weight = float(
            getattr(config, "latent_reasoning_loss_weight", 0.0)
        )

        expected_steps = self.residual_sid_num_layers - 1
        if self.latent_reasoning_enabled:
            if self.latent_reasoning_mode != "branch_conditioned_interleaved":
                raise ValueError(
                    "latent_reasoning_mode must be "
                    "'branch_conditioned_interleaved'."
                )
            if self.latent_reasoning_num_steps != expected_steps:
                raise ValueError(
                    "Branch-conditioned interleaved reasoning requires one "
                    "latent step before every SID layer after A: "
                    f"steps={self.latent_reasoning_num_steps}, "
                    f"expected={expected_steps}."
                )
            if not 0.0 <= self.latent_reasoning_dropout < 1.0:
                raise ValueError(
                    "latent_reasoning_dropout must satisfy 0 <= dropout < 1."
                )
            if self.latent_reasoning_loss_weight != 0.0:
                raise ValueError(
                    "branch_conditioned_interleaved uses only the formal "
                    "A/B/C/D task losses; latent_reasoning_loss_weight must "
                    "remain 0.0."
                )

        # Three branch-conditioned thought blocks for B/C/D.
        self.latent_reasoning_blocks = nn.ModuleList(
            ResidualSIDBlock(
                config.hidden_size,
                self.latent_reasoning_dropout,
            )
            for _ in range(expected_steps)
        )

    def _restricted_logits(
        self,
        hidden: torch.Tensor,
        layer: int,
    ) -> torch.Tensor:
        start = self.residual_sid_layer_starts[layer]
        size = self.residual_sid_layer_sizes[layer]
        weight = self.lm_head.weight[start : start + size]
        return F.linear(hidden.to(weight.dtype), weight).float()

    def _validate_aux_inputs(
        self,
        anchor_indices: torch.LongTensor,
        target_global_ids: torch.LongTensor,
    ) -> None:
        if anchor_indices.ndim != 2 or anchor_indices.shape[-1] != 2:
            raise ValueError("anchor_indices must have shape [N, 2].")
        if (
            target_global_ids.ndim != 2
            or target_global_ids.shape[-1] != self.residual_sid_num_layers
        ):
            raise ValueError(
                "target_global_ids must have shape "
                f"[N, {self.residual_sid_num_layers}]."
            )
        if anchor_indices.shape[0] != target_global_ids.shape[0]:
            raise ValueError(
                "anchor_indices and target_global_ids must have the same N."
            )

    def _sid_layer_ce(
        self,
        hidden: torch.Tensor,
        target_global_ids: torch.LongTensor,
        layer: int,
    ) -> torch.Tensor:
        logits = self._restricted_logits(hidden, layer)
        local_target = (
            target_global_ids[:, layer].long()
            - self.residual_sid_layer_starts[layer]
        )
        layer_size = self.residual_sid_layer_sizes[layer]
        if torch.any(local_target < 0) or torch.any(local_target >= layer_size):
            raise ValueError(
                f"Out-of-range target for SID layer {layer}: "
                f"layer_size={layer_size}."
            )
        return F.cross_entropy(
            logits,
            local_target,
            reduction="none",
        )

    def _interleaved_transition(
        self,
        current: torch.Tensor,
        previous_embedding: torch.Tensor,
        transition_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Think on the current hard branch, then perform formal transition.

        Args:
            current: branch-specific state before predicting the next SID.
            previous_embedding: embedding of the actual previous hard SID.
            transition_index: 0/1/2 for A->B/B->C/C->D.

        Returns:
            thought: hidden state after the latent reasoning block.
            next_hidden: state used to predict the next formal SID.
        """
        fused_latent = torch.cat(
            [current, previous_embedding],
            dim=-1,
        )
        thought = (
            current
            + self.latent_reasoning_blocks[transition_index](fused_latent)
        )

        fused_formal = torch.cat(
            [thought, previous_embedding],
            dim=-1,
        )
        next_hidden = (
            thought
            + self.sid_residual_blocks[transition_index](fused_formal)
        )
        return thought, next_hidden

    def compute_residual_sid_loss(
        self,
        raw_anchor: torch.Tensor,
        target_global_ids: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Teacher-forced branch-conditioned loss for SID B/C/D.

        Training uses the gold previous SID at each step, exactly matching the
        inference graph where each surviving beam uses its own selected hard
        previous SID.  The only difference is teacher forcing versus beam
        selection.
        """
        if raw_anchor.ndim != 2:
            raise ValueError("raw_anchor must have shape [N, hidden].")
        if (
            target_global_ids.ndim != 2
            or target_global_ids.shape[-1] != self.residual_sid_num_layers
        ):
            raise ValueError(
                "target_global_ids must have shape "
                f"[N, {self.residual_sid_num_layers}]."
            )

        current = raw_anchor
        token_losses = []

        for layer in range(1, self.residual_sid_num_layers):
            previous_global_id = target_global_ids[:, layer - 1].long()
            previous_embedding = self.model.embed_tokens(previous_global_id)
            _, current = self._interleaved_transition(
                current=current,
                previous_embedding=previous_embedding,
                transition_index=layer - 1,
            )
            layer_loss = self._sid_layer_ce(
                hidden=current,
                target_global_ids=target_global_ids,
                layer=layer,
            )
            token_losses.append(layer_loss)

        stacked = torch.stack(token_losses, dim=-1)
        return stacked.mean(), stacked

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        return_dict: Optional[bool] = None,
        residual_sid_anchor_indices: Optional[torch.LongTensor] = None,
        residual_sid_target_global_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> ResidualSIDCausalLMOutput:
        return_dict = (
            return_dict
            if return_dict is not None
            else self.config.use_return_dict
        )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            return_dict=True,
            **kwargs,
        )
        last_hidden_state = outputs.last_hidden_state

        if isinstance(logits_to_keep, int):
            hidden_slice = (
                slice(-logits_to_keep, None)
                if logits_to_keep > 0
                else slice(None)
            )
        else:
            hidden_slice = logits_to_keep

        if self.chunked_loss_computer:
            logits = last_hidden_state[:, hidden_slice, :]
        else:
            logits = self.lm_head(last_hidden_state[:, hidden_slice, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
            )

        residual_sid_loss = None
        residual_sid_token_losses = None
        residual_sid_count = None

        if (
            residual_sid_anchor_indices is not None
            and residual_sid_target_global_ids is not None
            and residual_sid_anchor_indices.numel() > 0
        ):
            self._validate_aux_inputs(
                residual_sid_anchor_indices,
                residual_sid_target_global_ids,
            )
            batch_index = residual_sid_anchor_indices[:, 0].long()
            sequence_index = residual_sid_anchor_indices[:, 1].long()
            raw_anchor = last_hidden_state[batch_index, sequence_index]

            (
                residual_sid_loss,
                residual_sid_token_losses,
            ) = self.compute_residual_sid_loss(
                raw_anchor=raw_anchor,
                target_global_ids=residual_sid_target_global_ids,
            )
            residual_sid_count = torch.tensor(
                residual_sid_token_losses.numel(),
                dtype=torch.long,
                device=last_hidden_state.device,
            )

        if not return_dict:
            base = (
                logits,
                outputs.past_key_values,
                outputs.hidden_states,
                outputs.attentions,
            )
            return ((loss,) + base) if loss is not None else base

        return ResidualSIDCausalLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            residual_sid_loss=residual_sid_loss,
            residual_sid_token_losses=residual_sid_token_losses,
            residual_sid_count=residual_sid_count,
            latent_reasoning_loss=None,
            latent_reasoning_count=None,
        )
