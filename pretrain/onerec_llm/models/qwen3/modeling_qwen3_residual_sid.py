"""Qwen3 causal LM with trainable residual transitions for hierarchical SID tokens.

The backbone still predicts the first SID layer from the hidden state at
``<|sid_begin|>``.  Later SID layers are predicted by lightweight residual
blocks:

    h_0 = backbone_hidden_at_sid_begin
    h_l = h_{l-1} + R_{l-1}([h_0, E(target_{l-1})])

During SFT the previous SID token uses teacher forcing.  During inference the
first layer is Top-B once, while every later layer is greedy Top-1 for each of
those B paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

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
    """Causal-LM output plus the auxiliary residual-SID loss."""

    residual_sid_loss: Optional[torch.FloatTensor] = None
    residual_sid_token_losses: Optional[torch.FloatTensor] = None
    residual_sid_count: Optional[torch.LongTensor] = None


class Qwen3ForCausalLMResidualSID(Qwen3ForCausalLM):
    """Qwen3 whose SFT checkpoint contains the residual SID blocks."""

    def __init__(self, config):
        super().__init__(config)

        starts = tuple(
            int(x) for x in getattr(config, "residual_sid_layer_starts", ())
        )
        sizes = tuple(
            int(x) for x in getattr(config, "residual_sid_layer_sizes", ())
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

        self.sid_residual_blocks = nn.ModuleList(
            ResidualSIDBlock(config.hidden_size, self.residual_sid_dropout)
            for _ in range(self.residual_sid_num_layers - 1)
        )

    def _restricted_logits(
        self,
        hidden: torch.Tensor,
        layer: int,
    ) -> torch.Tensor:
        start = self.residual_sid_layer_starts[layer]
        size = self.residual_sid_layer_sizes[layer]
        weight = self.lm_head.weight[start : start + size]
        # Large matrix multiply uses the model parameter dtype; CE is FP32.
        return F.linear(hidden.to(weight.dtype), weight).float()

    def compute_residual_sid_loss(
        self,
        last_hidden_state: torch.Tensor,
        anchor_indices: torch.LongTensor,
        target_global_ids: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute teacher-forced residual loss for SID layers 1..L-1.

        Args:
            last_hidden_state: [batch, sequence, hidden].
            anchor_indices: [num_sid, 2], columns are batch index and the
                position of ``<|sid_begin|>``.
            target_global_ids: [num_sid, num_layers], containing a/b/c...
                global vocabulary IDs.
        """
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

        batch_index = anchor_indices[:, 0].long()
        sequence_index = anchor_indices[:, 1].long()
        anchor = last_hidden_state[batch_index, sequence_index]
        current = anchor
        token_losses = []

        for layer in range(1, self.residual_sid_num_layers):
            previous_global_id = target_global_ids[:, layer - 1].long()
            previous_embedding = self.model.embed_tokens(previous_global_id)
            fused = torch.cat([anchor, previous_embedding], dim=-1)
            current = current + self.sid_residual_blocks[layer - 1](fused)

            logits = self._restricted_logits(current, layer)
            local_target = (
                target_global_ids[:, layer].long()
                - self.residual_sid_layer_starts[layer]
            )
            layer_loss = F.cross_entropy(
                logits,
                local_target,
                reduction="none",
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
            # Retains the original repository's chunked-loss contract.
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
            residual_sid_loss, residual_sid_token_losses = (
                self.compute_residual_sid_loss(
                    last_hidden_state=last_hidden_state,
                    anchor_indices=residual_sid_anchor_indices,
                    target_global_ids=residual_sid_target_global_ids,
                )
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
        )
