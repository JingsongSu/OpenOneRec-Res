import torch
import torch.nn as nn
import torch.nn.functional as F
from onerec_llm.utils.time_tracker import TimeTracker

# ===================================================================
# Cross-Entropy Loss Function
# ===================================================================

class CrossEntropyLoss(nn.Module):
    """
    An efficient CrossEntropyLoss module that avoids redundant calculations.
    It first computes per-token losses and then manually applies the reduction.
    (Based on the user-provided, superior implementation).
    """
    def __init__(self,
                 ignore_index: int = -100,
                 return_token_loss: bool = False,
                 shift_labels: bool = True,
                 reduction: str = "mean"):
        super().__init__()
        self.ignore_index = ignore_index
        self.return_token_loss = return_token_loss
        self.reduction = reduction
        self.shift_labels = shift_labels

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        """
        Args:
            logits (torch.Tensor): A single tensor of shape (..., vocab_size).
            labels (torch.Tensor): Ground truth labels.
        """
        vocab_size = logits.shape[-1]
        
        if self.shift_labels:
          logits = logits[:, :-1, :]
          labels = labels[:, 1:]

        # Reshape for cross-entropy calculation
        logits_flat = logits.float().reshape(-1, vocab_size)
        labels_flat = labels.reshape(-1)

        # Step 1: Compute per-token loss. This is the base for all other calculations.
        per_token_loss = F.cross_entropy(
            logits_flat,
            labels_flat,
            ignore_index=self.ignore_index,
            reduction="none"
        )
        
        # Step 2: Manually apply reduction to get the final loss.
        loss = per_token_loss.sum()
        if self.reduction == "mean":
            # Ensure we divide by the number of valid (non-ignored) tokens
            total_elements = (labels_flat != self.ignore_index).sum()
            if total_elements > 0:
                loss /= total_elements
            else: # Handle case where all tokens are ignored
                loss.zero_()

        # Return what's requested
        if self.return_token_loss:
            return loss, per_token_loss
        
        return loss


# ===================================================================
# Memory-Efficient Chunked Loss Computer
# ===================================================================

class ChunkedLossComputer:
    """
    Memory-efficient chunked loss computer for solving OOM issues caused by large lm_head in LLMs.

    By computing the input sequence in chunks and manually accumulating gradients,
    it avoids allocating huge intermediate tensors for the entire sequence at once.

    Note: The returned loss has already been backpropagated and detached,
    and cannot be used for operations requiring gradients.
    """
    def __init__(self, lm_head: nn.Module, loss_fn: nn.Module, minibatch_size: int, shift_labels: bool = True):
        """
        Initialize the chunked loss computer.

        Args:
            lm_head: The output layer of the language model (typically nn.Linear)
            loss_fn: Loss function, must return (avg_loss, per_token_loss) tuple
            minibatch_size: Size of each chunk, used to control memory usage
            shift_labels: Whether to shift labels (for autoregressive models)
        """
        if not isinstance(lm_head, nn.Module) or not isinstance(loss_fn, nn.Module):
            raise TypeError("lm_head and loss_fn must be instances of nn.Module")
            
        self.lm_head = lm_head
        self.loss_fn = loss_fn
        self.minibatch_size = minibatch_size
        self.shift_labels = shift_labels
        self.loss_info = {}
        self.ticker = TimeTracker()

    def forward_and_backward(self, input: torch.Tensor, labels: torch.Tensor, loss_fn_args: dict = {}):
        """
        Execute chunked forward and backward propagation.

        Args:
            input: Input tensor with shape [batch_size, seq_len, hidden_dim]
            labels: Label tensor with shape [batch_size, seq_len]
            loss_fn_args: Additional arguments passed to the loss function

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (final_avg_loss, per_token_loss)

        Note: The returned loss has already been backpropagated and detached,
        and cannot be used for operations requiring gradients.
        """
        self.ticker.tick("lm_head")
        params = list(self.lm_head.parameters())
        grad_accs = [torch.zeros_like(p) for p in params]
        grad_input_full = torch.zeros_like(input)

        total_loss_sum_for_reporting = torch.tensor(0.0, device=input.device)
        all_per_token_losses = []

        seq_len = input.size(1)
        
        # Calculate total number of valid elements
        labels_to_count = labels[:, 1:] if self.shift_labels else labels
        total_elements = (labels_to_count != getattr(self.loss_fn, 'ignore_index', -100)).sum()
        
        if total_elements.item() == 0:
            return torch.tensor(0.0, device=input.device), None

        # Chunked forward and gradient accumulation
        for i in range(0, seq_len, self.minibatch_size):
            start, end = i, min(i + self.minibatch_size, seq_len)
            input_chunk = input[:, start:end, :].detach().requires_grad_()
            
            logits_chunk = self.lm_head(input_chunk)

            if self.shift_labels:
                label_start, label_end = start + 1, end + 1
                labels_chunk = labels[:, label_start:label_end]
                # Ensure logits and labels have matching lengths
                if logits_chunk.size(1) > labels_chunk.size(1):
                    logits_chunk = logits_chunk[:, :labels_chunk.size(1), :]
            else:
                labels_chunk = labels[:, start:end]

            if labels_chunk.numel() == 0:
                continue

            logits_flat = logits_chunk.reshape(-1, self.lm_head.out_features)
            labels_flat = labels_chunk.reshape(-1)
            
            # Compute loss
            loss_chunk_avg, per_token_loss_chunk = self.loss_fn(logits_flat, labels_flat, **loss_fn_args)

            # Convert to sum loss for backward propagation
            valid_tokens_in_chunk = (labels_flat != getattr(self.loss_fn, 'ignore_index', -100)).sum()
            
            if valid_tokens_in_chunk.item() == 0:
                all_per_token_losses.append(per_token_loss_chunk.detach())
                continue
            
            loss_chunk_sum = loss_chunk_avg * valid_tokens_in_chunk

            # Manually compute gradients and accumulate
            tensors_to_grad = [p for p in params if p.requires_grad] + [input_chunk]
            grads = torch.autograd.grad(outputs=loss_chunk_sum, inputs=tensors_to_grad, retain_graph=False)
        
            grad_idx = 0
            for j in range(len(params)):
                if params[j].requires_grad:
                    grad_accs[j] += grads[grad_idx]
                    grad_idx += 1
            grad_input_full[:, start:end, :] = grads[grad_idx]

            total_loss_sum_for_reporting += loss_chunk_sum.detach()
            all_per_token_losses.append(per_token_loss_chunk.detach())
        
        # Apply accumulated gradients
        for j, p in enumerate(params):
            if p.requires_grad:
                p.grad = grad_accs[j] / total_elements

        self.ticker.tick("llm")        
        input.backward(gradient=grad_input_full / total_elements)
        self.ticker.tick("done")
        
        final_avg_loss = (total_loss_sum_for_reporting / total_elements).detach()
        per_token_loss = torch.cat(all_per_token_losses) if all_per_token_losses else None
        final_avg_loss.requires_grad = True

        self.loss_info = {
            'loss': final_avg_loss,
            'per_token_loss': per_token_loss
        }
        return final_avg_loss, per_token_loss
