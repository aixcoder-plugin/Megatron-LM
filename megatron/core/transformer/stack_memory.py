"""StackMemory module for StackTrans/FANformer-style memory.

This is a Megatron-friendly implementation:
- Parameters are *replicated* across tensor-parallel ranks (no TP sharding).
- Gradients are synchronized across TP via Megatron's finalize_model_grads:
  - when sequence_parallel=True: SUM grads across TP (param.sequence_parallel=True)
  - otherwise: AVG grads across TP (param.average_gradients_across_tp_domain=True)

Training path only (no inference cache/step in v1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor
import torch.nn as nn

from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig


@dataclass
class StackMemoryState:
    """Per-forward, per-stage StackMemory state threaded through layers."""

    stack: Optional[Tensor] = None
    mask: Optional[Tensor] = None


class StackMemory(MegatronModule):
    """Vectorized stack memory (token-wise; depth-state across layers).

    Expected shapes (Megatron layout):
    - hidden_states: [s, b, h]
    - memory_stack:  [s, b, n_heads, stack_slots, stack_dim]
    - memory_mask:   [s, b, n_heads, stack_slots] (float mask/probabilities)
    """

    def __init__(self, config: TransformerConfig):
        super().__init__(config=config)
        self.config: TransformerConfig = config

        self.num_mem_heads = self.config.stack_memory_num_heads
        self.stack_slots = self.config.stack_memory_slots

        # 承接主模型的 HiddenStates
        self.head_dim = self.config.hidden_size // self.num_mem_heads
        self.stack_dim = self.config.stack_memory_dim
        self.use_compression = self.stack_dim != self.head_dim

        # Action prediction head: logits for push/pop/noop for each head.
        self.action_head = nn.Linear(
            self.config.hidden_size,
            3 * self.num_mem_heads,
            bias=self.config.add_bias_linear,
        )

        # Slot gating projection for attention over stack slots.
        self.gate_proj = nn.Linear(
            self.stack_dim,
            1,
            bias=self.config.add_bias_linear,
        )

        # Optional compression (per-head).
        if self.use_compression:
            self.down_proj = nn.Linear(
                self.head_dim,
                self.stack_dim,
                bias=self.config.add_bias_linear,
            )
            self.up_proj = nn.Linear(
                self.stack_dim,
                self.head_dim,
                bias=self.config.add_bias_linear,
            )
        else:
            self.down_proj = None
            self.up_proj = None

        # Residual scaling.
        self.res_weight = nn.Parameter(torch.ones(1))

        self._mark_parameters_for_tp_grad_sync()

    def _mark_parameters_for_tp_grad_sync(self) -> None:
        """因为参数太小没用张量并行，所以要在不同的TP rank上复制一份权重，并保持更新同步"""

        # If TP=1, nothing to do.
        if self.config.tensor_model_parallel_size <= 1:
            return

        if self.config.sequence_parallel:
            # Each TP rank sees different tokens -> need SUM across TP.
            for p in self.parameters():
                setattr(p, "sequence_parallel", True)
        else:
            # Each TP rank sees identical tokens -> AVG to keep params identical.
            for p in self.parameters():
                setattr(p, "average_gradients_across_tp_domain", True)

    def init_state(self, hidden_states: Tensor) -> Tuple[Tensor, Tensor]:
        """Initialize (stack, mask) for the given hidden_states shape/device."""
        seq_len, batch_size, _ = hidden_states.shape

        stack = torch.zeros(
            (seq_len, batch_size, self.num_mem_heads, self.stack_slots, self.stack_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        # Keep mask in fp32 for stability; values are in [0,1].
        mask = torch.zeros(
            (seq_len, batch_size, self.num_mem_heads, self.stack_slots),
            dtype=torch.float32,
            device=hidden_states.device,
        )
        return stack, mask

    @staticmethod
    def _vectorized_update(
        stack: Tensor, mask: Tensor, actions: Tensor, k_values: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Vectorized push/pop/noop update over stack slots.

        Args:
            stack:   [s, b, hds, slots, dim], Current stack state
            mask:    [s, b, hds, slots], Current stack mask
            actions: [s, b, hds, 3], Action probabilities (push, pop, noop), should sum to 1
            k_values:[s, b, hds, dim], New values to potentially push
        """
        # Push: insert at slot 0, shift others down (drop last).
        push_stack = torch.cat([k_values.unsqueeze(-2), stack[..., :-1, :]], dim=-2)
        push_mask = torch.cat([torch.ones_like(mask[..., :1]), mask[..., :-1]], dim=-1)

        # Pop: remove slot 0, shift others up, zero-fill last.
        pop_stack = torch.cat([stack[..., 1:, :], torch.zeros_like(stack[..., :1, :])], dim=-2)
        pop_mask = torch.cat([mask[..., 1:], torch.zeros_like(mask[..., :1])], dim=-1)

        push_w = actions[..., 0]
        pop_w = actions[..., 1]
        noop_w = actions[..., 2]

        # Weighted sum (avoid stacking 3 big tensors).
        push_w_stack = push_w.unsqueeze(-1).unsqueeze(-1)  # [s,b,hds,1,1]
        pop_w_stack = pop_w.unsqueeze(-1).unsqueeze(-1)
        noop_w_stack = noop_w.unsqueeze(-1).unsqueeze(-1)

        new_stack = push_w_stack * push_stack + pop_w_stack * pop_stack + noop_w_stack * stack

        push_w_mask = push_w.unsqueeze(-1)  # [s,b,hds,1]
        pop_w_mask = pop_w.unsqueeze(-1)
        noop_w_mask = noop_w.unsqueeze(-1)

        new_mask = push_w_mask * push_mask + pop_w_mask * pop_mask + noop_w_mask * mask
        return new_stack, new_mask

    def forward(
        self,
        hidden_states: Tensor,
        memory_stack: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass.

        Args:
            hidden_states: [s, b, h]
            memory_stack:  [s, b, n_heads, slots, stack_dim]
            memory_mask:   [s, b, n_heads, slots]
        """
        seq_len, batch_size, _ = hidden_states.shape

        if memory_stack is None or memory_mask is None:
            memory_stack, memory_mask = self.init_state(hidden_states)

        action_logits = self.action_head(hidden_states) / math.sqrt(self.head_dim)

        # float32 for stability; 
        actions = torch.softmax(
            action_logits.reshape(seq_len, batch_size, self.num_mem_heads, 3),
            dim=-1,
            dtype=torch.float32,
        ).to(dtype=hidden_states.dtype)

        # TODO: splited chunks should be compatible with MultiheadAttention?
        k_values = hidden_states.reshape(seq_len, batch_size, self.num_mem_heads, self.head_dim)
        if self.use_compression:
            assert self.down_proj is not None
            k_values = self.down_proj(k_values)  # [s,b,hds,stack_dim]
        
        new_stack, new_mask = self._vectorized_update(
            memory_stack, memory_mask, actions, k_values
        ) # [s, b, hds, slots, dim], [s, b, hds, slots]

        gate_scores = self.gate_proj(new_stack).squeeze(-1)  # [s,b,hds,slots]
        gate_logits = gate_scores.float() + (1.0 - new_mask) * (-1.0e9)
        gate_weights = torch.softmax(gate_logits, dim=-1, dtype=torch.float32).to(
            dtype=new_stack.dtype
        )

        # Aggregate memory output: sum over slots.
        memory_output = (new_stack * gate_weights.unsqueeze(-1)).sum(dim=-2)  # [s,b,hds,dim]

        # Decompress if needed and merge heads.
        if self.use_compression:
            assert self.up_proj is not None
            memory_output = self.up_proj(memory_output)  # [s,b,hds,head_dim]

        memory_output = memory_output.reshape(seq_len, batch_size, -1)  # [s,b,h]

        # Residual connection.
        output = hidden_states + memory_output * self.res_weight

        return output, new_stack, new_mask

