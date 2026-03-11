from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import NoReturn, Optional, Tuple, Union

import torch
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.tensor_parallel.mappings import (
    gather_from_tensor_model_parallel_region,
    scatter_to_sequence_parallel_region,
)
from megatron.core.utils import (
    divide,
    nvtx_range_pop,
    nvtx_range_push,
    get_tensor_model_parallel_group_if_none,
    get_pg_size,
)
from megatron.core.transformer.identity_op import IdentityOp

from .transformer_config import TransformerConfig

from megatron.training.utils import print_rank_0

from megatron.core.dist_checkpointing.mapping import (
    ReplicaId,
    ShardedStateDict,
)


@dataclass
class FanSubmodules:
    input_layernorm: Union[ModuleSpec, type] = IdentityOp
    linear_fc1: Union[ModuleSpec, type] = None
    activation_func: Union[ModuleSpec, type] = None
    linear_fc2: Union[ModuleSpec, type] = None


class FanQKVLinear(MegatronModule):
    """FAN-based projection to replace attention `linear_qkv`.

    This module is designed to be drop-in compatible with `build_module(...)`
    calls in `SelfAttention`, i.e. it accepts the same keyword arguments
    typically passed to ColumnParallelLinear/TEColumnParallelLinear.

    Notes:
    - It performs an internal FAN feature transform (cos/sin + activation),
      then projects to the requested `output_size`.
    - To keep the FAN split deterministic, the intermediate projection is
      gathered across TP ranks.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: TransformerConfig,
        init_method,
        bias: bool = True,
        gather_output: bool = False,
        skip_bias_add: bool = False,
        is_expert: bool = False,
        tp_comm_buffer_name: Optional[str] = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        submodules: Optional[FanSubmodules] = None,
        p_ratio: Optional[float] = None,
        use_p_bias: Optional[bool] = None,
        **kwargs,
    ):
        super().__init__(config=config)
        self.config: TransformerConfig = config

        # Keep compatibility with TE helpers (e.g., set_save_original_input()).
        self._save_original_input = False

        self.input_size = input_size
        self.output_size = output_size

        # compatibility with multi-query attention && group-query attention
        self.query_projection_size = self.config.kv_channels * self.config.num_attention_heads
        self.kv_projection_size = self.config.kv_channels * self.config.num_query_groups

        self.p_ratio = (
            p_ratio if p_ratio is not None else getattr(self.config, "fan_p_ratio", 0.25)
        )
        self.use_p_bias = (
            use_p_bias
            if use_p_bias is not None
            else getattr(self.config, "fan_use_p_bias", True)
        )

        assert 0.0 <= self.p_ratio <= 0.5, "p_ratio must be between 0 and 0.5"

        self.fan_no_compress = getattr(self.config, "fan_no_compress", False)


        p_output_size = int(self.input_size * self.p_ratio)
        g_output_size = self.input_size - p_output_size * 2
        self.fused_dims = (p_output_size, g_output_size)

        if self.fan_no_compress:
            assert p_output_size == self.input_size // 2, "p_ratio must be 0.5 for fan_no_compress"

        # We gather the intermediate so we can split on full (p, g) dims safely.
        fc1_out_size = p_output_size + g_output_size  # = input_size - p_output_size

        tp_group = get_tensor_model_parallel_group_if_none(tp_group, is_expert=is_expert)
        self.tp_group = tp_group
        self.world_size = get_pg_size(self.tp_group)

        self.hidden_size_per_attention_head = divide(
            self.query_projection_size, self.config.num_attention_heads
        )
        self.num_attention_heads_per_partition = divide(self.config.num_attention_heads, self.world_size)
        self.num_query_groups_per_partition = divide(self.config.num_query_groups, self.world_size)

        assert submodules is not None, "FanQKVLinear requires `submodules` to be provided."
        assert (
            submodules.linear_fc1 is not None and submodules.linear_fc2 is not None
        ), "FanQKVLinear requires `linear_fc1` and `linear_fc2` submodules."

        self.input_layernorm = build_module(
            submodules.input_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        if self.fan_no_compress:
            # No fc1 compression; sin/cos applied directly on hidden_states.
            # fc2 input dimension doubles (cos + sin).
            self.linear_fc1 = None
            fc2_input_size = 2 * self.input_size
        else:
            fc2_input_size = self.input_size
            self.linear_fc1 = build_module(
                submodules.linear_fc1,
                self.input_size,
                fc1_out_size,
                config=self.config,
                init_method=init_method,
                gather_output=False,
                bias=self.use_p_bias,
                skip_bias_add=False,
                is_expert=is_expert,
                tp_comm_buffer_name=(
                    f"{tp_comm_buffer_name}_fan_fc1" if tp_comm_buffer_name else "fan_qkv_fc1"
                ),
                tp_group=tp_group,
            )

        self.activation_func = None
        if self.config.use_te_activation_func and submodules.activation_func is not None:
            self.activation_func = build_module(submodules.activation_func, config=self.config)
        elif submodules.activation_func is not None:
            self.activation_func = self.config.activation_func
        
        if self.activation_func is not None:
            print_rank_0("WARNING: FanQKVLinear: activation_func is not None")

        print_rank_0(f"FanLinear: p_ratio: {self.p_ratio}, fused_dims: {self.fused_dims}, input_size: {self.input_size}, output_size: {self.output_size}, activation_func: {self.activation_func}, fan_enable_qk_fan: {self.config.fan_enable_qk_fan}, fan_use_p_bias: {self.use_p_bias}, input_layernorm: {self.input_layernorm}, fan_no_compress: {self.fan_no_compress}")

        self.linear_fc2 = None
        self.linear_fc2_q = None
        self.linear_fc2_k = None
        self.linear_fc2_v = None
        if self.config.fan_enable_qk_fan:
            assert skip_bias_add is False, "skip_bias_add must be False for FAN-QKV"
            self.linear_fc2_q = build_module(
                submodules.linear_fc2,
                fc2_input_size,
                self.query_projection_size,
                config=self.config,
                init_method=init_method,
                gather_output=gather_output,
                bias=bias,
                skip_bias_add=skip_bias_add,
                is_expert=is_expert,
                tp_comm_buffer_name=f"{tp_comm_buffer_name}_fan_fc2_q",
                tp_group=tp_group,
            )
            self.linear_fc2_k = build_module(
                submodules.linear_fc2,
                fc2_input_size,
                self.kv_projection_size,
                config=self.config,
                init_method=init_method,
                gather_output=gather_output,
                bias=bias,
                skip_bias_add=skip_bias_add,
                is_expert=is_expert,
                tp_comm_buffer_name=f"{tp_comm_buffer_name}_fan_fc2_k",
                tp_group=tp_group,
            )
            self.linear_fc2_v = build_module(
                submodules.linear_fc2,
                self.input_size,
                self.kv_projection_size,
                config=self.config,
                init_method=init_method,
                gather_output=gather_output,
                bias=bias,
                skip_bias_add=skip_bias_add,
                is_expert=is_expert,
                tp_comm_buffer_name=f"{tp_comm_buffer_name}_fan_fc2_v",
                tp_group=tp_group,
            )

            if getattr(self.config, "sequence_parallel", False) and not self.fan_no_compress:
                self.__set_disable_sequence_parallel(self.linear_fc2_q)
                self.__set_disable_sequence_parallel(self.linear_fc2_k)
                # NOTE: Keep sequence-parallel semantics for V.
                # - Q/K consume `fan_hidden`, which is already full-seq (linear_fc1 does SP all-gather),
                #   so we must disable SP to avoid gathering again.
                # - V consumes the original `hidden_states`, which may still be sequence-parallel
                #   (sharded along dim0), so we keep SP enabled on linear_fc2_v to all-gather.
                # When fan_no_compress is True, fan_hidden is still SP-sharded (no fc1 all-gather),
                # so Q/K also need SP to all-gather the seq dim.

        else:
            # Final projection keeps the original `linear_qkv` parallel semantics.
            self.linear_fc2 = build_module(
                submodules.linear_fc2,
                fc2_input_size,
                self.output_size,
                config=self.config,
                init_method=init_method,
                gather_output=gather_output,
                bias=bias,
                skip_bias_add=skip_bias_add,
                is_expert=is_expert,
                tp_comm_buffer_name=tp_comm_buffer_name,
                tp_group=tp_group,
            )

            if getattr(self.config, "sequence_parallel", False) and not self.fan_no_compress:
                self.__set_disable_sequence_parallel(self.linear_fc2)

        # Propagate save_original_input to inner TE modules if needed.
        if self._save_original_input:
            self.save_original_input = True
    
    def __set_disable_sequence_parallel(self, module: MegatronModule):
        # must not apply sequence-parallel all-gather again.
        if hasattr(module, "sequence_parallel"):
            module.sequence_parallel = False
        if hasattr(module, "allreduce_dgrad"):
            world_size = get_pg_size(self.tp_group)
            disable_grad_reduce = bool(getattr(module, "disable_grad_reduce", False))
            module.allreduce_dgrad = (world_size > 1) and (not disable_grad_reduce)
        for attr in (
            "ub_overlap_rs_fprop",
            "ub_overlap_ag_dgrad",
            "ub_overlap_ag_fprop",
            "ub_overlap_rs_dgrad",
            "ub_bulk_dgrad",
            "ub_bulk_wgrad",
        ):
            if hasattr(module, attr):
                setattr(module, attr, False)

    @property
    def save_original_input(self) -> bool:
        return bool(self._save_original_input)

    @save_original_input.setter
    def save_original_input(self, value: bool) -> None:
        self._save_original_input = bool(value)
        for m in (getattr(self, "linear_fc1", None), getattr(self, "linear_fc2", None)):
            if m is not None and hasattr(m, "save_original_input"):
                m.save_original_input = bool(value)

    def _fan_transform(self, hidden_states: torch.Tensor):
        """Core FAN feature transform shared by QKV and Norm paths.

        Returns:
            hidden_states: the (possibly layernorm'd) input, needed by V projection.
            fan_hidden: FAN features.
                - compress mode: [cos(p), sin(p), g], same hidden size as input.
                - no-compress mode: [cos(h), sin(h)], 2× hidden size.
        """
        nvtx_range_push(suffix="fan_input_layernorm")
        hidden_states = self.input_layernorm(hidden_states)
        nvtx_range_pop(suffix="fan_input_layernorm")

        if self.fan_no_compress:
            # No fc1 compression: apply sin/cos directly on hidden_states.
            # fan_hidden has 2× hidden_size.
            nvtx_range_push(suffix="fan_no_compress_sincos")
            fan_hidden = torch.cat(
                (torch.cos(hidden_states), torch.sin(hidden_states)), dim=-1
            )
            nvtx_range_pop(suffix="fan_no_compress_sincos")
        else:
            nvtx_range_push(suffix="fan_fc1")
            # [sq, b, h] --> [sq, b, (p_output_size + g_output_size)], h = p_output_size * 2 + g_output_size
            pg, _ = self.linear_fc1(hidden_states)
            pg = gather_from_tensor_model_parallel_region(pg, group=self.tp_group)
            nvtx_range_pop(suffix="fan_fc1")

            nvtx_range_push(suffix="fan_activation")
            p, g = pg.split(self.fused_dims, dim=-1)
            if self.activation_func is not None:
                fan_hidden = torch.cat((torch.cos(p), torch.sin(p), self.activation_func(g)), dim=-1)
            else:
                fan_hidden = torch.cat((torch.cos(p), torch.sin(p), g), dim=-1)
            nvtx_range_pop(suffix="fan_activation")

        return hidden_states, fan_hidden

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        """Returns (output, bias) like Megatron linear layers."""
        hidden_states, fan_hidden = self._fan_transform(hidden_states)

        nvtx_range_push(suffix="fan_qkv_fc2")
        if self.config.fan_enable_qk_fan:
            # [sq, b, h] --> [sq, b, kv_channels * num_attention_heads/tp_size]
            output_q, _ = self.linear_fc2_q(fan_hidden)
            # [sq, b, h] --> [sq, b, kv_channels * num_query_groups/tp_size]
            output_k, _ = self.linear_fc2_k(fan_hidden)
            # [sq, b, h] --> [sq, b, kv_channels * num_query_groups/tp_size]
            output_v, _ = self.linear_fc2_v(hidden_states)

            # [sq, b, kv_channels * num_attention_heads/tp_size] --> [sq, b, num_query_groups/tp_size, kv_channels * num_attention_heads // tp_size // (num_query_groups/tp_size)]
            output_q = output_q.view(
                output_q.size(0),
                output_q.size(1),
                self.num_query_groups_per_partition,
                self.num_attention_heads_per_partition // self.num_query_groups_per_partition * self.hidden_size_per_attention_head
            )

            # [sq, b, kv_channels * num_query_groups/tp_size] --> [sq, b, num_query_groups/tp_size, kv_channels]
            output_k = output_k.view(
                output_k.size(0),
                output_k.size(1),
                self.num_query_groups_per_partition,
                self.hidden_size_per_attention_head
            )

            # [sq, b, kv_channels * num_query_groups/tp_size] --> [sq, b, num_query_groups/tp_size, kv_channels]
            output_v = output_v.view(
                output_v.size(0),
                output_v.size(1),
                self.num_query_groups_per_partition,
                self.hidden_size_per_attention_head
            )
            
            # [sq, b, num_query_groups/tp_size, (num_attention_heads/tp_size // num_query_groups/tp_size + 2) * kv_channels]
            output = torch.cat((output_q, output_k, output_v), dim=-1)

            output = output.view(
                output.size(0),
                output.size(1),
                self.num_query_groups_per_partition * \
                (self.num_attention_heads_per_partition // self.num_query_groups_per_partition + 2) * \
                self.hidden_size_per_attention_head
            )
            output_bias = None

        else:
            # [sq, b, h] --> [sq, b, num_query_groups * (np/num_query_groups + 2) * kv_channels)]
            output, output_bias = self.linear_fc2(fan_hidden)
        nvtx_range_pop(suffix="fan_qkv_fc2")

        return output, output_bias

    def backward_dw(self):
        if self.linear_fc2 is not None and hasattr(self.linear_fc2, "backward_dw"):
            self.linear_fc2.backward_dw()
        if self.linear_fc2_k is not None and hasattr(self.linear_fc2_k, "backward_dw"):
            self.linear_fc2_k.backward_dw()
        if self.linear_fc2_v is not None and hasattr(self.linear_fc2_v, "backward_dw"):
            self.linear_fc2_v.backward_dw()
        if self.linear_fc1 is not None and hasattr(self.linear_fc1, "backward_dw"):
            self.linear_fc1.backward_dw()


class FanLayer(FanQKVLinear):
    """Thin adapter over :class:`FanQKVLinear` for the ``pre_mlp_layernorm`` position.

    Reuses the core FAN logic (:meth:`_fan_transform`) from the parent class.

    Interface differences vs ``FanQKVLinear``:
    1. ``__init__`` accepts ``(config, hidden_size, eps)`` -- the signature that
       ``build_module`` uses for layer-norm modules.
    2. Always builds a single ``linear_fc2`` (hidden -> hidden), no Q/K/V split.
    3. ``forward()`` returns a **single tensor** (not ``(output, bias)``).
    """

    def __init__(
        self,
        config: TransformerConfig,
        hidden_size: int,
        eps: float = 1e-5,  # kept for interface compat, unused
        *,
        submodules: Optional[FanSubmodules] = None,
        p_ratio: Optional[float] = None,
        use_p_bias: Optional[bool] = None,
        **kwargs,
    ):
        super().__init__(
            input_size=hidden_size,
            output_size=hidden_size,
            config=config,
            init_method=config.init_method,
            bias=config.add_bias_linear,
            gather_output=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="fan_norm",
            submodules=submodules,
            p_ratio=p_ratio,
            use_p_bias=use_p_bias,
        )
        self.linear_fc2_q = None
        self.linear_fc2_k = None
        self.linear_fc2_v = None
        if self.fan_no_compress:
            # Need fc2 to project 2*hidden_size → hidden_size.
            # Parent may have built one (non-qk-fan path); if not, build here.
            if self.linear_fc2 is None:
                self.linear_fc2 = build_module(
                    submodules.linear_fc2,
                    2 * hidden_size,
                    hidden_size,
                    config=config,
                    init_method=config.init_method,
                    bias=config.add_bias_linear,
                    gather_output=False,
                    skip_bias_add=False,
                    is_expert=False,
                    tp_comm_buffer_name="fan_norm_fc2",
                    tp_group=self.tp_group,
                )
        else:
            self.linear_fc2 = IdentityOp()

    def forward(self, hidden_states, **kwargs):
        """Return a single tensor, matching the layernorm interface."""
        _, fan_hidden = self._fan_transform(hidden_states)
        if self.fan_no_compress:
            # fan_hidden is (seq/tp, b, 2h) in SP, (seq, b, 2h) otherwise.
            # Apply fc2 to compress 2h → h.
            fan_hidden, _ = self.linear_fc2(fan_hidden)
            # After ColumnParallelLinear (with SP): (seq, b, h/tp).
            # Gather across TP to recover full hidden dim.
            fan_hidden = gather_from_tensor_model_parallel_region(
                fan_hidden, group=self.tp_group
            )
            # Now (seq, b, h). Scatter back for SP if needed.
            if getattr(self.config, "sequence_parallel", False):
                fan_hidden = scatter_to_sequence_parallel_region(fan_hidden)
        else:
            # fc1 (SP) does all-gather on seq dim: (seq/tp, b, h) -> (seq, b, ...).
            # Scatter back so the output shape matches the SP-sharded residual.
            if getattr(self.config, "sequence_parallel", False):
                fan_hidden = scatter_to_sequence_parallel_region(fan_hidden)
        return fan_hidden

