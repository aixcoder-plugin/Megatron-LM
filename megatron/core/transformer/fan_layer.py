from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import NoReturn, Optional, Tuple, Union

import torch
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.tensor_parallel.mappings import gather_from_tensor_model_parallel_region
from megatron.core.utils import (
    nvtx_range_pop,
    nvtx_range_push,
    get_tensor_model_parallel_group_if_none,
    get_pg_size,
)


from .transformer_config import TransformerConfig

from megatron.training.utils import print_rank_0

from megatron.core.dist_checkpointing.mapping import (
    ReplicaId,
    ShardedStateDict,
)


@dataclass
class FanSubmodules:

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

        self.p_ratio = (
            p_ratio if p_ratio is not None else getattr(self.config, "fan_p_ratio", 0.25)
        )
        self.use_p_bias = (
            use_p_bias
            if use_p_bias is not None
            else getattr(self.config, "fan_use_p_bias", True)
        )

        assert 0.0 <= self.p_ratio <= 0.5, "p_ratio must be between 0 and 0.5"

        p_output_size = int(self.input_size * self.p_ratio)
        g_output_size = self.input_size - p_output_size * 2
        self.fused_dims = (p_output_size, g_output_size)

        # We gather the intermediate so we can split on full (p, g) dims safely.
        fc1_out_size = p_output_size + g_output_size  # = input_size - p_output_size

        tp_group = get_tensor_model_parallel_group_if_none(tp_group, is_expert=is_expert)
        self.tp_group = tp_group

        assert submodules is not None, "FanQKVLinear requires `submodules` to be provided."
        assert (
            submodules.linear_fc1 is not None and submodules.linear_fc2 is not None
        ), "FanQKVLinear requires `linear_fc1` and `linear_fc2` submodules."

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


        # Final projection keeps the original `linear_qkv` parallel semantics.
        self.linear_fc2 = build_module(
            submodules.linear_fc2,
            self.input_size,
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

        # must not apply sequence-parallel all-gather again.
        if getattr(self.config, "sequence_parallel", False):
            if hasattr(self.linear_fc2, "sequence_parallel"):
                self.linear_fc2.sequence_parallel = False
            if hasattr(self.linear_fc2, "allreduce_dgrad"):
                world_size = get_pg_size(tp_group)
                disable_grad_reduce = bool(getattr(self.linear_fc2, "disable_grad_reduce", False))
                self.linear_fc2.allreduce_dgrad = (world_size > 1) and (not disable_grad_reduce)
            for attr in (
                "ub_overlap_rs_fprop",
                "ub_overlap_ag_dgrad",
                "ub_overlap_ag_fprop",
                "ub_overlap_rs_dgrad",
                "ub_bulk_dgrad",
                "ub_bulk_wgrad",
            ):
                if hasattr(self.linear_fc2, attr):
                    setattr(self.linear_fc2, attr, False)

        # Propagate save_original_input to inner TE modules if needed.
        if self._save_original_input:
            self.save_original_input = True

    @property
    def save_original_input(self) -> bool:
        return bool(self._save_original_input)

    @save_original_input.setter
    def save_original_input(self, value: bool) -> None:
        self._save_original_input = bool(value)
        for m in (getattr(self, "linear_fc1", None), getattr(self, "linear_fc2", None)):
            if m is not None and hasattr(m, "save_original_input"):
                m.save_original_input = bool(value)

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        """Returns (output, bias) like Megatron linear layers."""
        nvtx_range_push(suffix="fan_qkv_fc1")
        pg, _ = self.linear_fc1(hidden_states)
        # Make pg non-parallel to safely split into (p, g) on full dims.
        pg = gather_from_tensor_model_parallel_region(pg, group=self.tp_group)
        nvtx_range_pop(suffix="fan_qkv_fc1")

        nvtx_range_push(suffix="fan_qkv_activation")
        p, g = pg.split(self.fused_dims, dim=-1)
        if self.activation_func is not None:
            fan_hidden = torch.cat((torch.cos(p), torch.sin(p), self.activation_func(g)), dim=-1)
        else:
            fan_hidden = torch.cat((torch.cos(p), torch.sin(p), g), dim=-1)
        nvtx_range_pop(suffix="fan_qkv_activation")

        nvtx_range_push(suffix="fan_qkv_fc2")
        output, output_bias = self.linear_fc2(fan_hidden)
        nvtx_range_pop(suffix="fan_qkv_fc2")

        return output, output_bias

    def backward_dw(self):
        if hasattr(self.linear_fc2, "backward_dw"):
            self.linear_fc2.backward_dw()
        if hasattr(self.linear_fc1, "backward_dw"):
            self.linear_fc1.backward_dw()


class FanLinear(MegatronModule):
    """

    We use the following notation:
     h: hidden size
     p: number of tensor model parallel partitions
     b: batch size
     s: sequence length
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: FanSubmodules,
        input_size: Optional[int] = None,
        output_size: Optional[int] = None,
        p_ratio: Optional[float] = None,
        use_p_bias: Optional[bool] = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        super().__init__(config=config)

        self.config: TransformerConfig = config

        self.input_size = input_size if input_size is not None else self.config.hidden_size
        self.output_size = output_size if output_size is not None else self.config.hidden_size
        self.p_ratio = (
            p_ratio if p_ratio is not None else getattr(self.config, "fan_p_ratio", 0.25)
        )
        self.use_p_bias = (
            use_p_bias
            if use_p_bias is not None
            else getattr(self.config, "fan_use_p_bias", True)
        )

        # Ensure the p_ratio is within a valid range
        assert 0 <= self.p_ratio <= 0.5, "p_ratio must be between 0 and 0.5"

        p_output_size = int(self.input_size * self.p_ratio)
        g_output_size = self.input_size - p_output_size * 2  # Account for cosine and sine terms

        self.fused_dims = (p_output_size, g_output_size)

        tp_group = get_tensor_model_parallel_group_if_none(tp_group, is_expert=False)


        self.linear_fc1 = build_module(
            submodules.linear_fc1,
            self.input_size,
            p_output_size + g_output_size,
            config=self.config,
            init_method=self.config.init_method,
            # TODO: fc1的输出虽然存在一系列的操作，但都是元素级的，理论上能在fc2算完之后再通信聚合，但是后续对hidden_size 维度做split，
            # 这个和 ColumnParallelLinear 的维度一致，所以存在冲突，需要额外增加一层聚合的通信，这个能通过mask优化解决
            gather_output=True,
            bias=self.use_p_bias,
            # TODO: 需要确认，全连接层的bias后续只有部分过激活函数，bias不能和后续的激活函数做算子融合
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="fc1",
            tp_group=tp_group,
        )

        if self.config.use_te_activation_func and not (submodules.activation_func is None):
            self.activation_func = build_module(submodules.activation_func, config=self.config)
        else:
            self.activation_func = self.config.activation_func

        self.linear_fc2 = build_module(
            submodules.linear_fc2,
            self.input_size,
            self.output_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
            # Fan fc2 uses ColumnParallelLinear to stay compatible with sequence_parallel.
            gather_output=True,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="fc2",
            tp_group=tp_group,
        )

    def forward(self, hidden_states, per_token_scale=None):
        """Perform the forward pass through the fan block."""
        # [s, b, h] => [s, b, (p_output_size + g_output_size)/p]
        nvtx_range_push(suffix="linear_fc1")
        intermediate_non_parallel, _ = self.linear_fc1(hidden_states)
        nvtx_range_pop(suffix="linear_fc1")

        nvtx_range_push(suffix="activation")
        p, g = intermediate_non_parallel.split(self.fused_dims, dim=-1)
        intermediate_non_parallel = torch.cat((torch.cos(p), torch.sin(p), self.activation_func(g)), dim=-1)
        nvtx_range_pop(suffix="activation")

        # [s, b, h]
        nvtx_range_push(suffix="linear_fc2")
        output, _ = self.linear_fc2(intermediate_non_parallel)
        nvtx_range_pop(suffix="linear_fc2")

        return output, None

    # pylint: disable=missing-function-docstring
    def sharded_state_dict(
        self, prefix: str = "", sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        """Return the sharded state dictionary of the module."""
        sharded_state_dict = {}
        for name, module in self._modules.items():
            sub_sd = module.sharded_state_dict(f"{prefix}{name}.", sharded_offsets, metadata)
            sharded_state_dict.update(sub_sd)
        return sharded_state_dict

    def backward_dw(self):
        self.linear_fc2.backward_dw()
        self.linear_fc1.backward_dw()


