#!/usr/bin/env python3
"""
Formal Language Tasks (Megatron/mcore)

目的：
- 复用 `test_formal_language.py` 的数据生成与评测逻辑
- 将模型实现替换为 Megatron-LM(mcore) 的 `GPTModel`（支持开启 FanQKV 与 StackMemory）
- 用快速合成任务检查 Megatron 训练/前向/反向/并行（尤其是 fan/stack 路径）是否正常

建议运行（单卡）：
  torchrun --standalone --nproc_per_node 1 \
    aix_moe_train/test_formal_language/test_formal_language_megatron.py \
      --task parity --model both --epochs 5 --batch-size 64 \
      --fan-layer-enabled --stack-memory-enabled --stack-memory-num-heads 4 --stack-memory-dim 16

建议运行（多卡 TP）：
  torchrun --standalone --nproc_per_node 4 \
    aix_moe_train/test_formal_language/test_formal_language_megatron.py \
      --task parity --model both --epochs 5 --batch-size 64 \
      --tensor-model-parallel-size 4 --pipeline-model-parallel-size 1 \
      --fan-layer-enabled --stack-memory-enabled --stack-memory-num-heads 4 --stack-memory-dim 16
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, DistributedSampler

# Ensure we import the *repo* Megatron-LM, not a pip-installed `megatron` package.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from megatron.core import parallel_state
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)


# ============================================================================
# Synthetic tasks (same as original torch script, but model-agnostic)
# ============================================================================


class ParityCheckDataset(Dataset):
    """Parity Check: 输入 ab 字符串，输出 True/False（b 的数量是否为偶数）"""

    def __init__(self, min_len=1, max_len=20, num_samples=5000, seed=42):
        random.seed(seed)
        self.samples: List[Tuple[str, bool]] = []
        for _ in range(num_samples):
            length = random.randint(min_len, max_len)
            s = "".join(random.choice("ab") for _ in range(length))
            label = s.count("b") % 2 == 0
            self.samples.append((s, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class EvenPairsDataset(Dataset):
    """Even Pairs: ab/ba 子串数量是否为偶数"""

    def __init__(self, min_len=2, max_len=20, num_samples=5000, seed=42):
        random.seed(seed)
        self.samples: List[Tuple[str, bool]] = []
        for _ in range(num_samples):
            length = random.randint(min_len, max_len)
            s = "".join(random.choice("ab") for _ in range(length))
            count = sum(1 for i in range(len(s) - 1) if s[i : i + 2] in ["ab", "ba"])
            label = count % 2 == 0
            self.samples.append((s, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class CycleNavigationDataset(Dataset):
    """Cycle Navigation: 0=不动, 1=+1, 2=-1，从 0 出发，最终位置 mod 5"""

    def __init__(self, min_len=1, max_len=20, num_samples=5000, seed=42):
        random.seed(seed)
        self.samples: List[Tuple[List[int], int]] = []
        for _ in range(num_samples):
            length = random.randint(min_len, max_len)
            moves = [random.randint(0, 2) for _ in range(length)]
            position = 0
            for m in moves:
                if m == 1:
                    position = (position + 1) % 5
                elif m == 2:
                    position = (position - 1) % 5
            self.samples.append((moves, position))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class ReverseStringDataset(Dataset):
    """Reverse String: 输入符号序列，输出反转序列"""

    def __init__(self, min_len=1, max_len=10, num_samples=5000, seed=42, vocab_size=5):
        random.seed(seed)
        self.samples: List[Tuple[List[int], List[int]]] = []
        self.vocab_size = vocab_size
        for _ in range(num_samples):
            length = random.randint(min_len, max_len)
            s = [random.randint(0, vocab_size - 1) for _ in range(length)]
            rev = s[::-1]
            self.samples.append((s, rev))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class BalancedParenthesesDataset(Dataset):
    """Balanced Parentheses: 括号是否匹配（50% 平衡生成，50% 随机）"""

    def __init__(self, min_len=2, max_len=20, num_samples=5000, seed=42):
        random.seed(seed)
        self.samples: List[Tuple[str, bool]] = []
        for _ in range(num_samples):
            if random.random() < 0.5:
                s, label = self._generate_balanced(random.randint(min_len // 2, max_len // 2))
            else:
                length = random.randint(min_len, max_len)
                s = "".join(random.choice("()") for _ in range(length))
                label = self._is_balanced(s)
            self.samples.append((s, label))

    def _generate_balanced(self, n):
        if n == 0:
            return "", True
        s = ["("] * n + [")"] * n
        random.shuffle(s)
        result = []
        count = 0
        for c in s:
            if c == "(":
                result.append(c)
                count += 1
            else:
                if count > 0:
                    result.append(c)
                    count -= 1
        while count > 0:
            result.append(")")
            count -= 1
        return "".join(result), True

    def _is_balanced(self, s):
        count = 0
        for c in s:
            if c == "(":
                count += 1
            else:
                count -= 1
            if count < 0:
                return False
        return count == 0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class ModularArithmeticDataset(Dataset):
    """Modular Arithmetic: 带括号表达式求值 mod 5"""

    def __init__(self, max_depth=3, num_samples=5000, seed=42):
        random.seed(seed)
        self.samples: List[Tuple[str, int]] = []
        self.ops = ["+", "-", "*"]
        for _ in range(num_samples):
            expr, value = self._generate_expr(random.randint(1, max_depth))
            self.samples.append((expr, value % 5))

    def _generate_expr(self, depth):
        if depth == 0 or random.random() < 0.3:
            val = random.randint(0, 4)
            return str(val), val

        left_expr, left_val = self._generate_expr(depth - 1)
        right_expr, right_val = self._generate_expr(depth - 1)
        op = random.choice(self.ops)

        if op == "+":
            result = (left_val + right_val) % 5
        elif op == "-":
            result = (left_val - right_val) % 5
        else:
            result = (left_val * right_val) % 5

        expr = f"({left_expr}{op}{right_expr})"
        return expr, result

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ============================================================================
# Tokenizers (same vocab design as original torch script)
# ============================================================================


def create_binary_classification_tokenizer():
    return {
        "<pad>": 0,
        "<bos>": 1,
        "<eos>": 2,
        "<sep>": 3,
        "a": 4,
        "b": 5,
        "(": 6,
        ")": 7,
        "T": 8,
        "F": 9,
    }


def create_cycle_tokenizer():
    return {
        "<pad>": 0,
        "<bos>": 1,
        "<eos>": 2,
        "<sep>": 3,
        "0": 4,
        "1": 5,
        "2": 6,
        "r0": 7,
        "r1": 8,
        "r2": 9,
        "r3": 10,
        "r4": 11,
    }


def create_reverse_tokenizer(vocab_size=5):
    tok = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<sep>": 3}
    for i in range(vocab_size):
        tok[f"v{i}"] = 4 + i
    return tok


def create_modular_tokenizer():
    return {
        "<pad>": 0,
        "<bos>": 1,
        "<eos>": 2,
        "<sep>": 3,
        "0": 4,
        "1": 5,
        "2": 6,
        "3": 7,
        "4": 8,
        "+": 9,
        "-": 10,
        "*": 11,
        "(": 12,
        ")": 13,
        "r0": 14,
        "r1": 15,
        "r2": 16,
        "r3": 17,
        "r4": 18,
    }


# ============================================================================
# Megatron-style dataset: return tokens/labels/attention_mask/loss_mask/position_ids
# ============================================================================


def _build_causal_attention_mask(seq_len: int) -> torch.Tensor:
    # Same convention as GPTDataset: bool mask where True means masked-out positions.
    # Shape: [1, seq, seq]
    attn = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.float32)).unsqueeze(0)
    return attn < 0.5


@dataclass
class PackedSample:
    full_tokens: List[int]  # length = seq_len + 1


class MegatronFormalDataset(Dataset):
    def __init__(
        self,
        raw_dataset: Dataset,
        tokenizer: Dict[str, int],
        seq_len: int,
        task_name: str,
        reverse_vocab_size: int = 5,
    ):
        self.raw_dataset = raw_dataset
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.task_name = task_name
        self.reverse_vocab_size = reverse_vocab_size
        self.pad_id = tokenizer["<pad>"]
        self.sep_id = tokenizer["<sep>"]
        self.attention_mask = _build_causal_attention_mask(seq_len)
        self.position_ids = torch.arange(seq_len, dtype=torch.long)

        # Basic sanity
        assert self.pad_id == 0, "This script assumes <pad> id is 0."

    def __len__(self):
        return len(self.raw_dataset)

    def _encode_one(self, item) -> PackedSample:
        t = self.tokenizer

        if self.task_name in ("parity", "even_pairs", "balanced"):
            s, label = item
            seq = [t["<bos>"]] + [t[c] for c in s] + [t["<sep>"], t["T"] if label else t["F"], t["<eos>"]]
        elif self.task_name == "cycle":
            moves, result = item
            seq = [t["<bos>"]] + [t[str(m)] for m in moves] + [t["<sep>"], t[f"r{result}"], t["<eos>"]]
        elif self.task_name == "reverse":
            s, rev = item
            # tokenizer: v0..v{V-1}
            seq = [t["<bos>"]] + [t[f"v{v}"] for v in s] + [t["<sep>"]] + [t[f"v{v}"] for v in rev] + [
                t["<eos>"]
            ]
        elif self.task_name == "modular":
            expr, result = item
            seq = [t["<bos>"]] + [t[c] for c in expr if c in t] + [t["<sep>"], t[f"r{result}"], t["<eos>"]]
        else:
            raise ValueError(f"Unknown task_name={self.task_name}")

        # We will output tokens/labels of length seq_len, so need full length = seq_len + 1.
        if len(seq) > self.seq_len + 1:
            seq = seq[: self.seq_len + 1]
        else:
            seq = seq + [self.pad_id] * (self.seq_len + 1 - len(seq))

        return PackedSample(full_tokens=seq)

    def __getitem__(self, idx):
        packed = self._encode_one(self.raw_dataset[idx])
        text = torch.tensor(packed.full_tokens, dtype=torch.long)  # [seq_len+1]
        tokens = text[:-1].contiguous()  # [seq_len]
        labels = text[1:].contiguous()  # [seq_len]

        # Loss mask: ignore pad labels
        loss_mask = torch.ones(self.seq_len, dtype=torch.float32)
        loss_mask[labels == self.pad_id] = 0.0

        # Keep embedding safe: map pad to 0 (already) and make labels pad->0
        tokens[tokens == self.pad_id] = 0
        labels[labels == self.pad_id] = 0

        return {
            "tokens": tokens,
            "labels": labels,
            "attention_mask": self.attention_mask,
            "loss_mask": loss_mask,
            "position_ids": self.position_ids,
        }


# ============================================================================
# Evaluation (same logic as original torch script, but works on Megatron logits)
# ============================================================================


@torch.no_grad()
def evaluate_classification(model: GPTModel, dataloader: DataLoader, device: torch.device, sep_id: int, pad_id: int):
    model.eval()
    correct = 0
    total = 0
    for batch in dataloader:
        input_ids = batch["tokens"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        position_ids = batch["position_ids"].to(device)

        logits = model(input_ids, position_ids, attention_mask, runtime_gather_output=True)
        preds = logits.argmax(-1)

        sep_mask = input_ids == sep_id
        for i in range(input_ids.size(0)):
            sep_pos = sep_mask[i].nonzero()
            if len(sep_pos) > 0:
                pos = sep_pos[0].item()
                if pos < labels.size(1) and labels[i, pos].item() != pad_id:
                    if preds[i, pos].item() == labels[i, pos].item():
                        correct += 1
                    total += 1
    # Aggregate across data-parallel replicas (keep TP semantics unchanged).
    dp_group = parallel_state.get_data_parallel_group(with_context_parallel=True)
    if dp_group.size() > 1:
        stats = torch.tensor([correct, total], device=device, dtype=torch.long)
        torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM, group=dp_group)
        correct = int(stats[0].item())
        total = int(stats[1].item())
    return correct / total if total > 0 else 0.0


@torch.no_grad()
def evaluate_sequence(model: GPTModel, dataloader: DataLoader, device: torch.device, sep_id: int, pad_id: int):
    model.eval()
    correct = 0
    total = 0
    for batch in dataloader:
        input_ids = batch["tokens"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        position_ids = batch["position_ids"].to(device)

        logits = model(input_ids, position_ids, attention_mask, runtime_gather_output=True)
        preds = logits.argmax(-1)

        for i in range(input_ids.size(0)):
            sep_pos = (input_ids[i] == sep_id).nonzero()
            if len(sep_pos) == 0:
                continue
            start = sep_pos[0].item()
            for j in range(start, labels.size(1)):
                if labels[i, j].item() != pad_id:
                    if preds[i, j].item() == labels[i, j].item():
                        correct += 1
                    total += 1
    dp_group = parallel_state.get_data_parallel_group(with_context_parallel=True)
    if dp_group.size() > 1:
        stats = torch.tensor([correct, total], device=device, dtype=torch.long)
        torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM, group=dp_group)
        correct = int(stats[0].item())
        total = int(stats[1].item())
    return correct / total if total > 0 else 0.0


# ============================================================================
# Megatron training loop (mcore schedule)
# ============================================================================


def _init_distributed(tp: int, pp: int):
    if not torch.distributed.is_available():
        raise RuntimeError("torch.distributed 不可用")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if not torch.distributed.is_initialized():
        # Newer PyTorch supports `device_id` to avoid rank->GPU mapping ambiguity warnings.
        try:
            torch.distributed.init_process_group(
                backend="nccl", device_id=torch.device("cuda", local_rank)
            )
        except TypeError:
            torch.distributed.init_process_group(backend="nccl")

    parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(tp, pp)


def _pad_vocab_size(vocab_size: int, multiple: int) -> int:
    if multiple <= 1:
        return vocab_size
    return int(math.ceil(vocab_size / multiple) * multiple)


def _build_model(
    *,
    vocab_size: int,
    seq_len: int,
    args,
    enable_fan: bool,
    enable_stack: bool,
) -> Tuple[GPTModel, TransformerConfig]:
    # dtype
    if args.dtype == "bf16":
        params_dtype = torch.bfloat16
        fp16 = False
        bf16 = True
    elif args.dtype == "fp16":
        params_dtype = torch.float16
        fp16 = True
        bf16 = False
    else:
        params_dtype = torch.float32
        fp16 = False
        bf16 = False

    # stack defaults (avoid 0/0 crash; also aligns with CLI help text)
    stack_num_heads = args.stack_memory_num_heads
    if enable_stack and stack_num_heads == 0:
        stack_num_heads = args.num_attention_heads
    head_dim = args.hidden_size // stack_num_heads if enable_stack else None
    stack_dim = args.stack_memory_dim
    if enable_stack and stack_dim == 0:
        assert head_dim is not None
        stack_dim = head_dim

    config = TransformerConfig(
        # architecture
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        ffn_hidden_size=args.ffn_hidden_size,
        num_attention_heads=args.num_attention_heads,
        normalization=args.normalization,
        init_method_std=0.04,
        # parallel
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        sequence_parallel=args.sequence_parallel,
        # training/misc
        use_cpu_initialization=True,
        params_dtype=params_dtype,
        pipeline_dtype=params_dtype,
        fp16=fp16,
        bf16=bf16,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        # FAN / StackMemory
        fan_qkv_enabled=enable_fan,
        fan_p_ratio=args.fan_p_ratio,
        fan_use_p_bias=args.fan_use_p_bias,
        stack_memory_enabled=enable_stack,
        stack_memory_num_heads=stack_num_heads,
        stack_memory_slots=args.stack_memory_slots,
        stack_memory_dim=stack_dim,
    )

    use_te = args.transformer_impl == "transformer_engine"
    if use_te:
        layer_spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=None,
            moe_grouped_gemm=False,
            qk_layernorm=False,
            multi_latent_attention=False,
            moe_use_legacy_grouped_gemm=False,
            qk_l2_norm=False,
            use_kitchen=False,
            use_te_activation_func=getattr(config, "use_te_activation_func", False),
            fan_qkv_enabled=enable_fan,
        )
    else:
        layer_spec = get_gpt_layer_local_spec(
            num_experts=None,
            moe_grouped_gemm=False,
            qk_layernorm=False,
            multi_latent_attention=False,
            moe_use_legacy_grouped_gemm=False,
            normalization=args.normalization,
            qk_l2_norm=False,
            use_kitchen=False,
            fan_qkv_enabled=enable_fan,
        )

    padded_vocab_size = _pad_vocab_size(vocab_size, args.tensor_model_parallel_size)

    model = GPTModel(
        config=config,
        transformer_layer_spec=layer_spec,
        vocab_size=padded_vocab_size,
        max_sequence_length=seq_len,
        pre_process=True,
        post_process=True,
        fp16_lm_cross_entropy=False,
        parallel_output=True,
        share_embeddings_and_output_weights=True,
        position_embedding_type="learned_absolute",
    ).cuda()

    return model, config


def _sync_tp_replicated_grads(model: torch.nn.Module, config: TransformerConfig):
    """只同步需要 TP 域同步的“复制参数”（目前主要是 StackMemory）。"""
    if config.tensor_model_parallel_size <= 1:
        return
    tp_group = parallel_state.get_tensor_model_parallel_group()
    tp_size = tp_group.size()
    for _, p in model.named_parameters():
        if p.grad is None:
            continue
        if getattr(p, "average_gradients_across_tp_domain", False):
            torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.SUM, group=tp_group)
            p.grad.div_(tp_size)
        elif config.sequence_parallel and getattr(p, "sequence_parallel", False):
            torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.SUM, group=tp_group)


def _sync_dp_grads(model: torch.nn.Module):
    """数据并行梯度同步（对大多数参数做 DP 平均）。"""
    dp_group = parallel_state.get_data_parallel_group(with_context_parallel=True)
    dp_size = dp_group.size()
    if dp_size <= 1:
        return
    for p in model.parameters():
        if p.grad is None:
            continue
        # MoE expert 参数通常会标记 allreduce=False；本测试脚本默认不启 MoE，但这里保守处理。
        if not getattr(p, "allreduce", True):
            continue
        torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.SUM, group=dp_group)
        p.grad.div_(dp_size)


def _train_one_model(
    *,
    model: GPTModel,
    config: TransformerConfig,
    train_loader: DataLoader,
    test_loader: DataLoader,
    task_name: str,
    is_classification: bool,
    sep_id: int,
    pad_id: int,
    device: torch.device,
    args,
) -> Tuple[float, float]:
    forward_backward_func = get_forward_backward_func()
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    def forward_step_func(data_iterator, model):
        batch = next(data_iterator)
        tokens = batch["tokens"].to(device)
        labels = batch["labels"].to(device)
        loss_mask = batch["loss_mask"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        position_ids = batch["position_ids"].to(device)

        output = model(tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask)

        def loss_func(output_tensor):
            losses = output_tensor.view(-1).float()
            lm = loss_mask.view(-1).float()
            loss_sum = torch.sum(losses * lm)
            num_tokens = lm.sum().clone().detach().to(torch.int)
            # For logging: [loss_sum, num_tokens]
            reporting = torch.cat([loss_sum.detach().view(1), num_tokens.detach().view(1).to(loss_sum.dtype)])
            return loss_sum, num_tokens, {"lm loss": reporting}

        return output, loss_func

    best_acc = 0.0
    last_acc = 0.0
    step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        train_iter = iter(train_loader)

        for _ in range(len(train_loader)):
            step += 1
            optim.zero_grad(set_to_none=True)
            losses_reduced = forward_backward_func(
                forward_step_func=forward_step_func,
                data_iterator=train_iter,
                model=model,
                num_microbatches=1,
                seq_length=args.seq_length,
                micro_batch_size=args.batch_size,
                decoder_seq_length=args.seq_length,
                forward_only=False,
            )

            _sync_tp_replicated_grads(model, config)
            _sync_dp_grads(model)
            if args.clip_grad is not None and args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optim.step()

            if step % args.log_interval == 0:
                # losses_reduced: list[dict] (one per microbatch)
                rep = losses_reduced[-1]["lm loss"]
                loss_avg = (rep[0] / torch.clamp(rep[1], min=1)).item()
                if torch.distributed.get_rank() == 0:
                    print(f"  Epoch {epoch:3d} | Step {step:6d} | Loss {loss_avg:.4f}")

        # epoch eval
        if is_classification:
            last_acc = evaluate_classification(model, test_loader, device, sep_id, pad_id)
        else:
            last_acc = evaluate_sequence(model, test_loader, device, sep_id, pad_id)
        best_acc = max(best_acc, last_acc)
        if torch.distributed.get_rank() == 0:
            print(f"  [Eval] Epoch {epoch:3d} | Acc {last_acc:.4f} | Best {best_acc:.4f}")

    return best_acc, last_acc


TASKS = {
    "parity": {
        "name": "Parity Check (RE)",
        "dataset_class": ParityCheckDataset,
        "tokenizer_fn": create_binary_classification_tokenizer,
        "is_classification": True,
        "seq_len": 64,
    },
    "even_pairs": {
        "name": "Even Pairs (RE)",
        "dataset_class": EvenPairsDataset,
        "tokenizer_fn": create_binary_classification_tokenizer,
        "is_classification": True,
        "seq_len": 64,
    },
    "cycle": {
        "name": "Cycle Navigation (RE)",
        "dataset_class": CycleNavigationDataset,
        "tokenizer_fn": create_cycle_tokenizer,
        "is_classification": True,
        "seq_len": 64,
    },
    "reverse": {
        "name": "Reverse String (DCF)",
        "dataset_class": ReverseStringDataset,
        "tokenizer_fn": lambda: create_reverse_tokenizer(5),
        "is_classification": False,
        "seq_len": 64,
    },
    "balanced": {
        "name": "Balanced Parentheses (DCF)",
        "dataset_class": BalancedParenthesesDataset,
        "tokenizer_fn": create_binary_classification_tokenizer,
        "is_classification": True,
        "seq_len": 64,
    },
    "modular": {
        "name": "Modular Arithmetic (DCF)",
        "dataset_class": ModularArithmeticDataset,
        "tokenizer_fn": create_modular_tokenizer,
        "is_classification": True,
        "seq_len": 64,
    },
}


def main():
    parser = argparse.ArgumentParser(description="Formal Language Tasks (Megatron/mcore)")

    parser.add_argument(
        "--task",
        type=str,
        default="all",
        choices=["all"] + list(TASKS.keys()),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["transformer", "fanformer", "stacktrans", "fanstack", "all"],
    )

    # Training
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64, dest="batch_size")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--train-samples", type=int, default=5000)
    parser.add_argument("--test-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)

    # Model config (small default)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--ffn-hidden-size", type=int, default=512)
    parser.add_argument("--num-attention-heads", type=int, default=4)
    parser.add_argument("--normalization", type=str, default="LayerNorm", choices=["LayerNorm", "RMSNorm"])
    parser.add_argument("--seq-length", type=int, default=64)
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "bf16", "fp16"])

    # Parallelism
    parser.add_argument("--tensor-model-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-model-parallel-size", type=int, default=1)
    parser.add_argument("--sequence-parallel", action="store_true")
    parser.add_argument(
        "--transformer-impl", type=str, default="local", choices=["local", "transformer_engine"]
    )

    # Fan/Stack flags (align with megatron/training/arguments.py naming)
    parser.add_argument("--fan-layer-enabled", "--fan-qkv-enabled", action="store_true", dest="fan_qkv_enabled")
    parser.add_argument("--fan-p-ratio", type=float, default=0.25, dest="fan_p_ratio")
    parser.add_argument("--fan-disable-p-bias", action="store_false", dest="fan_use_p_bias")
    parser.set_defaults(fan_use_p_bias=True)

    parser.add_argument("--stack-memory-enabled", action="store_true", dest="stack_memory_enabled")
    parser.add_argument("--stack-memory-num-heads", type=int, default=0, dest="stack_memory_num_heads")
    parser.add_argument("--stack-memory-slots", type=int, default=16, dest="stack_memory_slots")
    parser.add_argument("--stack-memory-dim", type=int, default=0, dest="stack_memory_dim")

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("该测试脚本依赖 CUDA（mcore schedule 内部固定用了 device='cuda'）。")

    if args.pipeline_model_parallel_size != 1:
        raise ValueError("当前脚本仅支持 PP=1（为了简化 logits/accuracy 评测）。")

    _init_distributed(args.tensor_model_parallel_size, args.pipeline_model_parallel_size)
    device = torch.device("cuda")

    # Seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    model_parallel_cuda_manual_seed(args.seed)

    model_types = ["transformer", "fanformer", "stacktrans", "fanstack"] if args.model == "all" else [args.model]
    tasks = list(TASKS.keys()) if args.task == "all" else [args.task]

    if torch.distributed.get_rank() == 0:
        print("Formal Language Tasks (Megatron/mcore)")
        print(f"  device={device}")
        print(f"  TP={args.tensor_model_parallel_size}, PP={args.pipeline_model_parallel_size}, SP={args.sequence_parallel}")

    all_results = {}

    for task_name in tasks:
        task = TASKS[task_name]
        if torch.distributed.get_rank() == 0:
            print("\n" + "=" * 60)
            print(f"任务: {task['name']}")
            print("=" * 60)

        tokenizer = task["tokenizer_fn"]()
        vocab_size = len(tokenizer)
        pad_id = tokenizer["<pad>"]
        sep_id = tokenizer["<sep>"]

        # build raw datasets
        if task_name == "reverse":
            train_raw = task["dataset_class"](num_samples=args.train_samples, seed=42, vocab_size=5)
            test_raw = task["dataset_class"](num_samples=args.test_samples, seed=123, vocab_size=5)
        else:
            train_raw = task["dataset_class"](num_samples=args.train_samples, seed=42)
            test_raw = task["dataset_class"](num_samples=args.test_samples, seed=123)

        train_ds = MegatronFormalDataset(train_raw, tokenizer, args.seq_length, task_name)
        test_ds = MegatronFormalDataset(test_raw, tokenizer, args.seq_length, task_name)

        # Sampler in DP domain: split across DP ranks but keep same samples within TP group.
        dp_rank = parallel_state.get_data_parallel_rank()
        dp_world = parallel_state.get_data_parallel_world_size()
        train_sampler = (
            DistributedSampler(train_ds, num_replicas=dp_world, rank=dp_rank, shuffle=True, seed=args.seed)
            if dp_world > 1
            else None
        )
        test_sampler = (
            DistributedSampler(test_ds, num_replicas=dp_world, rank=dp_rank, shuffle=False, seed=args.seed)
            if dp_world > 1
            else None
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            drop_last=True,
            num_workers=0,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=test_sampler,
            drop_last=False,
            num_workers=0,
        )

        results = {}
        for m in model_types:
            enable_fan = args.fan_qkv_enabled or (m in ("fanformer", "fanstack"))
            enable_stack = args.stack_memory_enabled or (m in ("stacktrans", "fanstack"))

            if torch.distributed.get_rank() == 0:
                print(f"\n--- {m.upper()} ---")
                print(f"  fan={enable_fan}, stack={enable_stack}")

            # ensure deterministic init per variant
            model_parallel_cuda_manual_seed(args.seed)

            model, config = _build_model(
                vocab_size=vocab_size,
                seq_len=args.seq_length,
                args=args,
                enable_fan=enable_fan,
                enable_stack=enable_stack,
            )

            params = sum(p.numel() for p in model.parameters())
            if torch.distributed.get_rank() == 0:
                print(f"  参数量: {params:,}")

            best_acc, final_acc = _train_one_model(
                model=model,
                config=config,
                train_loader=train_loader,
                test_loader=test_loader,
                task_name=task_name,
                is_classification=task["is_classification"],
                sep_id=sep_id,
                pad_id=pad_id,
                device=device,
                args=args,
            )

            results[m] = {"params": params, "best_acc": best_acc, "final_acc": final_acc}
            del model
            torch.cuda.empty_cache()

        all_results[task_name] = results

    if torch.distributed.get_rank() == 0:
        print("\n" + "=" * 80)
        print("最终结果汇总 (Best Acc)")
        print("=" * 80)
        header = f"{'任务':<25}"
        for m in model_types:
            header += f" {m:<12}"
        print(header)
        print("-" * 80)
        for task_name, results in all_results.items():
            row = f"{TASKS[task_name]['name']:<25}"
            for m in model_types:
                row += f" {results[m]['best_acc']:<12.4f}" if m in results else f" {'N/A':<12}"
            print(row)
        print()
    # Clean shutdown to avoid NCCL warning.
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()


"""

CUDA_VISIBLE_DEVICES=2 torchrun --standalone --nproc_per_node 1 \
  test_formal_language_megatron.py \
  --task cycle --model fanstack \
  --epochs 60 --batch-size 128 --train-samples 5000 --test-samples 1000 \
  --num-layers 4 --hidden-size 128 --ffn-hidden-size 512 --num-attention-heads 4 \
  --stack-memory-num-heads 4 --stack-memory-slots 16 --stack-memory-dim 16 2>&1 | tee training_cycle_fanstack.log

CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node 1 \
  test_formal_language_megatron.py \
  --task cycle --model all \
  --epochs 60 --batch-size 64 --train-samples 5000 --test-samples 1000 \
  --num-layers 4 --hidden-size 128 --ffn-hidden-size 512 --num-attention-heads 4 \
  --stack-memory-num-heads 4 --stack-memory-slots 16 --stack-memory-dim 16 2>&1 | tee training_cycle.log


CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nproc_per_node 1 \
  test_formal_language_megatron.py \
  --task reverse --model all \
  --epochs 60 --batch-size 64 --train-samples 5000 --test-samples 1000 \
  --num-layers 4 --hidden-size 128 --ffn-hidden-size 512 --num-attention-heads 4 \
  --stack-memory-num-heads 4 --stack-memory-slots 16 --stack-memory-dim 16 2>&1 | tee training_reverse.log

"""