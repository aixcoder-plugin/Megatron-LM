#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 formal language 训练日志中抽取 curves 并绘图（同任务多配置对比）。

支持的关键日志行（示例）：
  任务: Cycle Navigation (RE)
  参数量: 858,936
  Epoch   1 | Step     50 | Loss 1.2145
  [Eval] Epoch   1 | Acc 0.2030 | Best 0.2030

用法示例：
  python aix_moe_train/test_formal_language/plot_from_logs.py \
    /nfs100/jiangsiyuan/Megatron-LM/aix_moe_train/test_formal_language/training_cycle.log \
    /nfs100/jiangsiyuan/Megatron-LM/aix_moe_train/test_formal_language/training_cycle_fan.log \
    /nfs100/jiangsiyuan/Megatron-LM/aix_moe_train/test_formal_language/training_cycle_stack.log \
    --output-dir /nfs100/jiangsiyuan/Megatron-LM/aix_moe_train/test_formal_language/plots

python aix_moe_train/test_formal_language/plot_from_logs.py /nfs100/jiangsiyuan/Megatron-LM/aix_moe_train/test_formal_language/training_reverse.log \
    --output-dir /nfs100/jiangsiyuan/Megatron-LM/aix_moe_train/test_formal_language/plots_multi/

python aix_moe_train/test_formal_language/plot_from_logs.py /nfs100/jiangsiyuan/Megatron-LM/aix_moe_train/test_formal_language/training_cycle.log \
    --output-dir /nfs100/jiangsiyuan/Megatron-LM/aix_moe_train/test_formal_language/plots_multi/
"""

from __future__ import annotations

import argparse
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib


matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402


MODEL_COLORS = {
    # canonical names
    "transformer": "#1f77b4",
    "fanformer": "#ff7f0e",
    "stacktrans": "#2ca02c",
    "fanstack": "#d62728",
    "both": "#9467bd",
    # backward-compatible aliases
    "base": "#1f77b4",
    "fan": "#ff7f0e",
    "stack": "#2ca02c",
}

MODEL_MARKERS = {
    # canonical names
    "transformer": "o",
    "fanformer": "s",
    "stacktrans": "^",
    "fanstack": "D",
    "both": "X",
    # backward-compatible aliases
    "base": "o",
    "fan": "s",
    "stack": "^",
}


RE_TASK = re.compile(r"^\s*任务:\s*(?P<title>.+?)\s*$")
RE_GROUP = re.compile(r"^\s*---\s*(?P<group>.+?)\s*---\s*$")
RE_PARAMS = re.compile(r"^\s*参数量:\s*(?P<params>[\d,]+)\s*$")
RE_TRAIN = re.compile(
    r"^\s*Epoch\s+(?P<epoch>\d+)\s*\|\s*Step\s+\d+\s*\|\s*Loss\s+(?P<loss>[0-9]*\.?[0-9]+)\s*$"
)
RE_EVAL = re.compile(
    r"^\s*\[Eval\]\s*Epoch\s+(?P<epoch>\d+)\s*\|\s*Acc\s+(?P<acc>[0-9]*\.?[0-9]+)\s*\|.*$"
)


@dataclass
class ParsedRun:
    task_title: str
    params: Optional[int]
    train_loss_by_epoch: Dict[int, float]
    test_acc_by_epoch: Dict[int, float]


@dataclass
class ParsedGroupRun:
    task_title: str
    group_name: Optional[str]  # 来自 --- GROUP --- 的名字（已归一化）；可能为 None
    params: Optional[int]
    train_loss_by_epoch: Dict[int, float]
    test_acc_by_epoch: Dict[int, float]


def _safe_int_with_commas(s: str) -> Optional[int]:
    s = s.strip()
    if not s:
        return None
    try:
        return int(s.replace(",", ""))
    except ValueError:
        return None


def normalize_name(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", "", s)
    s = s.replace("-", "")
    return s


def canonical_model_type(model_type: str) -> str:
    """把各种别名统一成 canonical 名称，方便图例/配色一致。"""
    mt = normalize_name(model_type)
    aliases = {
        "base": "transformer",
        "transformer": "transformer",
        "fan": "fanformer",
        "fanformer": "fanformer",
        "stack": "stacktrans",
        "stacktrans": "stacktrans",
        "fanstack": "fanstack",
        "both": "both",
    }
    return aliases.get(mt, mt)


def infer_task_and_model_from_filename(log_path: str) -> Tuple[str, str]:
    """
    从文件名推断 task_key 和 model_type。

    规则（默认，统一成 canonical 名称）：
      - training_cycle.log -> (cycle, transformer)
      - training_cycle_fan.log -> (cycle, fanformer)
      - training_cycle_stack.log -> (cycle, stacktrans)
      - training_cycle_fanstack.log -> (cycle, fanstack)
    """
    base = os.path.basename(log_path)
    stem, _ = os.path.splitext(base)

    for prefix in ("training_", "train_", "log_", "output_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix) :]
            break

    parts = stem.split("_")
    known_models = {"base", "fan", "stack", "fanstack", "both", "transformer", "fanformer", "stacktrans"}

    if len(parts) >= 3 and parts[-2:] == ["fan", "stack"]:
        # 兼容 training_xxx_fan_stack.log
        return "_".join(parts[:-2]), "fanstack"

    if len(parts) >= 2 and parts[-1] in known_models:
        task_key = "_".join(parts[:-1])
        model_type = canonical_model_type(parts[-1])
        return task_key, model_type

    # 没有显式后缀时，默认当作 base
    return stem, "transformer"


def parse_single_task_log(log_path: str) -> Optional[ParsedRun]:
    """
    解析一个日志文件（默认认为只包含一个任务的训练过程）。
    如果日志为空/不含任务字段，则返回 None。

    训练 loss 可能同一 epoch 多次打印：这里默认取该 epoch 最后一条 loss。
    """
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.read().splitlines()
    except OSError:
        return None

    task_title: Optional[str] = None
    params: Optional[int] = None
    train_loss_by_epoch: Dict[int, float] = {}
    test_acc_by_epoch: Dict[int, float] = {}

    for line in lines:
        if task_title is None:
            m = RE_TASK.match(line)
            if m:
                task_title = m.group("title").strip()
            continue

        # 已进入任务段：解析 params / train / eval
        m = RE_PARAMS.match(line)
        if m and params is None:
            params = _safe_int_with_commas(m.group("params"))
            continue

        m = RE_TRAIN.match(line)
        if m:
            epoch = int(m.group("epoch"))
            loss = float(m.group("loss"))
            train_loss_by_epoch[epoch] = loss  # overwrite -> last
            continue

        m = RE_EVAL.match(line)
        if m:
            epoch = int(m.group("epoch"))
            acc = float(m.group("acc"))
            test_acc_by_epoch[epoch] = acc  # overwrite -> last
            continue

    if not task_title:
        return None

    return ParsedRun(
        task_title=task_title,
        params=params,
        train_loss_by_epoch=train_loss_by_epoch,
        test_acc_by_epoch=test_acc_by_epoch,
    )


def parse_log_with_groups(log_path: str) -> List[ParsedGroupRun]:
    """
    解析一个日志文件，支持：
      - 同一文件包含多个实验段（通过 --- GROUP --- 分隔）
      - 同一文件包含多个任务段（通过 任务: 分隔）

    训练 loss 可能同一 epoch 多次打印：默认取该 epoch 最后一条 loss。
    """
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.read().splitlines()
    except OSError:
        return []

    runs: List[ParsedGroupRun] = []

    task_title: Optional[str] = None
    group_name: Optional[str] = None
    params: Optional[int] = None
    train_loss_by_epoch: Dict[int, float] = {}
    test_acc_by_epoch: Dict[int, float] = {}

    def flush() -> None:
        nonlocal group_name, params, train_loss_by_epoch, test_acc_by_epoch
        if task_title and (train_loss_by_epoch or test_acc_by_epoch):
            runs.append(
                ParsedGroupRun(
                    task_title=task_title,
                    group_name=group_name,
                    params=params,
                    train_loss_by_epoch=train_loss_by_epoch,
                    test_acc_by_epoch=test_acc_by_epoch,
                )
            )
        group_name = None
        params = None
        train_loss_by_epoch = {}
        test_acc_by_epoch = {}

    for line in lines:
        m = RE_TASK.match(line)
        if m:
            # 新任务开始：先把上一段写出
            flush()
            task_title = m.group("title").strip()
            continue

        if task_title is None:
            continue

        m = RE_GROUP.match(line)
        if m:
            # 新实验组开始：把上一组写出
            flush()
            group_name = canonical_model_type(m.group("group"))
            continue

        # 已进入任务段：如果没遇到 group header，但出现了数据，也当作单组
        if group_name is None:
            group_name = None  # 让外层逻辑用文件名推断的 model_type

        m = RE_PARAMS.match(line)
        if m and params is None:
            params = _safe_int_with_commas(m.group("params"))
            continue

        m = RE_TRAIN.match(line)
        if m:
            epoch = int(m.group("epoch"))
            loss = float(m.group("loss"))
            train_loss_by_epoch[epoch] = loss
            continue

        m = RE_EVAL.match(line)
        if m:
            epoch = int(m.group("epoch"))
            acc = float(m.group("acc"))
            test_acc_by_epoch[epoch] = acc
            continue

    flush()
    return runs


def build_history(parsed: ParsedRun) -> Tuple[List[float], List[float], Optional[int]]:
    """把 epoch->value 的 dict 变成按 epoch=1..max_epoch 对齐的 list（缺失填 NaN）。"""
    max_epoch = 0
    if parsed.train_loss_by_epoch:
        max_epoch = max(max_epoch, max(parsed.train_loss_by_epoch.keys()))
    if parsed.test_acc_by_epoch:
        max_epoch = max(max_epoch, max(parsed.test_acc_by_epoch.keys()))

    if max_epoch <= 0:
        return [], [], parsed.params

    train_losses = [float("nan")] * max_epoch
    test_accs = [float("nan")] * max_epoch

    for e, v in parsed.train_loss_by_epoch.items():
        if 1 <= e <= max_epoch:
            train_losses[e - 1] = float(v)
    for e, v in parsed.test_acc_by_epoch.items():
        if 1 <= e <= max_epoch:
            test_accs[e - 1] = float(v)

    return train_losses, test_accs, parsed.params


def _finite_xy(values: List[float]) -> Tuple[List[int], List[float]]:
    xs: List[int] = []
    ys: List[float] = []
    for i, v in enumerate(values, start=1):
        if isinstance(v, float) and math.isfinite(v):
            xs.append(i)
            ys.append(v)
    return xs, ys


def plot_task_loss(
    task_key: str,
    task_title: str,
    histories: Dict[str, Dict[str, object]],
    output_dir: str,
) -> str:
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(8.5, 5.0))
    ax = plt.gca()

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    for idx, (model_type, hist) in enumerate(histories.items()):
        train_losses = hist["train_losses"]  # type: ignore[assignment]
        params = hist.get("params")  # type: ignore[assignment]

        xs, ys = _finite_xy(train_losses)  # type: ignore[arg-type]
        if not xs:
            continue

        color = MODEL_COLORS.get(model_type) or (color_cycle[idx % len(color_cycle)] if color_cycle else None)
        marker = MODEL_MARKERS.get(model_type, "o")
        label = f"{model_type}" + (f" ({int(params):,} params)" if isinstance(params, int) else "")

        ax.plot(
            xs,
            ys,
            color=color,
            marker=marker,
            markevery=max(1, len(xs) // 10),
            label=label,
            linewidth=2,
            markersize=6,
        )

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Training Loss", fontsize=12)
    ax.set_title(f"{task_title} - Training Loss", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()

    filename = os.path.join(output_dir, f"{task_key}_train_loss.png")
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    return filename


def plot_task_acc(
    task_key: str,
    task_title: str,
    histories: Dict[str, Dict[str, object]],
    output_dir: str,
) -> str:
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(8.5, 5.0))
    ax = plt.gca()

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    for idx, (model_type, hist) in enumerate(histories.items()):
        test_accs = hist["test_accs"]  # type: ignore[assignment]

        xs, ys = _finite_xy(test_accs)  # type: ignore[arg-type]
        if not xs:
            continue

        color = MODEL_COLORS.get(model_type) or (color_cycle[idx % len(color_cycle)] if color_cycle else None)
        marker = MODEL_MARKERS.get(model_type, "o")

        ax.plot(
            xs,
            ys,
            color=color,
            marker=marker,
            markevery=max(1, len(xs) // 10),
            label=f"{model_type}",
            linewidth=2,
            markersize=6,
        )

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_title(f"{task_title} - Test Accuracy", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()

    filename = os.path.join(output_dir, f"{task_key}_test_acc.png")
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    return filename


def extract_histories_from_logs(
    log_paths: List[str],
    labels: Optional[List[str]] = None,
) -> Dict[str, Tuple[str, Dict[str, Dict[str, object]]]]:
    """
    返回结构：
      all_tasks[task_key] = (task_title, histories)
      histories[model_type] = {'train_losses': [...], 'test_accs': [...], 'params': int|None}
    """
    if labels is not None and len(labels) != len(log_paths):
        raise ValueError("labels 数量必须与 log_paths 数量一致")

    all_tasks: Dict[str, Tuple[str, Dict[str, Dict[str, object]]]] = {}

    for i, log_path in enumerate(log_paths):
        file_task_key, inferred_model = infer_task_and_model_from_filename(log_path)
        inferred_model = canonical_model_type(inferred_model)
        file_label = canonical_model_type(labels[i]) if labels is not None else inferred_model

        runs = parse_log_with_groups(log_path)
        if not runs:
            # 兼容旧逻辑：有些日志可能没有 group header 且 parse 失败
            parsed = parse_single_task_log(log_path)
            if parsed is None:
                continue
            runs = [
                ParsedGroupRun(
                    task_title=parsed.task_title,
                    group_name=None,
                    params=parsed.params,
                    train_loss_by_epoch=parsed.train_loss_by_epoch,
                    test_acc_by_epoch=parsed.test_acc_by_epoch,
                )
            ]

        # 按 task_title 分组（同一文件可能包含多个任务）
        by_task: Dict[str, List[ParsedGroupRun]] = {}
        for r in runs:
            by_task.setdefault(r.task_title, []).append(r)

        # 如果文件只包含一个任务，用文件名推断的 task_key；否则用 title slug（避免多个任务挤到一个 key）
        if len(by_task) == 1:
            task_key_map = {next(iter(by_task.keys())): file_task_key}
        else:
            task_key_map = {t: normalize_name(t) for t in by_task.keys()}

        for task_title, task_runs in by_task.items():
            task_key = task_key_map[task_title]

            if task_key not in all_tasks:
                all_tasks[task_key] = (task_title, {})
            histories = all_tasks[task_key][1]

            # 多组实验：优先使用组名；单组：使用文件名推断/--label
            if len(task_runs) == 1:
                r = task_runs[0]
                train_losses, test_accs, params = build_history(
                    ParsedRun(
                        task_title=r.task_title,
                        params=r.params,
                        train_loss_by_epoch=r.train_loss_by_epoch,
                        test_acc_by_epoch=r.test_acc_by_epoch,
                    )
                )
                if not train_losses and not test_accs:
                    continue
                histories[file_label] = {"train_losses": train_losses, "test_accs": test_accs, "params": params}
            else:
                if labels is not None:
                    print(f"警告: {log_path} 包含多组实验，--label 将被忽略（改用日志内的 --- GROUP --- 名称）。")
                for r in task_runs:
                    mt = canonical_model_type(r.group_name or "unknown")
                    train_losses, test_accs, params = build_history(
                        ParsedRun(
                            task_title=r.task_title,
                            params=r.params,
                            train_loss_by_epoch=r.train_loss_by_epoch,
                            test_acc_by_epoch=r.test_acc_by_epoch,
                        )
                    )
                    if not train_losses and not test_accs:
                        continue
                    histories[mt] = {"train_losses": train_losses, "test_accs": test_accs, "params": params}

    return all_tasks


def main() -> int:
    parser = argparse.ArgumentParser(description="从 formal language 日志抽取曲线并绘图（同任务多配置对比）")
    parser.add_argument("logs", nargs="+", help="一个或多个 .log 文件路径")
    parser.add_argument("--output-dir", default="plots", help="输出目录（默认：plots）")
    parser.add_argument(
        "--label",
        action="append",
        default=None,
        help="为每个 log 指定一个 model_type（可重复多次；数量需与 logs 一致）。不提供则从文件名推断。",
    )
    parser.add_argument(
        "--only-task",
        default=None,
        help="只绘制指定 task_key（例如 cycle / reverse）。默认绘制所有解析到的任务。",
    )
    args = parser.parse_args()

    all_tasks = extract_histories_from_logs(args.logs, labels=args.label)
    if not all_tasks:
        print("未从日志中解析到任何可绘制的数据（可能日志为空或格式不匹配）。")
        return 1

    for task_key, (task_title, histories) in all_tasks.items():
        if args.only_task and task_key != args.only_task:
            continue
        if not histories:
            continue

        loss_path = plot_task_loss(task_key, task_title, histories, args.output_dir)
        acc_path = plot_task_acc(task_key, task_title, histories, args.output_dir)

        print(f"\n任务: {task_key} | {task_title}")
        print(f"  - Loss 图: {loss_path}")
        print(f"  - Acc  图: {acc_path}")
        print("  - histories keys:", ", ".join(histories.keys()))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

