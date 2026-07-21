#!/usr/bin/env bash
# =============================================================================
# data_preprocess.sh — 将原始 jsonl 预处理为 Megatron 训练可用的 indexed dataset
#
# 整体流程：
#   原始 jsonl  →  分片（8 份 part_*.jsonl）  →  tokenize  →  .bin + .idx
#
# 用法（在 Megatron-LM 仓库根目录执行）：
#   bash aix_moe_train/data_preprocess.sh
#   DATA_DIR=/path/to/jsonl OUTPUT_DIR=/path/to/out bash aix_moe_train/data_preprocess.sh
#
# 输入：
#   - 原始数据：${DATA_DIR}/*.jsonl，每行 JSON 需含 "text" 字段
#   - 或直接提供已分片文件：${SHARD_DIR}/part_{0..7}.jsonl
#   - Tokenizer：/models/Qwen3-30B-A3B（HuggingFaceTokenizer）
#
# 输出：
#   - 中间分片：${SHARD_DIR}/part_{0..7}.jsonl
#   - 训练数据：${OUTPUT_DIR}/${OUTPUT_PREFIX}_{0..7}_text_document.{bin,idx}
#   - 脚本末尾打印 8 个 --data-path 前缀，供训练脚本直接使用
# =============================================================================

set -euo pipefail

# 可通过环境变量覆盖；未设置时使用下方默认值
DATA_DIR=${DATA_DIR:-/models/nemotron_cc_v2_high_sample_50B/sample_100B_multi_source}  # 原始 jsonl 目录
SHARD_DIR=${SHARD_DIR:-${DATA_DIR}/manual_shards}                                         # 分片 jsonl 输出目录
OUTPUT_DIR=${OUTPUT_DIR:-${DATA_DIR}/outputs}                                             # tokenize 后 .bin/.idx 目录
OUTPUT_PREFIX=${OUTPUT_PREFIX:-aix_sample_100b_multi_source_processed_data}               # 输出文件名前缀

export DATA_DIR SHARD_DIR OUTPUT_DIR OUTPUT_PREFIX

# ---------------------------------------------------------------------------
# 数据分片逻辑（固定 8 片：part_0.jsonl ~ part_7.jsonl，输出到 SHARD_DIR）
#
# 按优先级依次尝试，命中即跳过后续步骤：
#
# 1. 复用已有分片
#    若 SHARD_DIR 下 8 个 part_*.jsonl 均已存在，则直接跳过分片。
#
# 2. 兼容旧版分片（软链接）
#    若 DATA_DIR 下存在 *_0.jsonl ~ *_7.jsonl（glob 匹配），
#    在 SHARD_DIR 中为每个 part_i.jsonl 创建指向对应旧文件的符号链接，不复制数据。
#
# 3. 从原始 jsonl 重新分片（轮询按行分配）
#    扫描 DATA_DIR 下所有 *.jsonl（排除 *_ 前缀的 legacy 文件），按文件名排序后依次读取；
#    采用 round-robin：第 1 行 -> part_0，第 2 行 -> part_1，…，第 8 行 -> part_7，再循环。
#    先写入 .jsonl.tmp 临时文件，全部完成后原子 rename 为 part_*.jsonl，避免中断产生半成品。
#
# 分片完成后，下方循环对每片独立调用 tools/preprocess_data.py 做 tokenize，
# 生成 8 份 *_text_document 前缀的数据，供训练时 --data-path 使用。
# ---------------------------------------------------------------------------

python3 - <<'PY'
import os
import socket
from pathlib import Path

data_dir = Path(os.environ["DATA_DIR"])
shard_dir = Path(os.environ["SHARD_DIR"])
NUM_SHARDS = 8
shards = [shard_dir / f"part_{i}.jsonl" for i in range(NUM_SHARDS)]
legacy_shards = [data_dir / f"*_{i}.jsonl" for i in range(NUM_SHARDS)]

# 策略 1：已有 manual 分片，跳过分片
if all(path.exists() for path in shards):
    print("manual shards already exist, skip splitting")
    raise SystemExit

# 策略 2：旧版 *_i.jsonl 存在，软链接到 part_i.jsonl
if all(path.is_file() for path in legacy_shards):
    shard_dir.mkdir(parents=True, exist_ok=True)
    for legacy_path, shard_path in zip(legacy_shards, shards):
        if shard_path.exists() or shard_path.is_symlink():
            continue
        shard_path.symlink_to(legacy_path)
        print(f"linked {shard_path} -> {legacy_path}")
    raise SystemExit

# 策略 3：收集原始 jsonl，排除 legacy 命名（*_ 前缀）
inputs = sorted(
    path
    for path in data_dir.glob("*.jsonl")
    if not path.name.startswith("*_")
)
if not inputs:
    existing = sorted(path.name for path in data_dir.glob("*")) if data_dir.exists() else []
    raise RuntimeError(
        f"no source jsonl files found under {data_dir}; "
        f"hostname: {socket.gethostname()}; "
        f"data_dir exists: {data_dir.exists()}; "
        f"legacy shards present: {[path.exists() for path in legacy_shards]}; "
        f"first entries: {existing[:20]}"
    )

# round-robin 按行写入 8 个临时文件，完成后原子 rename
shard_dir.mkdir(parents=True, exist_ok=True)
tmp_shards = [shard.with_suffix(".jsonl.tmp") for shard in shards]
writers = [path.open("w", encoding="utf-8") for path in tmp_shards]
try:
    shard_idx = 0
    for input_path in inputs:
        print(f"splitting {input_path}")
        with input_path.open("r", encoding="utf-8") as reader:
            for line in reader:
                writers[shard_idx].write(line)
                shard_idx = (shard_idx + 1) % len(writers)
finally:
    for writer in writers:
        writer.close()

for tmp_path, shard_path in zip(tmp_shards, shards):
    tmp_path.replace(shard_path)
PY

# ---------------------------------------------------------------------------
# Tokenize：对 8 个分片分别调用 Megatron preprocess_data.py
#
# 每片生成：
#   ${OUTPUT_DIR}/${OUTPUT_PREFIX}_${i}_text_document.bin  — token id 二进制数据
#   ${OUTPUT_DIR}/${OUTPUT_PREFIX}_${i}_text_document.idx  — 样本索引
#
# 参数说明：
#   --json-keys text       从 jsonl 每行 JSON 中取 "text" 字段
#   --append-eod/--append-bos  每条样本首尾追加 EOD/BOS token
#   --workers 16           并行 worker 数，可按 CPU 核数调整
# ---------------------------------------------------------------------------
mkdir -p "${OUTPUT_DIR}"

for i in 0 1 2 3 4 5 6 7; do
  python3 tools/preprocess_data.py \
    --input "${SHARD_DIR}/part_${i}.jsonl" \
    --json-keys text \
    --output-prefix "${OUTPUT_PREFIX}_${i}" \
    --output-dir "${OUTPUT_DIR}" \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model /models/Qwen3-30B-A3B \
    --workers 16 \
    --append-eod \
    --append-bos
done

# 打印训练脚本可直接粘贴的 --data-path 前缀（8 个，权重 1:1:...:1）
printf 'Use these prefixes in training --data-path:\\n'
for i in 0 1 2 3 4 5 6 7; do
  printf '  %s/%s_%s_text_document\\n' "${OUTPUT_DIR}" "${OUTPUT_PREFIX}" "${i}"
done

