#!/bin/bash
#
# Start a Megatron HTTP inference server for the 4B MoE + FAN checkpoint.
# The overall structure follows Megatron's native examples under:
#   examples/inference/
# and uses the official entrypoint:
#   tools/run_text_generation_server.py
#
# Usage:
#   # Single GPU, local process.
#   CUDA_VISIBLE_DEVICES=0 \
#   https_proxy=http://172.16.200.37:8888 \
#   bash aix_moe_train/inference/serve_4b_moe_full_new.sh \
#     [CHECKPOINT_PATH] \
#     [TOKENIZER_PATH] \
#     [SERVER_PORT]
#
#   # Single machine, multi-GPU example.
#   CUDA_VISIBLE_DEVICES=0,1,2,3 \
#   GPUS_PER_NODE=4 TP_SIZE=4 EP_SIZE=1 ETP_SIZE=1 \
#   bash aix_moe_train/inference/serve_4b_moe_full_new.sh \
#     [CHECKPOINT_PATH] \
#     [TOKENIZER_PATH] \
#     [SERVER_PORT] \
#     [EXTRA_MEGATRON_ARGS...]
#
# The HTTP endpoint will be:
#   http://<MASTER_ADDR>:<SERVER_PORT>/api

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

DEFAULT_PROXY_URL="${DEFAULT_PROXY_URL:-http://172.16.200.37:8888}"
export http_proxy="${http_proxy:-${HTTP_PROXY:-${DEFAULT_PROXY_URL}}}"
export https_proxy="${https_proxy:-${HTTPS_PROXY:-${DEFAULT_PROXY_URL}}}"
export HTTP_PROXY="${HTTP_PROXY:-${http_proxy}}"
export HTTPS_PROXY="${HTTPS_PROXY:-${https_proxy}}"
export no_proxy="${no_proxy:-localhost,127.0.0.1,::1}"
export NO_PROXY="${NO_PROXY:-${no_proxy}}"

PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-pypi.tuna.tsinghua.edu.cn}"
AUTO_INSTALL_PYTHON_PACKAGES="${AUTO_INSTALL_PYTHON_PACKAGES:-true}"

PROJECT_PATH="${PROJECT_PATH:-/mntdata-2/data}"
DEFAULT_CHECKPOINT_PATH="${PROJECT_PATH}/checkpoints/4b_moe_bf16_baseline"

CHECKPOINT_PATH="${1:-${CHECKPOINT_PATH:-${DEFAULT_CHECKPOINT_PATH}}}"
TOKENIZER_ARG="${2:-${TOKENIZER_ARG:-/models/Qwen3-30B-A3B}}"
SERVER_PORT="${3:-${SERVER_PORT:-5000}}"
USER_ARGS=("${@:4}")

GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
NUM_NODES="${NUM_NODES:-1}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-6000}"
NODE_RANK="${NODE_RANK:-0}"

# Megatron distributed checkpoints support resharding across different
# TP/PP/EP/ETP configurations in its test coverage, so we default to the
# simplest local setup first and only scale out when needed.
TP_SIZE="${TP_SIZE:-1}"
CP_SIZE="${CP_SIZE:-1}"
PP_SIZE="${PP_SIZE:-1}"
EP_SIZE="${EP_SIZE:-1}"
ETP_SIZE="${ETP_SIZE:-1}"

INFERENCE_MAX_BATCH_SIZE="${INFERENCE_MAX_BATCH_SIZE:-1}"
INFERENCE_MAX_SEQ_LENGTH="${INFERENCE_MAX_SEQ_LENGTH:-8192}"

NUM_LAYERS="${NUM_LAYERS:-24}"
SEQ_LENGTH="${SEQ_LENGTH:-8192}"
MAX_POSITION_EMBEDDINGS="${MAX_POSITION_EMBEDDINGS:-40960}"

PRETRAIN_SCRIPT_PATH="${REPO_ROOT}/aix_moe_train/inference/run_text_generation_server_local.py"

if [ ! -f "${PRETRAIN_SCRIPT_PATH}" ]; then
    echo "Error: ${PRETRAIN_SCRIPT_PATH} not found."
    exit 1
fi

if [ ! -d "${CHECKPOINT_PATH}" ]; then
    echo "Error: checkpoint directory not found: ${CHECKPOINT_PATH}"
    exit 1
fi

if [ ! -f "${CHECKPOINT_PATH}/latest_checkpointed_iteration.txt" ]; then
    echo "Error: ${CHECKPOINT_PATH} does not look like a Megatron checkpoint directory."
    echo "Missing file: ${CHECKPOINT_PATH}/latest_checkpointed_iteration.txt"
    exit 1
fi

is_true() {
    case "${1,,}" in
        1|true|yes|y|on) return 0 ;;
        *) return 1 ;;
    esac
}

MISSING_PIP_PACKAGES="$(
python - <<'PY'
import importlib.util

module_to_package = {
    "flask": "flask",
    "flask_restful": "flask-restful",
}
missing = [
    package_name
    for module_name, package_name in module_to_package.items()
    if importlib.util.find_spec(module_name) is None
]
print(" ".join(missing))
PY
)"

if [ -n "${MISSING_PIP_PACKAGES}" ]; then
    if is_true "${AUTO_INSTALL_PYTHON_PACKAGES}"; then
        echo "Installing missing Python packages via Tsinghua mirror: ${MISSING_PIP_PACKAGES}"
        echo "  https_proxy=${https_proxy}"
        python -m pip install \
            --disable-pip-version-check \
            --retries 5 \
            --timeout 120 \
            -i "${PIP_INDEX_URL}" \
            --trusted-host "${PIP_TRUSTED_HOST}" \
            ${MISSING_PIP_PACKAGES}
    else
        echo "Missing python packages required by Megatron inference server: ${MISSING_PIP_PACKAGES}"
        echo "Enable AUTO_INSTALL_PYTHON_PACKAGES=true or install them manually, for example:"
        echo "  export https_proxy=${DEFAULT_PROXY_URL}"
        echo "  python -m pip install -i ${PIP_INDEX_URL} --trusted-host ${PIP_TRUSTED_HOST} ${MISSING_PIP_PACKAGES}"
        exit 1
    fi
fi

OPTIONAL_ARGS=()
if is_true "${TRUST_REMOTE_CODE:-false}"; then
    OPTIONAL_ARGS+=(--trust-remote-code)
fi
if is_true "${ENABLE_FLASH_DECODE:-false}"; then
    OPTIONAL_ARGS+=(--flash-decode)
fi
if is_true "${ENABLE_CUDA_GRAPH:-false}"; then
    OPTIONAL_ARGS+=(--enable-cuda-graph)
fi

DISTRIBUTED_ARGS=(
    --nproc_per_node "${GPUS_PER_NODE}"
    --nnodes "${NUM_NODES}"
    --node_rank "${NODE_RANK}"
    --master_addr "${MASTER_ADDR}"
    --master_port "${MASTER_PORT}"
)

MODEL_ARGS=(
    --use-mcore-models
    --transformer-impl transformer_engine
    --num-layers "${NUM_LAYERS}"
    --normalization RMSNorm
    --hidden-size 2048
    --ffn-hidden-size 6144
    --num-attention-heads 16
    --group-query-attention
    --num-query-groups 4
    --kv-channels 128
    --seq-length "${SEQ_LENGTH}"
    --max-position-embeddings "${MAX_POSITION_EMBEDDINGS}"
    --position-embedding-type rope
    --rotary-base 1000000
    --rotary-percent 1.0
    --attention-dropout 0.0
    --hidden-dropout 0.02
    --swiglu
    --init-method-std 0.0134
    --attention-backend fused
    --qk-layernorm
    --disable-bias-linear
)

FAN_ARGS=(
)

MOE_ARGS=(
    --moe-grouped-gemm
    --moe-token-dispatcher-type alltoall
    --moe-router-topk 2
    --num-experts 64
    --expert-tensor-parallel-size "${ETP_SIZE}"
    --expert-model-parallel-size "${EP_SIZE}"
    --moe-ffn-hidden-size 384
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 0.001
    --moe-layer-freq "([1]*${NUM_LAYERS})"
)

INFERENCE_ARGS=(
    --micro-batch-size 1
    --global-batch-size 1
    --bf16
    --tensor-model-parallel-size "${TP_SIZE}"
    --pipeline-model-parallel-size "${PP_SIZE}"
    --context-parallel-size "${CP_SIZE}"
    --sequence-parallel
    --distributed-timeout-minutes 60
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model "${TOKENIZER_ARG}"
    --load "${CHECKPOINT_PATH}"
    --no-load-optim
    --no-load-rng
    --exit-on-missing-checkpoint
    --inference-max-batch-size "${INFERENCE_MAX_BATCH_SIZE}"
    --inference-max-seq-length "${INFERENCE_MAX_SEQ_LENGTH}"
    --port "${SERVER_PORT}"
)

# Megatron's official examples often use `--use-checkpoint-args`, but this
# model depends on custom FAN and parallel arguments that are not fully
# restored by checkpoint arg loading, so we keep the structural arguments
# explicit here.

echo "Starting Megatron inference server"
echo "  repo root  : ${REPO_ROOT}"
echo "  checkpoint : ${CHECKPOINT_PATH}"
echo "  tokenizer  : ${TOKENIZER_ARG}"
echo "  endpoint   : http://${MASTER_ADDR}:${SERVER_PORT}/api"
echo "  dist       : nodes=${NUM_NODES} node_rank=${NODE_RANK} gpus_per_node=${GPUS_PER_NODE}"
echo "  parallel   : TP=${TP_SIZE} PP=${PP_SIZE} CP=${CP_SIZE} EP=${EP_SIZE} ETP=${ETP_SIZE}"
echo "  cache      : max_batch=${INFERENCE_MAX_BATCH_SIZE} max_seq=${INFERENCE_MAX_SEQ_LENGTH}"
echo "  proxy      : ${https_proxy}"
echo "  pip index  : ${PIP_INDEX_URL}"

if [ "${GPUS_PER_NODE}" -gt 1 ] && [ "${TP_SIZE}" -eq 1 ] && [ "${PP_SIZE}" -eq 1 ] && [ "${EP_SIZE}" -eq 1 ] && [ "${ETP_SIZE}" -eq 1 ]; then
    echo "  note       : multiple GPU processes requested, but all model-parallel sizes are 1."
    echo "               If you only want one GPU, keep GPUS_PER_NODE=1."
    echo "               If you want model parallel inference, set TP/PP/EP/ETP explicitly."
fi

torchrun "${DISTRIBUTED_ARGS[@]}" \
    "${PRETRAIN_SCRIPT_PATH}" \
    "${MODEL_ARGS[@]}" \
    "${FAN_ARGS[@]}" \
    "${MOE_ARGS[@]}" \
    "${INFERENCE_ARGS[@]}" \
    "${OPTIONAL_ARGS[@]}" \
    "${USER_ARGS[@]}"
