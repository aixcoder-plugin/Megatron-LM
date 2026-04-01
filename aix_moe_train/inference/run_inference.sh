#!/bin/bash
#
# Single-machine docker launcher for the Megatron HTTP inference server.
# If you do not need docker, run:
#   bash aix_moe_train/inference/serve_4b_moe_full_new.sh
#
# Usage example:
#   CUDA_VISIBLE_DEVICES=0 \
#   CHECKPOINT_PATH=/mntdata-2/data/checkpoints/4b_moe_bf16_baseline \
#   TOKENIZER_ARG=/models/Qwen3-30B-A3B \
#   bash aix_moe_train/inference/run_inference.sh
#
# Stop the container:
#   bash aix_moe_train/inference/run_inference.sh --stop
#
# Only create a shell container:
#   bash aix_moe_train/inference/run_inference.sh --no-serve

set -euo pipefail

stop="no"
dry_run="no"
no_serve="no"

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --stop) stop="yes" ;;
        --dry-run) dry_run="yes" ;;
        --no-serve|--no-train) no_serve="yes" ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

IMAGE_NAME="${IMAGE_NAME:-"nvcr.io/nvidia/pytorch:25.04-py3-megatron-260115"}"
WORKDIR="${WORKDIR:-"/nfs100/jiangsiyuan/Megatron-LM"}"
CONTAINER_NAME="${CONTAINER_NAME:-${CONTAINER_NAME_PREFIX:-megatron_4b_moe_infer_server}}"
INFER_SCRIPT_REL="${INFER_SCRIPT_REL:-"aix_moe_train/inference/serve_4b_moe_full_new.sh"}"

PROJECT_PATH="${PROJECT_PATH:-/mntdata-2/data}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-"${PROJECT_PATH}/checkpoints/4b_moe_bf16_baseline"}"
TOKENIZER_ARG="${TOKENIZER_ARG:-/models/Qwen3-30B-A3B}"
SERVER_PORT="${SERVER_PORT:-5000}"

MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-6000}"
NUM_NODES="${NUM_NODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

INFERENCE_MAX_BATCH_SIZE="${INFERENCE_MAX_BATCH_SIZE:-1}"
INFERENCE_MAX_SEQ_LENGTH="${INFERENCE_MAX_SEQ_LENGTH:-8192}"

TP_SIZE="${TP_SIZE:-1}"
CP_SIZE="${CP_SIZE:-1}"
PP_SIZE="${PP_SIZE:-1}"
EP_SIZE="${EP_SIZE:-1}"
ETP_SIZE="${ETP_SIZE:-1}"

NUM_LAYERS="${NUM_LAYERS:-24}"
SEQ_LENGTH="${SEQ_LENGTH:-8192}"
MAX_POSITION_EMBEDDINGS="${MAX_POSITION_EMBEDDINGS:-40960}"

TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-false}"
ENABLE_FLASH_DECODE="${ENABLE_FLASH_DECODE:-false}"
ENABLE_CUDA_GRAPH="${ENABLE_CUDA_GRAPH:-false}"

DEFAULT_PROXY_URL="${DEFAULT_PROXY_URL:-http://172.16.200.37:8888}"
export http_proxy="${http_proxy:-${HTTP_PROXY:-${DEFAULT_PROXY_URL}}}"
export https_proxy="${https_proxy:-${HTTPS_PROXY:-${DEFAULT_PROXY_URL}}}"
export HTTP_PROXY="${HTTP_PROXY:-${http_proxy}}"
export HTTPS_PROXY="${HTTPS_PROXY:-${https_proxy}}"
export no_proxy="${no_proxy:-localhost,127.0.0.1,::1}"
export NO_PROXY="${NO_PROXY:-${no_proxy}}"

export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-pypi.tuna.tsinghua.edu.cn}"
export AUTO_INSTALL_PYTHON_PACKAGES="${AUTO_INSTALL_PYTHON_PACKAGES:-true}"

IB_IFNAME="${IB_IFNAME:-"ibp185s0"}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-$IB_IFNAME}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-$IB_IFNAME}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-2}"
export NCCL_DEBUG="${NCCL_DEBUG:-ERROR}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"

DOCKER_BIN="${DOCKER_BIN:-docker}"
DOCKER_MOUNTS=(
  -v /dev/shm:/dev/shm
  -v /models:/models
  -v /mntdata-2:/mntdata-2
  -v /nfs100:/nfs100
  -v /nfsEDS:/nfsEDS
)

echo "CONTAINER_NAME           : ${CONTAINER_NAME}"
echo "CHECKPOINT_PATH          : ${CHECKPOINT_PATH}"
echo "TOKENIZER_ARG            : ${TOKENIZER_ARG}"
echo "SERVER_PORT              : ${SERVER_PORT}"
echo "CUDA_VISIBLE_DEVICES     : ${CUDA_VISIBLE_DEVICES}"
echo "GPUS_PER_NODE            : ${GPUS_PER_NODE}"
echo "PARALLEL                 : TP=${TP_SIZE} PP=${PP_SIZE} CP=${CP_SIZE} EP=${EP_SIZE} ETP=${ETP_SIZE}"
echo "HTTPS_PROXY              : ${https_proxy}"
echo "PIP_INDEX_URL            : ${PIP_INDEX_URL}"
echo "IMAGE_NAME               : ${IMAGE_NAME}"
echo "WORKDIR                  : ${WORKDIR}"

if [[ "${dry_run}" == "yes" ]]; then
  echo "[DRY-RUN] commands will only be printed"
fi

run_local() {
  local cmd="$1"
  if [[ "${dry_run}" == "yes" ]]; then
    echo "----- local cmd -----"
    echo "${cmd}"
    echo "---------------------"
    return 0
  fi
  bash -lc "${cmd}"
}

docker_stop_cmd() {
  cat <<EOS
${DOCKER_BIN} kill ${CONTAINER_NAME} >/dev/null 2>&1 || true
${DOCKER_BIN} rm -f ${CONTAINER_NAME} >/dev/null 2>&1 || true
EOS
}

docker_run_cmd() {
  local container_cmd
  if [[ "${no_serve}" == "yes" ]]; then
    container_cmd="/bin/bash"
  else
    container_cmd="/bin/bash -lc \"cd ${WORKDIR} && bash ${INFER_SCRIPT_REL}\""
  fi

  cat <<EOS
$(docker_stop_cmd)
${DOCKER_BIN} run -it -d --gpus all --net=host \\
  --name=${CONTAINER_NAME} \\
  -w ${WORKDIR} \\
  -e CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \\
  -e NUM_NODES=${NUM_NODES} \\
  -e MASTER_ADDR=${MASTER_ADDR} \\
  -e MASTER_PORT=${MASTER_PORT} \\
  -e NODE_RANK=${NODE_RANK} \\
  -e GPUS_PER_NODE=${GPUS_PER_NODE} \\
  -e NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME} \\
  -e GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME} \\
  -e NCCL_IB_DISABLE=${NCCL_IB_DISABLE} \\
  -e NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL} \\
  -e NCCL_DEBUG=${NCCL_DEBUG} \\
  -e CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS} \\
  -e PROJECT_PATH=${PROJECT_PATH} \\
  -e CHECKPOINT_PATH=${CHECKPOINT_PATH} \\
  -e TOKENIZER_ARG=${TOKENIZER_ARG} \\
  -e SERVER_PORT=${SERVER_PORT} \\
  -e DEFAULT_PROXY_URL=${DEFAULT_PROXY_URL} \\
  -e http_proxy=${http_proxy} \\
  -e https_proxy=${https_proxy} \\
  -e HTTP_PROXY=${HTTP_PROXY} \\
  -e HTTPS_PROXY=${HTTPS_PROXY} \\
  -e no_proxy=${no_proxy} \\
  -e NO_PROXY=${NO_PROXY} \\
  -e PIP_INDEX_URL=${PIP_INDEX_URL} \\
  -e PIP_TRUSTED_HOST=${PIP_TRUSTED_HOST} \\
  -e AUTO_INSTALL_PYTHON_PACKAGES=${AUTO_INSTALL_PYTHON_PACKAGES} \\
  -e INFERENCE_MAX_BATCH_SIZE=${INFERENCE_MAX_BATCH_SIZE} \\
  -e INFERENCE_MAX_SEQ_LENGTH=${INFERENCE_MAX_SEQ_LENGTH} \\
  -e TP_SIZE=${TP_SIZE} \\
  -e CP_SIZE=${CP_SIZE} \\
  -e PP_SIZE=${PP_SIZE} \\
  -e EP_SIZE=${EP_SIZE} \\
  -e ETP_SIZE=${ETP_SIZE} \\
  -e NUM_LAYERS=${NUM_LAYERS} \\
  -e SEQ_LENGTH=${SEQ_LENGTH} \\
  -e MAX_POSITION_EMBEDDINGS=${MAX_POSITION_EMBEDDINGS} \\
  -e TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE} \\
  -e ENABLE_FLASH_DECODE=${ENABLE_FLASH_DECODE} \\
  -e ENABLE_CUDA_GRAPH=${ENABLE_CUDA_GRAPH} \\
  ${DOCKER_MOUNTS[*]} \\
  ${IMAGE_NAME} \\
  ${container_cmd}
EOS
}

if [[ "${stop}" == "yes" ]]; then
  run_local "$(docker_stop_cmd)"
else
  run_local "$(docker_run_cmd)"
fi

echo "Done."
