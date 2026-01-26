#!/bin/bash
#
# 多机启动（docker + torchrun）辅助脚本：
# - 通过 SSH 在多台机器上启动同一个 Docker 镜像
# - 给容器注入 NUM_NODES / MASTER_ADDR / MASTER_PORT / NODE_RANK
# - 容器内直接执行 aix_moe_train/train_4b_moe_with_fan_stack.sh
#
# 用法示例（建议在 MASTER_ADDR 所在机器上执行）：
#   bash aix_moe_train/run_multi_node.sh
#   bash aix_moe_train/run_multi_node.sh --stop
#   bash aix_moe_train/run_multi_node.sh --dry-run
#   bash aix_moe_train/run_multi_node.sh --no-train   # 只起容器不跑训练
#
# 需要按实际集群修改变量：
#   SERVERS="10.103.255.4:0 10.103.255.5:1"
#   IMAGE_NAME="nvcr.io/nvidia/pytorch:25.04-py3-megatron-260115"
#   WORKDIR="/nfs100/jiangsiyuan/Megatron-LM"
#   CONTAINER_NAME_PREFIX="jsy_megatron_debug"
#   IB_IFNAME="ibp185s0"   # 或 ib1 等
#
# 说明：
# - MASTER_ADDR 默认取 SERVERS 的第一个 IP
# - MASTER_PORT 默认随机生成（可 export MASTER_PORT 固定）
#

set -euo pipefail

stop="no"
dry_run="no"
no_train="no"

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --stop) stop="yes" ;;
        --dry-run) dry_run="yes" ;;
        --no-train) no_train="yes" ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# -----------------------------
# 用户可配置项（支持环境变量覆盖）
# -----------------------------

# 节点列表：支持 "ip:rank" 或 "ip"（不写 rank 时按顺序自动分配 0..N-1）
SERVERS="${SERVERS:-"10.103.255.2:0 10.103.255.5:1"}"

# Docker 镜像 / 容器工作目录
IMAGE_NAME="${IMAGE_NAME:-"nvcr.io/nvidia/pytorch:25.04-py3-megatron-260115"}"
WORKDIR="${WORKDIR:-"/nfs100/jiangsiyuan/Megatron-LM"}"

# 容器名前缀：每个节点会追加 -r${NODE_RANK}
CONTAINER_NAME_PREFIX="${CONTAINER_NAME_PREFIX:-"megatron_4b_moe_with_fan_stack"}"

# 训练脚本（容器内路径，默认相对 WORKDIR）
TRAIN_SCRIPT_REL="${TRAIN_SCRIPT_REL:-"aix_moe_train/train_4b_moe_with_fan_stack_full.sh"}"


IB_IFNAME="${IB_IFNAME:-"ibp185s0"}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-$IB_IFNAME}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-$IB_IFNAME}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-2}"
export NCCL_DEBUG="${NCCL_DEBUG:-ERROR}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"

# docker/ssh 可选项
DOCKER_BIN="${DOCKER_BIN:-docker}"       # 如果需要 sudo：export DOCKER_BIN="sudo docker"
SSH_BIN="${SSH_BIN:-ssh}"
SSH_OPTS=(
  -oStrictHostKeyChecking=no
  -oUserKnownHostsFile=/dev/null
)

# 挂载（按你单机命令保持一致）
DOCKER_MOUNTS=(
  -v /dev/shm:/dev/shm
  -v /models:/models
  -v /nfs100:/nfs100
  -v /nfsEDS:/nfsEDS
)

# -----------------------------
# 解析 SERVERS -> MASTER_ADDR / NNODES / ranks
# -----------------------------

if [[ -z "${SERVERS// }" ]]; then
  echo "Error: SERVERS is empty."
  exit 1
fi

read -r -a SERVER_ENTRIES <<< "${SERVERS}"
FIRST_ENTRY="${SERVER_ENTRIES[0]}"
# SERVERS 条目允许写成 "user@ip:rank"，这里需要把 master 的实际 IP 提取出来给 torchrun 用
MASTER_ADDR_DEFAULT="${FIRST_ENTRY%%:*}"
MASTER_ADDR_DEFAULT="${MASTER_ADDR_DEFAULT##*@}"
MASTER_ADDR="${MASTER_ADDR:-$MASTER_ADDR_DEFAULT}"

NNODES_DEFAULT="${#SERVER_ENTRIES[@]}"
NUM_NODES="${NUM_NODES:-$NNODES_DEFAULT}"

# 默认随机端口（可 export MASTER_PORT 固定）
if [[ -z "${MASTER_PORT:-}" ]]; then
  MASTER_PORT="$((20000 + (RANDOM % 20000)))"
fi

# 判断 MASTER_ADDR 是否在本机（用于决定 master 节点走本地 docker 还是走 ssh）
MASTER_IS_LOCAL="no"
if command -v ip >/dev/null 2>&1; then
  if ip -o -4 addr show | awk '{print $4}' | cut -d/ -f1 | grep -q "^${MASTER_ADDR}$"; then
    MASTER_IS_LOCAL="yes"
    echo "[OK] MASTER_ADDR(${MASTER_ADDR}) 在本机 IP 列表中"
  else
    echo "[WARN] MASTER_ADDR(${MASTER_ADDR}) 不在本机 IP 列表中（将对 master 也使用 ssh 执行 docker）"
  fi
fi

echo "SERVERS      : ${SERVERS}"
echo "MASTER_ADDR  : ${MASTER_ADDR}"
echo "MASTER_PORT  : ${MASTER_PORT}"
echo "NUM_NODES    : ${NUM_NODES}"
echo "IB_IFNAME    : ${IB_IFNAME}"
echo "IMAGE_NAME   : ${IMAGE_NAME}"
echo "WORKDIR      : ${WORKDIR}"
echo "TRAIN_SCRIPT : ${TRAIN_SCRIPT_REL}"

if [[ "${dry_run}" == "yes" ]]; then
  echo "[DRY-RUN] 仅打印命令，不实际执行"
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

run_remote() {
  local addr="$1"
  local script="$2"
  if [[ "${dry_run}" == "yes" ]]; then
    echo "----- remote(${addr}) -----"
    echo "${script}"
    echo "---------------------------"
    return 0
  fi
  "${SSH_BIN}" "${SSH_OPTS[@]}" "${addr}" bash -s <<EOF
set -euo pipefail
${script}
EOF
}

docker_stop_cmd() {
  local cname="$1"
  cat <<EOS
${DOCKER_BIN} kill ${cname} >/dev/null 2>&1 || true
${DOCKER_BIN} rm -f ${cname} >/dev/null 2>&1 || true
EOS
}

docker_run_cmd() {
  local cname="$1"
  local node_rank="$2"

  # 训练命令：默认在容器内直接跑训练脚本；--no-train 时仅起一个 bash 方便手动进入
  local container_cmd
  if [[ "${no_train}" == "yes" ]]; then
    container_cmd="/bin/bash"
  else
    container_cmd="/bin/bash -lc \"cd ${WORKDIR} && bash ${TRAIN_SCRIPT_REL}\""
  fi

  cat <<EOS
$(docker_stop_cmd "${cname}")
${DOCKER_BIN} run -it -d --gpus all --net=host \\
  --name=${cname} \\
  -w ${WORKDIR} \\
  -e NUM_NODES=${NUM_NODES} \\
  -e MASTER_ADDR=${MASTER_ADDR} \\
  -e MASTER_PORT=${MASTER_PORT} \\
  -e NODE_RANK=${node_rank} \\
  -e NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME} \\
  -e GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME} \\
  -e NCCL_IB_DISABLE=${NCCL_IB_DISABLE} \\
  -e NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL} \\
  -e NCCL_DEBUG=${NCCL_DEBUG} \\
  -e CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS} \\
  ${DOCKER_MOUNTS[*]} \\
  ${IMAGE_NAME} \\
  ${container_cmd}
EOS
}

# -----------------------------
# 主循环：逐节点启动/停止
# -----------------------------

idx=0
for entry in "${SERVER_ENTRIES[@]}"; do
  # 允许 "user@ip:rank"：ssh 用 addr，MASTER_ADDR/比较用 ip
  addr="${entry%%:*}"
  ip="${addr##*@}"
  if [[ "${entry}" == *":"* ]]; then
    rank="${entry##*:}"
  else
    rank="${idx}"
  fi

  cname="${CONTAINER_NAME_PREFIX}-r${rank}"

  if [[ "${stop}" == "yes" ]]; then
    remote_script="$(docker_stop_cmd "${cname}")"
  else
    remote_script="$(docker_run_cmd "${cname}" "${rank}")"
  fi

  if [[ "${ip}" == "${MASTER_ADDR}" ]]; then
    if [[ "${MASTER_IS_LOCAL}" == "yes" ]]; then
      echo "==> [local/master] ${ip} rank=${rank} container=${cname}"
      run_local "${remote_script}"
    else
      echo "==> [remote/master] ${ip} rank=${rank} container=${cname}"
      run_remote "${addr}" "${remote_script}"
    fi
  else
    echo "==> [remote] ${ip} rank=${rank} container=${cname}"
    run_remote "${addr}" "${remote_script}"
  fi

  idx=$((idx + 1))
done

echo "Done."
