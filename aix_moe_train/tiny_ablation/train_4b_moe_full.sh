#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Environment variables for performance tuning
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
#export LOG_LEVEL=${LOG_LEVEL:-INFO}
#export NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:-19}
#export NVTE_FWD_LAYERNORM_SM_MARGIN=${NVTE_FWD_LAYERNORM_SM_MARGIN:-16}
#export NVTE_BWD_LAYERNORM_SM_MARGIN=${NVTE_BWD_LAYERNORM_SM_MARGIN:-16}
#export NCCL_P2P_NET_CHUNKSIZE=${NCCL_P2P_NET_CHUNKSIZE:-2097152}
#export NCCL_AVOID_RECORD_STREAMS=${NCCL_AVOID_RECORD_STREAMS:-1}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set +x

ProjectPath="/models/nemotron_cc_v2_high_sample_50B/sample_full"

EXP_NAME="${EXP_NAME:-"4b_moe_bf16_tiny_baseline"}"
CHECKPOINT_PATH=${1:-"${ProjectPath}/tensorboard_logs_tiny/checkpoints/${EXP_NAME}"}
TENSORBOARD_LOGS_PATH=${2:-"${ProjectPath}/tensorboard_logs_tiny/${EXP_NAME}"}
TOKENIZER_ARG=${3:-"/models/Qwen3-30B-A3B"}
DATA_ARG=${4:-"${ProjectPath}/nemotraon_ccv2_sample_full_processed_data_text_document"}
DATA_CACHE_PATH="${ProjectPath}/tensorboard_logs_tiny/data_cache_tiny"
mkdir -p "$DATA_CACHE_PATH"


# Create directories if they don't exist
mkdir -p "$(dirname "$CHECKPOINT_PATH")"
mkdir -p "$(dirname "$TENSORBOARD_LOGS_PATH")"

# Distributed training setup
GPUS_PER_NODE=8
NUM_NODES=${NUM_NODES:-1}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}
NODE_RANK=${NODE_RANK:-0}
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

# Path to the pretrain_gpt.py script, assuming this script is run from the root of the Megatron-LM repository
PRETRAIN_SCRIPT_PATH="pretrain_gpt.py"

# Fixed model and training parameters
TP_SIZE=1
CP_SIZE=1
PP_SIZE=1
MICRO_BATCH_SIZE=2
GLOBAL_BATCH_SIZE=128
NUM_LAYERS=6
DTYPE="bf16"
SEQ_LENGTH=8192
MAX_POSITION_EMBEDDINGS=40960



DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NUM_NODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

MODEL_ARGS=(
    --use-mcore-models
    --num-layers $NUM_LAYERS
    --normalization RMSNorm
    --hidden-size 2048
    --ffn-hidden-size 6144
    --num-attention-heads 16
    --group-query-attention
    --num-query-groups 4
    --kv-channels 128
    --seq-length $SEQ_LENGTH
    --max-position-embeddings $MAX_POSITION_EMBEDDINGS
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

FanStack_ARGS=(
    --log-stack-memory-to-tensorboard
    --stack-memory-enabled
    --stack-memory-slots 6
    --stack-memory-num-heads 8
    --stack-memory-dim 16
)

MOE_ARGS=(
        --moe-grouped-gemm
        --moe-token-dispatcher-type alltoall
        --moe-router-topk 2
        --num-experts 16
        --expert-tensor-parallel-size 1
        --expert-model-parallel-size 1
        --moe-ffn-hidden-size 256
        --moe-router-load-balancing-type aux_loss
        --moe-aux-loss-coeff 0.001
        --moe-layer-freq "([1]*${NUM_LAYERS})"
)

TRAINING_ARGS=(
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE
    --train-samples 5088315
    --lr-decay-samples 1600000
    --lr-warmup-samples 6400
    --lr 0.00002
    --min-lr 0.000001
    --decoupled-lr 2.0e-5      # Specific to decoupled AdamW, ensure optimizer is compatible
    --decoupled-min-lr 4.5e-6  # Specific to decoupled AdamW
    --lr-decay-style cosine
    --clip-grad 1.0
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --bf16
    --grad-reduce-in-bf16
    --cross-entropy-loss-fusion
    --calculate-per-token-loss 
    --manual-gc 
    --empty-unused-memory-level 1 
)


# Model parallelism arguments
MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size $TP_SIZE
    --context-parallel-size $CP_SIZE
    # --pipeline-model-parallel-size $PP_SIZE # Not explicitly set in llama script options, assume 1 if not multi-node PP
    --sequence-parallel  # Always enable sequence parallelism with TP_SIZE=2
    --recompute-activations
)

# Distributed Data Parallel (DDP) arguments
# From original script's ddp_args
DDP_ARGS=(
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
)
TRAINING_ARGS+=("${DDP_ARGS[@]}")


# Data arguments (conditional for mock vs real data)
DATA_ARGS_LIST=()
DATA_ARGS_LIST+=(
    "--data-path $DATA_ARG"
    "--tokenizer-type HuggingFaceTokenizer" 
    "--tokenizer-model $TOKENIZER_ARG"
    "--data-cache-path ${DATA_CACHE_PATH}"
    "--split '98,2,0'"
    "--no-create-attention-mask-in-dataloader"
    "--no-mmap-bin-files"
    "--num-workers 1"
    # Note: --vocab-size might be inferred by HuggingFaceTokenizer or might need to be explicit.
    "--vocab-size 151936"
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 40
    --eval-iters 50
    --eval-interval 1000
    --save-interval 4000
    --log-throughput
    --profile
    --profile-step-start 4
    --profile-step-end 6
    --ckpt-format torch_dist 
    --distributed-timeout-minutes 60
    --save "$CHECKPOINT_PATH"
    --no-save-optim
    --no-save-rng
    --tensorboard-dir "$TENSORBOARD_LOGS_PATH"
)

LOAD_ARGS=(
    --load "$CHECKPOINT_PATH"
    --no-load-optim
    --no-load-rng
)

# Ensure pretrain_gpt.py is found
if [ ! -f "$PRETRAIN_SCRIPT_PATH" ]; then
    echo "Error: pretrain_gpt.py not found at $PRETRAIN_SCRIPT_PATH"
    echo "Please ensure you are running this script from the root of the Megatron-LM repository, and pretrain_gpt.py is present."
    exit 1
fi

# Run the training command
torchrun ${DISTRIBUTED_ARGS[@]} \
    "$PRETRAIN_SCRIPT_PATH" \
    ${MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${FanStack_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS_LIST[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    ${LOAD_ARGS[@]}

