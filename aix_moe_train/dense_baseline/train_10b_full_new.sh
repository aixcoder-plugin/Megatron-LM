#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Environment variables for performance tuning
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
#export LOG_LEVEL=${LOG_LEVEL:-INFO}
# export NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:-22}
#export NVTE_FWD_LAYERNORM_SM_MARGIN=${NVTE_FWD_LAYERNORM_SM_MARGIN:-16}
#export NVTE_BWD_LAYERNORM_SM_MARGIN=${NVTE_BWD_LAYERNORM_SM_MARGIN:-16}
# export NCCL_P2P_NET_CHUNKSIZE=${NCCL_P2P_NET_CHUNKSIZE:-2097152}
#export NCCL_AVOID_RECORD_STREAMS=${NCCL_AVOID_RECORD_STREAMS:-1}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set +x

ProjectPath="/mntdata-2/data-100B"

EXP_NAME="${EXP_NAME:-"1b_debse_baseline"}"
CHECKPOINT_PATH=${1:-"${ProjectPath}/checkpoints/${EXP_NAME}"}
TENSORBOARD_LOGS_PATH=${2:-"${ProjectPath}/tensorboard_logs/${EXP_NAME}"}
TOKENIZER_ARG=${3:-"/models/Qwen3-30B-A3B"}
DATA_ARG=${4:-""}
if [ -n "$DATA_ARG" ]; then
    DATA_PATHS=($DATA_ARG)
else
    DATA_PREFIX=aix_sample_100b_multi_source_processed_data
    DATA_PATHS=(
        "${ProjectPath}/${DATA_PREFIX}_0_text_document"
        "${ProjectPath}/${DATA_PREFIX}_1_text_document"
        "${ProjectPath}/${DATA_PREFIX}_2_text_document"
        "${ProjectPath}/${DATA_PREFIX}_3_text_document"
        "${ProjectPath}/${DATA_PREFIX}_4_text_document"
        "${ProjectPath}/${DATA_PREFIX}_5_text_document"
        "${ProjectPath}/${DATA_PREFIX}_6_text_document"
        "${ProjectPath}/${DATA_PREFIX}_7_text_document"
    )
fi
DATA_CACHE_PATH="${CHECKPOINT_PATH}/data_cache_${EXP_NAME}"
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
TP_SIZE=8
CP_SIZE=1
PP_SIZE=1
MICRO_BATCH_SIZE=2
GLOBAL_BATCH_SIZE=160
NUM_LAYERS=48
DTYPE="bf16"
SEQ_LENGTH=4096
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
    --hidden-size 4096
    --ffn-hidden-size 12288
    --num-attention-heads 32
    --group-query-attention
    --num-query-groups 8
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
    --untie-embeddings-and-output-weights
)

FanStack_ARGS=(
    --fan-layer-enabled
    --fan-p-ratio 0.125
    --fan-enable-v-fan
    --stack-memory-enabled
    --stack-memory-slots 24
    --stack-memory-num-heads 8
    --stack-memory-dim 16
)

MOE_ARGS=(
)

TRAINING_ARGS=(
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE
    --train-samples 30694000 # 10176630 for 40B  30694000 for 120B
    --lr-decay-samples 26000000 # 8000000 for 40B  26000000 for 120B
    --lr-warmup-samples 64000 # 12000 for 40B  64000 for 120B
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
    --data-path "${DATA_PATHS[@]}"
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model "$TOKENIZER_ARG"
    --data-cache-path "$DATA_CACHE_PATH"
    --split 99,1,0
    --dataloader-type cyclic
    "--no-create-attention-mask-in-dataloader"
    "--no-mmap-bin-files"
    --num-workers 8
    # Note: --vocab-size might be inferred by HuggingFaceTokenizer or might need to be explicit.
    --vocab-size 151936
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 40
    --eval-iters 100
    --eval-interval 4000
    --save-interval 8000
    --log-throughput
    --profile
    --profile-step-start 4
    --profile-step-end 6
    --ckpt-format torch_dist 
    --distributed-timeout-minutes 60
    --save "$CHECKPOINT_PATH"
    --no-save-optim
    --tensorboard-dir "$TENSORBOARD_LOGS_PATH"
    --tensorboard-queue-size 1
)

LOAD_ARGS=(
    --load "$CHECKPOINT_PATH"
    # --load /models/Qwen3-8B-Base-mcore-tp8/
    --no-load-optim
    # --finetune
    --override-opt_param-scheduler
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

