SERVERS="10.103.255.4:0 10.103.255.5:1" CONTAINER_NAME_PREFIX="megatron_1b_dense_baseline" bash aix_moe_train/dense_baseline/run_multi_node.sh




# MICRO_BATCH_SIZE=2
    # --log-stack-memory-to-tensorboard
    # --stack-memory-enabled
    # --stack-memory-slots 16
    # --stack-memory-num-heads 8
    # --stack-memory-dim 64

SERVERS="10.103.255.4:0 10.103.255.5:1" CONTAINER_NAME_PREFIX="megatron_1b_dense_stack_dim64" bash aix_moe_train/dense_baseline/run_multi_node.sh


# MICRO_BATCH_SIZE=4
    # --log-stack-memory-to-tensorboard
    # --stack-memory-enabled
    # --stack-memory-slots 16
    # --stack-memory-num-heads 8
    # --stack-memory-dim 16
SERVERS="10.103.255.2:0 10.103.255.1:1" CONTAINER_NAME_PREFIX="megatron_1b_dense_stack_dim16" bash aix_moe_train/dense_baseline/run_multi_node.sh


# MICRO_BATCH_SIZE=2
    # --log-stack-memory-to-tensorboard
    # --stack-memory-enabled
    # --stack-memory-slots 16
    # --stack-memory-num-heads 16
    # --stack-memory-dim 128
SERVERS="10.103.255.4:0 10.103.255.5:1" CONTAINER_NAME_PREFIX="megatron_1b_dense_stack_dim_full" bash aix_moe_train/dense_baseline/run_multi_node.sh


# fan
# MICRO_BATCH_SIZE=4
    # --fan-layer-enabled
    # --fan-p-ratio 0.125
SERVERS="10.103.255.2:0 10.103.255.1:1" CONTAINER_NAME_PREFIX="megatron_1b_dense_fan_p0.125" bash aix_moe_train/dense_baseline/run_multi_node.sh




# fan
# MICRO_BATCH_SIZE=4
    --fan-layer-enabled
    --fan-p-ratio 0.125
    --log-stack-memory-to-tensorboard
    --stack-memory-enabled
    --stack-memory-slots 12
    --stack-memory-num-heads 8
    --stack-memory-dim 16
SERVERS="10.103.255.4:0 10.103.255.5:1" CONTAINER_NAME_PREFIX="megatron_1b_dense_fan_p0.125_stack_dim16" bash aix_moe_train/dense_baseline/run_multi_node.sh




# fan-v-fan 替换 v 的 linear_fc2
MICRO_BATCH_SIZE=4
    --fan-layer-enabled
    --fan-p-ratio 0.125
    --fan-enable-v-fan
SERVERS="10.103.255.2:0 10.103.255.1:1" CONTAINER_NAME_PREFIX="megatron_1b_dense_fan_v_fan_p0.125" bash aix_moe_train/dense_baseline/run_multi_node.sh



    --fan-layer-enabled
    --fan-p-ratio 0.125
    --fan-enable-v-fan
    --stack-memory-enabled
    --stack-memory-slots 12
    --stack-memory-num-heads 8
    --stack-memory-dim 16
SERVERS="10.103.255.4:0 10.103.255.5:1" CONTAINER_NAME_PREFIX="megatron_1b_dense_fan_v_fan_p0.125_stack_dim16" bash aix_moe_train/dense_baseline/run_multi_node.sh



    --fan-layer-enabled
    --fan-p-ratio 0.125
    --fan-enable-v-fan
SERVERS="10.103.255.2:0 10.103.255.1:1" CONTAINER_NAME_PREFIX="megatron_1b_dense_fan_v_fan_p0.125_2" bash aix_moe_train/dense_baseline/run_multi_node.sh






    --fan-layer-enabled
    --fan-p-ratio 0.125
    --fan-enable-v-fan
    --stack-memory-enabled
    --stack-memory-slots 12
    --stack-memory-num-heads 8
    --stack-memory-dim 16
SERVERS="10.103.255.4:0 10.103.255.5:1" CONTAINER_NAME_PREFIX="megatron_1b_dense_fan_v_fan_p0.125_stack_dim16_2" bash aix_moe_train/dense_baseline/run_multi_node.sh



SERVERS="10.103.255.6:0" CONTAINER_NAME_PREFIX="megatron_1b_dense_baseline_2" bash aix_moe_train/dense_baseline/run_multi_node.sh


SERVERS="10.103.255.2:0 10.103.255.1:1 10.103.255.4:2 10.103.255.5:3" CONTAINER_NAME_PREFIX="megatron_1b_dense_baseline2" bash aix_moe_train/dense_baseline/run_multi_node.sh



TRAIN_SCRIPT_REL="aix_moe_train/dense_baseline/train_10b_full_new.sh" SERVERS="10.103.255.2:0 10.103.255.1:1 10.103.255.4:2 10.103.255.5:3" CONTAINER_NAME_PREFIX="megatron_12b_dense_baseline" bash aix_moe_train/dense_baseline/run_multi_node.sh



    --fan-layer-enabled
    --fan-p-ratio 0.125
    --fan-enable-v-fan
SERVERS="10.103.255.6:0" CONTAINER_NAME_PREFIX="megatron_1b_dense_fan_v_fan_p0.125_3" bash aix_moe_train/dense_baseline/run_multi_node.sh