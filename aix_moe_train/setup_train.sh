# stack memory 带详细的tensorboard log

    # --log-stack-memory-to-tensorboard
    # --stack-memory-enabled
    # --stack-memory-slots 16
    # --stack-memory-num-heads 8
    # --stack-memory-dim 16
SERVERS="10.103.255.4:0 10.103.255.1:1" CONTAINER_NAME_PREFIX="megatron_4b_moe_stack_dim16_softmask_tb_log" bash aix_moe_train/run_multi_node.sh


# stack memory 带详细的tensorboard log

    # --log-stack-memory-to-tensorboard
    # --stack-memory-enabled
    # --stack-memory-slots 16
    # --stack-memory-num-heads 8
    # --stack-memory-dim 64
SERVERS="10.103.255.4:0 10.103.255.1:1" CONTAINER_NAME_PREFIX="megatron_4b_moe_stack_dim64_softmask_tb_log" bash aix_moe_train/run_multi_node.sh


# stack memory 带详细的tensorboard log
# MICRO_BATCH_SIZE=1
    # --log-stack-memory-to-tensorboard
    # --stack-memory-enabled
    # --stack-memory-slots 16
    # --stack-memory-num-heads 16
    # --stack-memory-dim 128
SERVERS="10.103.255.2:0 10.103.255.1:1" CONTAINER_NAME_PREFIX="megatron_4b_moe_stack_dim-full_softmask_tb_log" bash aix_moe_train/run_multi_node.sh



# fan-v-fan 替换 v 的 linear_fc2
# MICRO_BATCH_SIZE=2
    # --fan-layer-enabled
    # --fan-p-ratio 0.125
    # --fan-enable-v-fan
    # --fan-enable-qk-fan
SERVERS="10.103.255.6:0 10.103.255.7:1" CONTAINER_NAME_PREFIX="megatron_4b_moe_fan_v_fan" bash aix_moe_train/run_multi_node.sh