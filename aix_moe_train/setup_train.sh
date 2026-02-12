# stack memory 带详细的tensorboard log

    # --log-stack-memory-to-tensorboard
    # --stack-memory-enabled
    # --stack-memory-slots 16
    # --stack-memory-num-heads 8
    # --stack-memory-dim 16
SERVERS="10.103.255.4:0 10.103.255.1:1" CONTAINER_NAME_PREFIX="megatron_4b_moe_stack_dim16_softmask_tb_log" bash aix_moe_train/run_multi_node.sh