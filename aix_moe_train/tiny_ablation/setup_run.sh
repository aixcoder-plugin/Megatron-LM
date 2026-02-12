SERVERS="10.103.255.4:0" CONTAINER_NAME_PREFIX="tiny_moe_baseline" bash aix_moe_train/tiny_ablation/run_multi_node.sh


SERVERS="10.103.255.1:0" CONTAINER_NAME_PREFIX="tiny_moe_fan_p0_fused_qkv_norm" bash aix_moe_train/tiny_ablation/run_multi_node.sh


    # --fan-layer-enabled
    # --fan-p-ratio 0
    # --fan-disable-qk-fan







SERVERS="10.103.255.2:0" CONTAINER_NAME_PREFIX="tiny_moe_fan_p0.125_split_qkv_norm" bash aix_moe_train/tiny_ablation/run_multi_node.sh


    # --fan-layer-enabled
    # --fan-p-ratio 0.125



SERVERS="10.103.255.4:0" CONTAINER_NAME_PREFIX="tiny_moe_fan_p0_split_qkv_norm2" bash aix_moe_train/tiny_ablation/run_multi_node.sh


    # --fan-layer-enabled
    # --fan-p-ratio 0



SERVERS="10.103.255.1:0" CONTAINER_NAME_PREFIX="tiny_moe_stack_dim16" bash aix_moe_train/tiny_ablation/run_multi_node.sh

    # --log-stack-memory-to-tensorboard
    # --stack-memory-enabled
    # --stack-memory-slots 6
    # --stack-memory-num-heads 8
    # --stack-memory-dim 16



SERVERS="10.103.255.5:0" CONTAINER_NAME_PREFIX="tiny_moe_stack_dim1" bash aix_moe_train/tiny_ablation/run_multi_node.sh

    # --log-stack-memory-to-tensorboard
    # --stack-memory-enabled
    # --stack-memory-slots 6
    # --stack-memory-num-heads 8
    # --stack-memory-dim 1






SERVERS="10.103.255.1:0" CONTAINER_NAME_PREFIX="tiny_moe_stack_dim16" bash aix_moe_train/tiny_ablation/run_multi_node.sh
SERVERS="10.103.255.1:0" CONTAINER_NAME_PREFIX="tiny_moe_stack_dim16-fix-mask" bash aix_moe_train/tiny_ablation/run_multi_node.sh

    --log-stack-memory-to-tensorboard
    --stack-memory-enabled
    --stack-memory-slots 6
    --stack-memory-num-heads 8
    --stack-memory-dim 16



SERVERS="10.103.255.4:0" CONTAINER_NAME_PREFIX="tiny_moe_stack_dim1" bash aix_moe_train/tiny_ablation/run_multi_node.sh
SERVERS="10.103.255.4:0" CONTAINER_NAME_PREFIX="tiny_moe_stack_dim1-fix-mask" bash aix_moe_train/tiny_ablation/run_multi_node.sh

    --log-stack-memory-to-tensorboard
    --stack-memory-enabled
    --stack-memory-slots 6
    --stack-memory-num-heads 8
    --stack-memory-dim 1