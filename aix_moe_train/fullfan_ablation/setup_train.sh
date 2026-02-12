# only fan without norm

    # --fan-layer-enabled
    # --fan-p-ratio 0.5
    # --fan-disable-qk-fan
    # --visualize-model-structure
    # --fan-layer-no-norm-enabled
    # --fan-pre-mlp-norm-enabled
SERVERS="10.103.255.6:0 10.103.255.7:1" CONTAINER_NAME_PREFIX="megatron_4b_moe_full_fan" bash aix_moe_train/fullfan_ablation/run_multi_node.sh