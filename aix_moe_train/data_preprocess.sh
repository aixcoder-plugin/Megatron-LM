python tools/preprocess_data.py \
    --input /models/nemotron_cc_v2_high_sample_50B/sample_100k/train_100000.jsonl \
    --json-keys text \
    --output-prefix nemotraon_ccv2_sample_100k_processed_data \
    --output-dir /models/nemotron_cc_v2_high_sample_50B/sample_100k \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model /models/Qwen3-30B-A3B \
    --workers 8 \
    --append-eod \
    --append-bos

# docker run -it -d --gpus all --net=host -v /dev/shm:/dev/shm -v /models:/models -v /nfs100:/nfs100/ -v /nfsEDS/:/nfsEDS/ -w /nfs100/jiangsiyuan/Megatron-LM --name=jsy_megatron_debug nvcr.io/nvidia/pytorch:25.04-py3-megatron-260115 /bin/bash