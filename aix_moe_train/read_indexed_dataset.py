import sys
import os
import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from megatron.core.datasets import indexed_dataset


def read_and_print_samples(path_prefix, num_samples=5, max_tokens_per_sample=8192, tokenizer=None):

    print(f"\n=== 读取数据文件: {path_prefix} ===")
    
    idx_path = f"{path_prefix}.idx"
    bin_path = f"{path_prefix}.bin"
    
    if not os.path.exists(idx_path):
        print(f"错误：找不到索引文件 {idx_path}")
        return
    if not os.path.exists(bin_path):
        print(f"错误：找不到数据文件 {bin_path}")
        return
    
    # 创建 IndexedDataset 对象
    try:
        dataset = indexed_dataset.IndexedDataset(path_prefix, multimodal=False, mmap=True)
    except Exception as e:
        print(f"加载数据集时出错: {e}")
        return
    
    print(f"\n数据集统计信息：")
    print(f"  - 总序列数: {len(dataset)}")
    print(f"  - 总文档数: {dataset.document_indices.shape[0] - 1 if hasattr(dataset, 'document_indices') else 'N/A'}")
    
    if hasattr(dataset, 'sequence_lengths'):
        seq_lengths = dataset.sequence_lengths
        print(f"  - 序列长度统计:")
        print(f"    • 最小长度: {seq_lengths.min()}")
        print(f"    • 最大长度: {seq_lengths.max()}")
        print(f"    • 平均长度: {seq_lengths.mean():.2f}")
        print(f"    • 总token数: {seq_lengths.sum()}")
    
    actual_samples = min(num_samples, len(dataset))
    
    # 生成随机采样索引
    if len(dataset) > actual_samples:
        sample_indices = np.random.choice(len(dataset), actual_samples, replace=False)
        sample_indices = sorted(sample_indices)
    else:
        # 如果数据集小于要求的样本数，就全部打印
        sample_indices = range(len(dataset))
    
    print(f"\n\n采样 {actual_samples} 个样本：")
    print("=" * 80)
    
    for i, idx in enumerate(sample_indices):
        sample = dataset[idx]
        
        # 如果返回的是元组（多模态数据），只取第一个元素
        if isinstance(sample, tuple):
            tokens = sample[0]
        else:
            tokens = sample
        
        if len(tokens) > max_tokens_per_sample:
            display_tokens = tokens[:max_tokens_per_sample]
            truncated = True
        else:
            display_tokens = tokens
            truncated = False
        
        print(f"\n样本 #{idx} (序列长度: {len(tokens)} tokens):")
        print("-" * 40)
        
        print(f"Token IDs: {display_tokens.tolist()}")
        if tokenizer is not None:
            print(f"Text: \n{tokenizer.decode(display_tokens.tolist())}")
        if truncated:
            print(f"... (截断显示，实际共 {len(tokens)} 个tokens)")
        
        print("\n\n\n", flush=True)
    


def main():

    path_prefix = "/models/nemotron_cc_v2_high_sample_50B/sample_100k/nemotraon_ccv2_sample_100k_processed_data_text_document"
    num_samples = 5
    max_tokens = 8192
    tokenizer = AutoTokenizer.from_pretrained("/models/Qwen3-30B-A3B")
    
    read_and_print_samples(
        path_prefix=path_prefix,
        num_samples=num_samples,
        max_tokens_per_sample=max_tokens,
        tokenizer=tokenizer
    )


if __name__ == '__main__':
    main()
