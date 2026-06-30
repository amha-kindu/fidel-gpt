import os
import json
import argparse
import numpy as np
from tqdm import tqdm
import sentencepiece as spm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

def analyze_token_lengths(file_path, tokenizer, max_len, max_samples=None):
    lengths = []
    total_tokens = 0
    empty_samples = 0
    truncated_samples = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        if max_samples:
            lines = lines[:max_samples]
            
        for line in tqdm(lines, desc="Processing samples"):
            try:
                data = json.loads(line)
                text = data if isinstance(data, str) else data.get('text', '')
                
                if not text.strip():
                    empty_samples += 1
                    continue
                    
                tokens = tokenizer.EncodeAsIds(text)
                token_len = len(tokens)
                lengths.append(token_len)
                total_tokens += token_len
                
                if token_len >= max_len:
                    truncated_samples += 1
                    
            except json.JSONDecodeError:
                continue
    
    return {
        'lengths': np.array(lengths),
        'total_samples': len(lengths),
        'empty_samples': empty_samples,
        'truncated_samples': truncated_samples,
        'total_tokens': total_tokens,
        'avg_length': total_tokens / len(lengths) if lengths else 0
    }

def visualize_stats(stats, output_dir, tokenizer, max_len):
    os.makedirs(output_dir, exist_ok=True)
    writer = SummaryWriter(output_dir)
    
    lengths = stats['lengths']
    
    # Basic statistics
    print("\nToken Length Statistics:")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Empty samples: {stats['empty_samples']}")
    print(f"Samples that would be truncated: {stats['truncated_samples']}")
    print(f"Total tokens: {stats['total_tokens']}")
    print(f"Average length: {stats['avg_length']:.2f}")
    print(f"Min length: {np.min(lengths)}")
    print(f"Max length: {np.max(lengths)}")
    print(f"Median length: {np.median(lengths)}")
    print(f"95th percentile: {np.percentile(lengths, 95)}")
    
    # Add histogram to TensorBoard
    writer.add_histogram('token_lengths', lengths, 0)
    
    # Add text summary
    writer.add_text('stats', 
                   f"Total samples: {stats['total_samples']}<br>"
                   f"Empty samples: {stats['empty_samples']}<br>"
                   f"Truncated samples: {stats['truncated_samples']}<br>"
                   f"Average length: {stats['avg_length']:.2f}<br>"
                   f"Max length: {np.max(lengths)}<br>"
                   f"95th percentile: {np.percentile(lengths, 95)}")
    
    # Create and save matplotlib figure
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=50, alpha=0.7)
    plt.axvline(max_len, color='r', linestyle='dashed', linewidth=1)
    plt.title(f'Token Length Distribution (max_len={max_len})')
    plt.xlabel('Number of tokens')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'token_length_histogram.png'))
    plt.close()
    
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze token lengths in JSONL file')
    parser.add_argument('--file', type=str, required=True, help='Path to JSONL file')
    parser.add_argument('--tokenizer', type=str, required=True, help='Path to SentencePiece model')
    parser.add_argument('--max-len', type=int, default=256, help='Maximum sequence length')
    parser.add_argument('--output-dir', type=str, default='./token_stats', help='Output directory for TensorBoard')
    parser.add_argument('--max-samples', type=int, default=None, help='Maximum number of samples to process')
    
    args = parser.parse_args()
    
    # Initialize tokenizer
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(args.tokenizer)
    
    # Analyze token lengths
    stats = analyze_token_lengths(args.file, tokenizer, args.max_len, args.max_samples)
    
    # Visualize results
    visualize_stats(stats, args.output_dir, tokenizer, args.max_len)
    
    print(f"\nAnalysis complete. View results with:")
    print(f"tensorboard --logdir={args.output_dir}")
    print(f"Histogram saved to {os.path.join(args.output_dir, 'token_length_histogram.png')}")