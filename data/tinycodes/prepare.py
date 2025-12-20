"""
TinyCodes Data Preparation
--------------------------
Downloads and tokenizes Python code datasets for training SpinNet.

Uses: bigcode/the-stack-smol (public, Python subset, ~100MB)

Usage:
    python data/tinycodes/prepare.py
"""

import os
import numpy as np
from datasets import load_dataset
import tiktoken

# Output directory
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

def prepare_tinycodes():
    print("Loading The Stack (smol) Python dataset from HuggingFace...")
    
    # Use the-stack-smol which is a small sample of The Stack
    # Filter for Python
    try:
        dataset = load_dataset(
            "bigcode/the-stack-smol",
            data_dir="data/python",
            split="train",
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Failed to load the-stack-smol: {e}")
        print("Trying alternative: flytech/python-codes-25k...")
        dataset = load_dataset("flytech/python-codes-25k", split="train")
    
    print(f"Total examples: {len(dataset):,}")
    
    # Use GPT-2 tokenizer
    enc = tiktoken.get_encoding("gpt2")
    
    # Split into train/val (95/5)
    dataset = dataset.shuffle(seed=42)
    split_idx = int(len(dataset) * 0.95)
    train_data = dataset.select(range(split_idx))
    val_data = dataset.select(range(split_idx, len(dataset)))
    
    print(f"Train examples: {len(train_data):,}")
    print(f"Val examples: {len(val_data):,}")
    
    def get_code(example):
        """Extract code from different dataset formats."""
        # Try common field names
        for field in ['content', 'code', 'output', 'response', 'text']:
            if field in example and example[field]:
                return example[field]
        return str(example)
    
    def tokenize_split(data, name):
        """Tokenize a split and save to .bin file."""
        print(f"\nTokenizing {name} split...")
        
        all_tokens = []
        for i, example in enumerate(data):
            code = get_code(example)
            if code:
                tokens = enc.encode_ordinary(code)
                all_tokens.extend(tokens)
            
            if (i + 1) % 10000 == 0:
                print(f"  Processed {i+1:,} examples...")
        
        # Convert to numpy and save
        tokens_array = np.array(all_tokens, dtype=np.uint16)
        output_path = os.path.join(DATA_DIR, f"{name}.bin")
        tokens_array.tofile(output_path)
        print(f"  Saved {len(tokens_array):,} tokens to {output_path}")
        return len(tokens_array)
    
    # Tokenize both splits
    train_tokens = tokenize_split(train_data, "train")
    val_tokens = tokenize_split(val_data, "val")
    
    print(f"\n{'='*50}")
    print(f"Train tokens: {train_tokens:,}")
    print(f"Val tokens: {val_tokens:,}")
    print(f"Total tokens: {train_tokens + val_tokens:,}")
    print("Done!")

if __name__ == "__main__":
    prepare_tinycodes()
