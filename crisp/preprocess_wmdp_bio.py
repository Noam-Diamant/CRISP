"""
Preprocessing script for WMDP-bio dataset.

This script loads the WMDP-bio forget corpus from HuggingFace,
cleans and formats the text into appropriate paragraph lengths,
and saves the processed data as JSONL files.

Usage:
    python preprocess_wmdp_bio.py
"""

import os
import json
from datasets import load_dataset
from tqdm.auto import tqdm

from globals import DATA_PATH, SEED
from data import prepare_text

def preprocess_wmdp_bio_forget(
    output_path: str = None,
    max_len: int = 1000,
    dataset_name: str = "cais/wmdp-bio-forget-corpus"
):
    """
    Load and preprocess WMDP-bio forget corpus from HuggingFace.
    
    Args:
        output_path: Path to save cleaned data. If None, uses DATA_PATH/wmdp/bio/
        max_len: Maximum paragraph length
        dataset_name: HuggingFace dataset name
    """
    print(f"Loading WMDP-bio forget corpus from {dataset_name}...")
    
    try:
        # Load raw data from HuggingFace
        raw_data = load_dataset(dataset_name, split="train")
        
        print(f"Loaded {len(raw_data)} examples from HuggingFace")
        
        # Extract text field
        if 'text' in raw_data.column_names:
            texts = raw_data['text']
        elif 'content' in raw_data.column_names:
            texts = raw_data['content']
        else:
            # Try to find any text field
            print(f"Available columns: {raw_data.column_names}")
            texts = raw_data[raw_data.column_names[0]]
        
        print(f"Preprocessing {len(texts)} text examples...")
        
        # Clean and format paragraphs
        cleaned_data = prepare_text(texts, max_len=max_len)
        
        print(f"Processed into {len(cleaned_data)} paragraphs")
        
        # Create output directory
        if output_path is None:
            output_path = os.path.join(DATA_PATH, "wmdp", "bio", "bio_forget_dataset_cleaned.jsonl")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save as JSONL
        print(f"Saving cleaned data to {output_path}...")
        with open(output_path, 'w', encoding='utf-8') as f:
            for text in tqdm(cleaned_data, desc="Writing JSONL"):
                json.dump({"text": text}, f)
                f.write('\n')
        
        print(f"✓ Successfully saved {len(cleaned_data)} cleaned examples")
        print(f"  Output: {output_path}")
        
        # Print statistics
        lengths = [len(text) for text in cleaned_data]
        print(f"\nStatistics:")
        print(f"  Total examples: {len(cleaned_data)}")
        print(f"  Min length: {min(lengths)}")
        print(f"  Max length: {max(lengths)}")
        print(f"  Average length: {sum(lengths) / len(lengths):.0f}")
        
        return cleaned_data
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print(f"\nNote: You may need to request access to {dataset_name}")
        print(f"Visit: https://huggingface.co/datasets/{dataset_name}")
        raise

def preprocess_wmdp_bio_retain(
    output_path: str = None,
    max_len: int = 1000,
    dataset_name: str = "cais/wmdp-bio-retain-corpus"
):
    """
    Load and preprocess WMDP-bio retain corpus from HuggingFace (optional).
    
    Note: This is optional since the main pipeline uses Wikipedia as retain data.
    
    Args:
        output_path: Path to save cleaned data. If None, uses DATA_PATH/wmdp/bio/
        max_len: Maximum paragraph length
        dataset_name: HuggingFace dataset name
    """
    print(f"Loading WMDP-bio retain corpus from {dataset_name}...")
    
    try:
        # Load raw data from HuggingFace
        raw_data = load_dataset(dataset_name, split="train")
        
        print(f"Loaded {len(raw_data)} examples from HuggingFace")
        
        # Extract text field
        if 'text' in raw_data.column_names:
            texts = raw_data['text']
        elif 'content' in raw_data.column_names:
            texts = raw_data['content']
        else:
            print(f"Available columns: {raw_data.column_names}")
            texts = raw_data[raw_data.column_names[0]]
        
        print(f"Preprocessing {len(texts)} text examples...")
        
        # Clean and format paragraphs
        cleaned_data = prepare_text(texts, max_len=max_len)
        
        print(f"Processed into {len(cleaned_data)} paragraphs")
        
        # Create output directory
        if output_path is None:
            output_path = os.path.join(DATA_PATH, "wmdp", "bio", "bio_retain_dataset_cleaned.jsonl")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save as JSONL
        print(f"Saving cleaned data to {output_path}...")
        with open(output_path, 'w', encoding='utf-8') as f:
            for text in tqdm(cleaned_data, desc="Writing JSONL"):
                json.dump({"text": text}, f)
                f.write('\n')
        
        print(f"✓ Successfully saved {len(cleaned_data)} cleaned examples")
        print(f"  Output: {output_path}")
        
        # Print statistics
        lengths = [len(text) for text in cleaned_data]
        print(f"\nStatistics:")
        print(f"  Total examples: {len(cleaned_data)}")
        print(f"  Min length: {min(lengths)}")
        print(f"  Max length: {max(lengths)}")
        print(f"  Average length: {sum(lengths) / len(lengths):.0f}")
        
        return cleaned_data
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print(f"\nNote: You may need to request access to {dataset_name}")
        print(f"Visit: https://huggingface.co/datasets/{dataset_name}")
        raise

if __name__ == "__main__":
    print("=" * 60)
    print("WMDP-Bio Data Preprocessing")
    print("=" * 60)
    print()
    
    # Preprocess forget corpus (required)
    print("[1/2] Processing FORGET corpus...")
    print("-" * 60)
    try:
        forget_data = preprocess_wmdp_bio_forget()
        print("\n✓ Forget corpus preprocessing complete!")
    except Exception as e:
        print(f"\n✗ Failed to preprocess forget corpus: {e}")
        print("\nIf you don't have access, you can:")
        print("  1. Request access at: https://huggingface.co/datasets/cais/wmdp-bio-forget-corpus")
        print("  2. Or use the HuggingFace fallback in load_wmdp_data() (will preprocess on-the-fly)")
    
    print("\n" + "=" * 60)
    
    # Optionally preprocess retain corpus
    print("\n[2/2] Processing RETAIN corpus (OPTIONAL)...")
    print("-" * 60)
    print("Note: The pipeline uses Wikipedia by default, so this is optional.")
    
    user_input = input("Preprocess WMDP-bio retain corpus? (y/N): ").lower()
    
    if user_input == 'y':
        try:
            retain_data = preprocess_wmdp_bio_retain()
            print("\n✓ Retain corpus preprocessing complete!")
        except Exception as e:
            print(f"\n✗ Failed to preprocess retain corpus: {e}")
    else:
        print("Skipping retain corpus preprocessing.")
        print("The pipeline will use Wikipedia (wikitext-2) as retain data.")
    
    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print("=" * 60)

