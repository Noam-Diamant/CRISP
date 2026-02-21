"""
Preprocessing script for WMDP datasets (bio and cyber).

This script loads the WMDP forget corpus from HuggingFace,
cleans and formats the text into appropriate paragraph lengths,
and saves the processed data as JSONL files.

Usage:
    python preprocess_wmdp_bio.py --dataset bio     # Process bio only
    python preprocess_wmdp_bio.py --dataset cyber   # Process cyber only
    python preprocess_wmdp_bio.py --dataset both    # Process both
"""

import os
import json
import argparse
from datasets import load_dataset
from tqdm.auto import tqdm

from globals import DATA_PATH, SEED
from data import prepare_text

DATASET_CONFIGS = {
    "bio": {
        "forget_corpus": "cais/wmdp-corpora",
        "forget_subset": "bio-forget-corpus",
        "retain_corpus": "cais/wmdp-corpora",
        "retain_subset": "bio-retain-corpus",
        "output_dir": "bio",
    },
    "cyber": {
        "forget_corpus": "cais/wmdp-corpora",
        "forget_subset": "cyber-forget-corpus",
        "retain_corpus": "cais/wmdp-corpora",
        "retain_subset": "cyber-retain-corpus",
        "output_dir": "cyber",
    },
}


def preprocess_wmdp_forget(
    dataset_type: str,
    output_path: str = None,
    max_len: int = 1000,
):
    """
    Load and preprocess WMDP forget corpus from HuggingFace.
    
    Args:
        dataset_type: Either "bio" or "cyber"
        output_path: Path to save cleaned data. If None, uses DATA_PATH/wmdp/{type}/
        max_len: Maximum paragraph length
    """
    config = DATASET_CONFIGS[dataset_type]
    dataset_name = config["forget_corpus"]
    subset = config["forget_subset"]
    
    print(f"Loading WMDP-{dataset_type} forget corpus from {dataset_name} (subset: {subset})...")
    
    try:
        raw_data = load_dataset(dataset_name, subset, split="train")
        
        print(f"Loaded {len(raw_data)} examples from HuggingFace")
        
        if 'text' in raw_data.column_names:
            texts = raw_data['text']
        elif 'content' in raw_data.column_names:
            texts = raw_data['content']
        else:
            print(f"Available columns: {raw_data.column_names}")
            texts = raw_data[raw_data.column_names[0]]
        
        print(f"Preprocessing {len(texts)} text examples...")
        
        cleaned_data = prepare_text(texts, max_len=max_len)
        
        print(f"Processed into {len(cleaned_data)} paragraphs")
        
        if output_path is None:
            output_path = os.path.join(
                DATA_PATH, "wmdp", config["output_dir"], 
                f"{dataset_type}_forget_dataset_cleaned.jsonl"
            )
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        print(f"Saving cleaned data to {output_path}...")
        with open(output_path, 'w', encoding='utf-8') as f:
            for text in tqdm(cleaned_data, desc="Writing JSONL"):
                json.dump({"text": text}, f)
                f.write('\n')
        
        print(f"Successfully saved {len(cleaned_data)} cleaned examples")
        print(f"  Output: {output_path}")
        
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


def preprocess_wmdp_retain(
    dataset_type: str,
    output_path: str = None,
    max_len: int = 1000,
):
    """
    Load and preprocess WMDP retain corpus from HuggingFace (optional).
    
    Note: This is optional since the main pipeline uses Wikipedia as retain data.
    
    Args:
        dataset_type: Either "bio" or "cyber"
        output_path: Path to save cleaned data. If None, uses DATA_PATH/wmdp/{type}/
        max_len: Maximum paragraph length
    """
    config = DATASET_CONFIGS[dataset_type]
    dataset_name = config["retain_corpus"]
    subset = config["retain_subset"]
    
    print(f"Loading WMDP-{dataset_type} retain corpus from {dataset_name} (subset: {subset})...")
    
    try:
        raw_data = load_dataset(dataset_name, subset, split="train")
        
        print(f"Loaded {len(raw_data)} examples from HuggingFace")
        
        if 'text' in raw_data.column_names:
            texts = raw_data['text']
        elif 'content' in raw_data.column_names:
            texts = raw_data['content']
        else:
            print(f"Available columns: {raw_data.column_names}")
            texts = raw_data[raw_data.column_names[0]]
        
        print(f"Preprocessing {len(texts)} text examples...")
        
        cleaned_data = prepare_text(texts, max_len=max_len)
        
        print(f"Processed into {len(cleaned_data)} paragraphs")
        
        if output_path is None:
            output_path = os.path.join(
                DATA_PATH, "wmdp", config["output_dir"],
                f"{dataset_type}_retain_dataset_cleaned.jsonl"
            )
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        print(f"Saving cleaned data to {output_path}...")
        with open(output_path, 'w', encoding='utf-8') as f:
            for text in tqdm(cleaned_data, desc="Writing JSONL"):
                json.dump({"text": text}, f)
                f.write('\n')
        
        print(f"Successfully saved {len(cleaned_data)} cleaned examples")
        print(f"  Output: {output_path}")
        
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


def preprocess_wmdp_bio_forget(output_path: str = None, max_len: int = 1000, dataset_name: str = None):
    """Backward-compatible wrapper for bio forget preprocessing."""
    return preprocess_wmdp_forget("bio", output_path, max_len)


def preprocess_wmdp_bio_retain(output_path: str = None, max_len: int = 1000, dataset_name: str = None):
    """Backward-compatible wrapper for bio retain preprocessing."""
    return preprocess_wmdp_retain("bio", output_path, max_len)

def process_dataset(dataset_type: str, process_retain: bool = False):
    """Process a single dataset type (bio or cyber)."""
    print(f"\n{'=' * 60}")
    print(f"Processing WMDP-{dataset_type.upper()} Dataset")
    print("=" * 60)
    
    print(f"\n[1/2] Processing {dataset_type.upper()} FORGET corpus...")
    print("-" * 60)
    try:
        forget_data = preprocess_wmdp_forget(dataset_type)
        print(f"\nForget corpus preprocessing complete!")
    except Exception as e:
        print(f"\nFailed to preprocess forget corpus: {e}")
        print("\nIf you don't have access, you can:")
        print(f"  1. Request access at: https://huggingface.co/datasets/cais/wmdp-corpora")
        print("  2. Or use the HuggingFace fallback in load_wmdp_data() (will preprocess on-the-fly)")
    
    if process_retain:
        print(f"\n[2/2] Processing {dataset_type.upper()} RETAIN corpus...")
        print("-" * 60)
        try:
            retain_data = preprocess_wmdp_retain(dataset_type)
            print(f"\nRetain corpus preprocessing complete!")
        except Exception as e:
            print(f"\nFailed to preprocess retain corpus: {e}")
    else:
        print(f"\n[2/2] Skipping {dataset_type.upper()} RETAIN corpus (use --retain to include)")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess WMDP datasets (bio and/or cyber)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python preprocess_wmdp_bio.py --dataset bio      # Process bio only
    python preprocess_wmdp_bio.py --dataset cyber    # Process cyber only
    python preprocess_wmdp_bio.py --dataset both     # Process both datasets
    python preprocess_wmdp_bio.py --dataset bio --retain  # Include retain corpus
        """
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        choices=["bio", "cyber", "both"],
        default="bio",
        help="Which dataset to preprocess: bio, cyber, or both (default: bio)"
    )
    parser.add_argument(
        "--retain",
        action="store_true",
        help="Also preprocess the retain corpus (optional, pipeline uses Wikipedia by default)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("WMDP Data Preprocessing")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Include retain corpus: {args.retain}")
    
    if args.dataset == "both":
        datasets_to_process = ["bio", "cyber"]
    else:
        datasets_to_process = [args.dataset]
    
    for dataset_type in datasets_to_process:
        process_dataset(dataset_type, process_retain=args.retain)
    
    print("\n" + "=" * 60)
    print("All preprocessing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

