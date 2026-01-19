#!/usr/bin/env python3
"""
Script to run unlearning experiments with varying dataset sizes.

This script performs systematic unlearning experiments by varying the number of
examples used for both forget and retain sets. It evaluates metrics at each step
and saves processed features with appropriate suffixes.
"""

import argparse
import json
import os
import sys

# IMPORTANT: Set CUDA_VISIBLE_DEVICES BEFORE importing torch
# Parse --gpu argument early to set CUDA_VISIBLE_DEVICES before torch import
# This must happen before torch import because PyTorch checks CUDA availability during import
gpu_arg = "3"  # default (matches argparse default)
if "--gpu" in sys.argv:
    gpu_idx = sys.argv.index("--gpu")
    if gpu_idx + 1 < len(sys.argv):
        gpu_arg = sys.argv[gpu_idx + 1]
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_arg

import torch
from datetime import datetime
from typing import Dict, List, Any
import gc

# Get the parent directory (CRISP/crisp/) and change to it
# This is needed because eval.py and other modules use relative paths
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
os.chdir(parent_dir)

from globals import GEMMA_2_2B, LLAMA_3_1_8B, set_seed, SEED
from crisp import CRISP, CRISPConfig
from unlearn import unlearn_lora, UnlearnConfig
from data import load_hp_data, load_wmdp_data, HPDataConfig, WMDPDataConfig
from data import genenrate_hp_eval_text, generate_bio_eval_text
from sae import JumpReLUSAE, TopkSae
from eval import get_mcq_accuracy
from utils import save_cached_features, load_cached_features


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run unlearning experiments with varying dataset sizes"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="gemma-2-2b",
        choices=["gemma-2-2b", "llama-3.1-8b"],
        help="Model to use for experiments (default: gemma-2-2b)"
    )
    
    parser.add_argument(
        "--target",
        type=str,
        default="hp",
        choices=["hp", "bio"],
        help="Target domain to unlearn (default: hp)"
    )
    
    parser.add_argument(
        "--retain",
        type=str,
        default="book",
        choices=["book", "wiki"],
        help="Retain set to use for HP target, or wiki for bio (default: book)"
    )
    
    parser.add_argument(
        "--dataset-sizes",
        type=int,
        nargs="+",
        default=[10, 25, 50, 100, 250, 500, 1000, 1500, 2500],
        help="List of dataset sizes to experiment with (default: 10 25 50 100 250 500 1000 1500 2500)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiment_results",
        help="Directory to save experiment results (default: experiment_results)"
    )
    
    parser.add_argument(
        "--gpu",
        type=str,
        default="3",
        help="GPU device ID (default: 0)"
    )
    
    parser.add_argument(
        "--max-length",
        type=int,
        default=1000,
        help="Maximum length for text processing (default: 1000)"
    )
    
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip experiments for which results already exist"
    )
    
    parser.add_argument(
        "--vary-dataset",
        type=str,
        default="both",
        choices=["both", "feature_extraction", "unlearning"],
        help="What to vary with dataset size: 'both' (default), 'feature_extraction' only, or 'unlearning' only"
    )
    
    return parser.parse_args()


def get_model_config(model_name: str, target: str) -> Dict[str, Any]:
    """Get configuration for the specified model."""
    if model_name == "gemma-2-2b":
        return {
            "model_card": GEMMA_2_2B,
            "sae_layers": list(range(4, 15, 2)),
            "sae_class": JumpReLUSAE,
            "model_name_short": "gemma",
            "unlearn": {
                "learning_rate": 1e-5,
                "k_features": 10,
                "alpha": 5,
            }
        }
    else:  # llama-3.1-8b
        return {
            "model_card": LLAMA_3_1_8B,
            "sae_layers": list(range(4, 30, 2)),
            "sae_class": TopkSae,
            "model_name_short": "llama",
            "unlearn": {
                "learning_rate": 2e-5,
                "k_features": 10,
                "alpha": 30,
            }
        }


def initialize_model(config: Dict[str, Any]) -> CRISP:
    """Initialize the CRISP model with SAEs."""
    print(f"\nInitializing model: {config['model_card']}")
    print(f"Operating on layers: {config['sae_layers']}")
    
    # Create CRISP config
    crisp_config = CRISPConfig(
        layers=config['sae_layers'],
        model_name=config['model_card'],
        bf16=True
    )
    
    # Initialize CRISP model
    crisp = CRISP(config=crisp_config)
    
    return crisp


def load_data(target: str, retain: str, n_examples: int, max_length: int):
    """Load forget and retain datasets."""
    if target == "hp":
        print(f"Loading HP data with {n_examples} examples, retain type: {retain}")
        data = load_hp_data(benign=retain, n_examples=n_examples, max_len=max_length)
        return data["forget"], data["retain"]
    else:  # bio
        print(f"Loading WMDP Bio data with {n_examples} examples, retain type: {retain}")
        data = load_wmdp_data(target_type="bio", retain_type=retain, n_examples=n_examples)
        return data["forget"], data["retain"]


def create_data_config(target: str, retain: str, n_examples: int, max_length: int):
    """Create data configuration object with n_examples suffix."""
    if target == "hp":
        return HPDataConfig(
            retain_type=retain,
            n_examples=n_examples,
            max_length=max_length,
            min_length=max_length
        )
    else:  # bio
        return WMDPDataConfig(
            forget_type="bio",
            retain_type=retain,
            n_examples=n_examples,
            max_length=max_length,
            min_length=max_length
        )


def evaluate_model(crisp: CRISP, target: str, eval_type: str = "before") -> Dict[str, float]:
    """Evaluate model on target and general knowledge tasks."""
    print(f"\nEvaluating model ({eval_type} unlearning)...")
    
    metrics = {}
    
    if target == "hp":
        # Evaluate HP accuracy
        print("Evaluating Harry Potter accuracy...")
        hp_acc = get_mcq_accuracy(crisp, type="hp", verbose=True)
        metrics["hp_accuracy"] = float(hp_acc)
        
        # Evaluate MMLU accuracy
        print("Evaluating MMLU accuracy...")
        mmlu_acc = get_mcq_accuracy(crisp, type="mmlu", verbose=True)
        metrics["mmlu_accuracy"] = float(mmlu_acc)
        
    else:  # bio
        # Evaluate WMDP Bio accuracy
        print("Evaluating WMDP Bio accuracy...")
        bio_acc = get_mcq_accuracy(crisp, type="wmdp_bio", verbose=True)
        metrics["wmdp_bio_accuracy"] = float(bio_acc)
        
        # Evaluate MMLU accuracy
        print("Evaluating MMLU accuracy...")
        mmlu_acc = get_mcq_accuracy(crisp, type="mmlu", verbose=True)
        metrics["mmlu_accuracy"] = float(mmlu_acc)
    
    return metrics


def run_single_experiment(
    n_examples: int,
    target: str,
    retain: str,
    max_length: int,
    model_config: Dict[str, Any],
    output_dir: str,
    metrics_before: Dict[str, float],
    vary_mode: str = "both",
    max_n_examples: int = None
) -> Dict[str, Any]:
    """Run a single unlearning experiment for a given number of examples.
    
    Args:
        n_examples: Number of examples to use (interpretation depends on vary_mode)
        vary_mode: What to vary - 'both', 'feature_extraction', or 'unlearning'
        max_n_examples: Maximum dataset size (used when vary_mode is not 'both')
    """
    
    print("\n" + "="*80)
    print(f"Running experiment with {n_examples} examples (vary_mode: {vary_mode})")
    print("="*80)
    
    # Initialize model
    crisp = initialize_model(model_config)
    
    # Determine dataset sizes based on vary_mode
    if vary_mode == "both":
        # Both feature extraction and unlearning use n_examples
        feature_n_examples = n_examples
        unlearn_n_examples = n_examples
    elif vary_mode == "feature_extraction":
        # Feature extraction uses n_examples, unlearning uses max
        feature_n_examples = n_examples
        unlearn_n_examples = max_n_examples
    elif vary_mode == "unlearning":
        # Feature extraction uses max, unlearning uses n_examples
        feature_n_examples = max_n_examples
        unlearn_n_examples = n_examples
    else:
        raise ValueError(f"Unknown vary_mode: {vary_mode}")
    
    # Load data for feature extraction
    print(f"Loading data for feature extraction ({feature_n_examples} examples)...")
    forget_data_features, retain_data_features = load_data(target, retain, feature_n_examples, max_length)
    
    # Create data config for feature extraction
    data_config_features = create_data_config(target, retain, feature_n_examples, max_length)
    
    # Create unlearn config
    unlearn_config = UnlearnConfig(
        data_type=target,
        learning_rate=model_config['unlearn']['learning_rate'],
        k_features=model_config['unlearn']['k_features'],
        alpha=model_config['unlearn']['alpha'],
        save_model=False,  # Do not save models to save disk space
        beta=0.99,
        gamma=0.01,
        batch_size=4,
        lora_rank=4,
        verbose=target
    )
    
    # Handle feature extraction
    if vary_mode == "both":
        # Features will be extracted inside unlearn_lora with n_examples data
        forget_data_unlearn = forget_data_features
        retain_data_unlearn = retain_data_features
        data_config_unlearn = data_config_features
    else:
        # Pre-compute features with feature_n_examples data
        print(f"\n--- Extracting features ({feature_n_examples} examples) ---")
        crisp.process_multi_texts_batch(
            text_target=forget_data_features,
            text_benign=retain_data_features,
            data_config=data_config_features,
            batch_size=unlearn_config.batch_size
        )
        
        # Load data for unlearning (different size if vary_mode != 'both')
        if unlearn_n_examples != feature_n_examples:
            print(f"Loading data for unlearning ({unlearn_n_examples} examples)...")
            forget_data_unlearn, retain_data_unlearn = load_data(target, retain, unlearn_n_examples, max_length)
        else:
            forget_data_unlearn = forget_data_features
            retain_data_unlearn = retain_data_features
        
        # Data config for unlearning (for cache naming)
        data_config_unlearn = create_data_config(target, retain, unlearn_n_examples, max_length)
    
    # Perform unlearning
    print(f"\n--- Performing Unlearning ({unlearn_n_examples} examples) ---")
    unlearn_lora(
        crisp=crisp,
        text_target=forget_data_unlearn,
        text_benign=retain_data_unlearn,
        config=unlearn_config,
        data_config=data_config_unlearn
    )
    
    # Evaluate after unlearning
    print("\n--- Evaluating After Unlearning ---")
    metrics_after = evaluate_model(crisp, target, eval_type="after")
    
    # Compile results
    results = {
        "n_examples": n_examples,
        "vary_mode": vary_mode,
        "feature_extraction_n_examples": feature_n_examples,
        "unlearning_n_examples": unlearn_n_examples,
        "target": target,
        "retain": retain,
        "model": model_config['model_card'],
        "max_length": max_length,
        "timestamp": datetime.now().isoformat(),
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "unlearn_config": unlearn_config.to_dict(),
        "data_config_features": data_config_features.to_dict(),
        "data_config_unlearn": data_config_unlearn.to_dict()
    }
    
    # Calculate improvement metrics
    if target == "hp":
        target_key = "hp_accuracy"
        retain_key = "mmlu_accuracy"
    else:
        target_key = "wmdp_bio_accuracy"
        retain_key = "mmlu_accuracy"
    
    results["target_accuracy_drop"] = metrics_before[target_key] - metrics_after[target_key]
    results["target_accuracy_drop_percent"] = (
        (metrics_before[target_key] - metrics_after[target_key]) / metrics_before[target_key] * 100
        if metrics_before[target_key] > 0 else 0
    )
    results["retain_accuracy_drop"] = metrics_before[retain_key] - metrics_after[retain_key]
    results["retain_accuracy_drop_percent"] = (
        (metrics_before[retain_key] - metrics_after[retain_key]) / metrics_before[retain_key] * 100
        if metrics_before[retain_key] > 0 else 0
    )
    
    # Save individual experiment results
    vary_suffix = f"_vary_{vary_mode}" if vary_mode != "both" else ""
    exp_filename = f"experiment_n{n_examples}_{target}_{retain}_{model_config['model_name_short']}{vary_suffix}.json"
    exp_path = os.path.join(output_dir, exp_filename)
    with open(exp_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved experiment results to: {exp_path}")
    
    # Clean up to save memory
    del crisp
    torch.cuda.empty_cache()
    gc.collect()
    
    return results


def save_summary_results(all_results: List[Dict[str, Any]], output_dir: str, args):
    """Save summary of all experiments."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    vary_suffix = f"_vary_{args.vary_dataset}" if args.vary_dataset != "both" else ""
    summary_filename = f"summary_{args.target}_{args.retain}_{args.model.replace('.', '_')}{vary_suffix}_{timestamp}.json"
    summary_path = os.path.join(output_dir, summary_filename)
    
    summary = {
        "experiment_info": {
            "target": args.target,
            "retain": args.retain,
            "model": args.model,
            "max_length": args.max_length,
            "dataset_sizes": args.dataset_sizes,
            "vary_mode": args.vary_dataset,
            "timestamp": timestamp
        },
        "results": all_results
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Saved summary results to: {summary_path}")
    print(f"{'='*80}")
    
    # Print summary table
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(f"{'N Examples':<12} {'Target Acc Before':<18} {'Target Acc After':<18} {'Drop %':<10} {'Retain Acc Before':<18} {'Retain Acc After':<18} {'Drop %':<10}")
    print("-"*120)
    
    for result in all_results:
        n_examples = result['n_examples']
        target_key = "hp_accuracy" if args.target == "hp" else "wmdp_bio_accuracy"
        retain_key = "mmlu_accuracy"
        
        target_before = result['metrics_before'][target_key]
        target_after = result['metrics_after'][target_key]
        target_drop = result['target_accuracy_drop_percent']
        
        retain_before = result['metrics_before'][retain_key]
        retain_after = result['metrics_after'][retain_key]
        retain_drop = result['retain_accuracy_drop_percent']
        
        print(f"{n_examples:<12} {target_before:<18.4f} {target_after:<18.4f} {target_drop:<10.2f} "
              f"{retain_before:<18.4f} {retain_after:<18.4f} {retain_drop:<10.2f}")
    
    print("="*80)


def main():
    """Main execution function."""
    args = parse_args()
    
    # GPU is already set before torch import (see top of file)
    # Verify it matches the argument
    if os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu:
        print(f"Warning: CUDA_VISIBLE_DEVICES was set to {os.environ.get('CUDA_VISIBLE_DEVICES')} but --gpu is {args.gpu}")
        print("Setting CUDA_VISIBLE_DEVICES now (may not take effect if torch was already imported)")
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # Set random seed
    set_seed(SEED)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get model configuration
    model_config = get_model_config(args.model, args.target)
    
    # Validate retain set for target
    if args.target == "bio" and args.retain == "book":
        print("Warning: 'book' retain set not typically used with 'bio' target. Using 'wiki' instead.")
        args.retain = "wiki"
    
    print("\n" + "="*80)
    print("UNLEARNING EXPERIMENT CONFIGURATION")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Target: {args.target}")
    print(f"Retain: {args.retain}")
    print(f"Dataset sizes: {args.dataset_sizes}")
    print(f"Vary mode: {args.vary_dataset}")
    print(f"Output directory: {args.output_dir}")
    print(f"GPU: {args.gpu}")
    print(f"Max length: {args.max_length}")
    print("="*80)
    
    # Get maximum dataset size for varying modes
    max_n_examples = max(args.dataset_sizes) if args.dataset_sizes else 2500
    
    # Evaluate original model once (before any unlearning)
    print("\n" + "="*80)
    print("EVALUATING ORIGINAL MODEL (once for all experiments)")
    print("="*80)
    crisp_original = initialize_model(model_config)
    metrics_before = evaluate_model(crisp_original, args.target, eval_type="before")
    
    # Clean up original model to free memory
    del crisp_original
    torch.cuda.empty_cache()
    gc.collect()
    
    print("\n" + "="*80)
    print("STARTING EXPERIMENTS WITH VARYING DATASET SIZES")
    print("="*80)
    
    # Run experiments for each dataset size
    all_results = []
    
    for n_examples in args.dataset_sizes:
        # Check if results already exist
        vary_suffix = f"_vary_{args.vary_dataset}" if args.vary_dataset != "both" else ""
        exp_filename = f"experiment_n{n_examples}_{args.target}_{args.retain}_{model_config['model_name_short']}{vary_suffix}.json"
        exp_path = os.path.join(args.output_dir, exp_filename)
        
        if args.skip_existing and os.path.exists(exp_path):
            print(f"\n{'='*80}")
            print(f"Skipping n_examples={n_examples} (results already exist)")
            print(f"{'='*80}")
            with open(exp_path, 'r') as f:
                results = json.load(f)
            all_results.append(results)
            continue
        
        try:
            results = run_single_experiment(
                n_examples=n_examples,
                target=args.target,
                retain=args.retain,
                max_length=args.max_length,
                model_config=model_config,
                output_dir=args.output_dir,
                metrics_before=metrics_before,
                vary_mode=args.vary_dataset,
                max_n_examples=max_n_examples
            )
            all_results.append(results)
            
        except Exception as e:
            print(f"\n{'!'*80}")
            print(f"ERROR in experiment with n_examples={n_examples}: {str(e)}")
            print(f"{'!'*80}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save summary results
    if all_results:
        save_summary_results(all_results, args.output_dir, args)
    else:
        print("\nNo experiments completed successfully.")


if __name__ == "__main__":
    main()
