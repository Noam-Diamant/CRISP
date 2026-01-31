#!/usr/bin/env python3
"""
Script to run unlearning experiments with varying numbers of features (k_features).

This script performs systematic unlearning experiments by varying the number of
features selected for unlearning. It can optionally fix the number of samples used
for feature extraction and unlearning, and supplement learned features with random ones.

Special case: Setting k_features=0 with --supplement-with-random allows testing
unlearning with purely random features (no salient features).
Example: --feature-counts 0 --supplement-with-random 10
"""

import argparse
import json
import os
import sys
import random
import glob

# IMPORTANT: Set CUDA_VISIBLE_DEVICES BEFORE importing torch
# Parse --gpu argument early to set CUDA_VISIBLE_DEVICES before torch import
# This must happen before torch import because PyTorch checks CUDA availability during import
gpu_arg = "0"  # default (matches argparse default)
if "--gpu" in sys.argv:
    gpu_idx = sys.argv.index("--gpu")
    if gpu_idx + 1 < len(sys.argv):
        gpu_arg = sys.argv[gpu_idx + 1]
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_arg

import torch
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
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
        description="Run unlearning experiments with varying numbers of features (k_features)"
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
        "--feature-counts",
        type=int,
        nargs="+",
        default=[1, 2, 3, 5, 7, 10, 15, 20, 25, 30],
        help="List of k_features values to experiment with. Use 0 with --supplement-with-random for purely random features (default: 1 2 3 5 7 10 15 20 25 30)"
    )
    
    parser.add_argument(
        "--n-samples-extraction",
        type=int,
        default=10,
        help="Fixed number of samples for feature extraction (default: use all available data)"
    )
    
    parser.add_argument(
        "--n-samples-unlearning",
        type=int,
        default=2500,
        help="Fixed number of samples for unlearning (default: use all available data)"
    )
    
    parser.add_argument(
        "--supplement-with-random",
        type=int,
        default=None,
        help="Total number of features to reach by supplementing with random features (default: None)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiment_results_features",
        help="Directory to save experiment results (default: experiment_results_features)"
    )
    
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
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


def load_data(target: str, retain: str, n_examples: Optional[int], max_length: int):
    """Load forget and retain datasets."""
    if target == "hp":
        print(f"Loading HP data with {n_examples if n_examples else 'all'} examples, retain type: {retain}")
        data = load_hp_data(benign=retain, n_examples=n_examples, max_len=max_length)
        return data["forget"], data["retain"]
    else:  # bio
        print(f"Loading WMDP Bio data with {n_examples if n_examples else 'all'} examples, retain type: {retain}")
        data = load_wmdp_data(target_type="bio", retain_type=retain, n_examples=n_examples)
        return data["forget"], data["retain"]


def create_data_config(target: str, retain: str, n_examples: Optional[int], max_length: int):
    """Create data configuration object."""
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


def supplement_features_with_random(
    salient_features: torch.Tensor,
    total_features: int,
    max_features_available: int,
    layer_idx: int,
    seed: int = SEED
) -> Tuple[torch.Tensor, int]:
    """
    Supplement salient features with random features.
    
    Args:
        salient_features: Tensor of salient feature indices
        total_features: Total number of features to reach
        max_features_available: Maximum number of features in the SAE
        layer_idx: Layer index (for reproducibility)
        seed: Random seed base
    
    Returns:
        combined_features: All features (salient + random)
        n_random: Number of random features added
    """
    n_salient = len(salient_features)
    n_random_needed = total_features - n_salient
    
    if n_random_needed <= 0:
        return salient_features, 0
    
    # Set seed for reproducibility (different for each layer)
    random.seed(seed + layer_idx)
    torch.manual_seed(seed + layer_idx)
    
    # Generate random features excluding salient ones
    available = list(set(range(max_features_available)) - set(salient_features.tolist()))
    
    if n_random_needed > len(available):
        print(f"Warning: Requested {n_random_needed} random features but only {len(available)} available. Using all available.")
        n_random_needed = len(available)
    
    random_indices = random.sample(available, n_random_needed)
    random_features = torch.tensor(random_indices, dtype=salient_features.dtype, device=salient_features.device)
    
    combined = torch.cat([salient_features, random_features])
    return combined, n_random_needed


class FeatureSupplementedCRISP(CRISP):
    """
    Extended CRISP class that supports supplementing salient features with random ones.
    """
    
    def __init__(self, config: CRISPConfig, supplement_total: Optional[int] = None):
        super().__init__(config)
        self.supplement_total = supplement_total
        self.n_random_features_per_layer = {}
    
    def get_salient_features(self, layer_idx, k_features, topk_filter: bool = True):
        """
        Override to optionally supplement with random features.
        Handles the special case where k_features=0 and only random features are used.
        """
        # If supplementation is not enabled, use parent method as-is
        if self.supplement_total is None or self.supplement_total <= k_features:
            salient_features = super().get_salient_features(layer_idx, k_features, topk_filter)
            self.n_random_features_per_layer[layer_idx] = 0
            return salient_features
        
        # Get the SAE for this layer to determine max features
        layer_name = f"layers.{layer_idx}"
        sae = self.model_saes.saes[layer_name]
        
        # Get max features from SAE dimensions
        if hasattr(sae, 'd_sae'):
            max_features = sae.d_sae
        elif hasattr(sae, 'num_latents'):
            max_features = sae.num_latents
        else:
            # Fallback: try to infer from encoder weight shape
            max_features = sae.encoder.weight.shape[0] if hasattr(sae, 'encoder') else 32768
        
        # Handle k=0 case: only random features
        if k_features == 0:
            #print(f"  Layer {layer_idx}: Using {self.supplement_total} random features (k=0)")
            # Create empty tensor for salient features with correct device
            device = next(self.model.parameters()).device
            salient_features = torch.tensor([], dtype=torch.long, device=device)
            
            # Generate purely random features
            combined_features, n_random = supplement_features_with_random(
                salient_features=salient_features,
                total_features=self.supplement_total,
                max_features_available=max_features,
                layer_idx=layer_idx
            )
            
            self.n_random_features_per_layer[layer_idx] = n_random
            return combined_features
        
        # Get salient features using parent method
        salient_features = super().get_salient_features(layer_idx, k_features, topk_filter)
        
        # Supplement with random features
        combined_features, n_random = supplement_features_with_random(
            salient_features=salient_features,
            total_features=self.supplement_total,
            max_features_available=max_features,
            layer_idx=layer_idx
        )
        
        self.n_random_features_per_layer[layer_idx] = n_random
        
        return combined_features


def run_single_experiment(
    k_features: int,
    n_samples_extraction: Optional[int],
    n_samples_unlearning: Optional[int],
    supplement_with_random: Optional[int],
    target: str,
    retain: str,
    max_length: int,
    model_config: Dict[str, Any],
    output_dir: str,
    metrics_before: Dict[str, float],
    timestamp: str,
) -> Dict[str, Any]:
    """Run a single unlearning experiment for a given k_features value.
    
    Args:
        k_features: Number of salient features to select
        n_samples_extraction: Number of samples for feature extraction (None = all)
        n_samples_unlearning: Number of samples for unlearning (None = all)
        supplement_with_random: Total features to reach with random supplementation (None = no supplementation)
        target: Target domain (hp, bio)
        retain: Retain set type
        max_length: Maximum text length
        model_config: Model configuration dict
        output_dir: Output directory for results
        metrics_before: Metrics from original model evaluation
        timestamp: Timestamp string for file naming
    """
    
    print("\n" + "="*80)
    print(f"Running experiment with k_features={k_features}")
    if k_features == 0 and supplement_with_random:
        print(f"  Using ONLY {supplement_with_random} random features (k=0, no salient features)")
    elif supplement_with_random:
        print(f"  Supplementing to {supplement_with_random} total features with random features")
    if n_samples_extraction:
        print(f"  Using {n_samples_extraction} samples for feature extraction")
    if n_samples_unlearning:
        print(f"  Using {n_samples_unlearning} samples for unlearning")
    print("="*80)
    
    # Initialize model (potentially with feature supplementation)
    if supplement_with_random:
        crisp_config = CRISPConfig(
            layers=model_config['sae_layers'],
            model_name=model_config['model_card'],
            bf16=True
        )
        crisp = FeatureSupplementedCRISP(config=crisp_config, supplement_total=supplement_with_random)
    else:
        crisp = initialize_model(model_config)
    
    # Load data for feature extraction
    print(f"\n--- Loading data for feature extraction ---")
    forget_data_features, retain_data_features = load_data(
        target, retain, n_samples_extraction, max_length
    )
    
    # Create data config for feature extraction
    data_config_features = create_data_config(target, retain, n_samples_extraction, max_length)
    
    # Create unlearn config with varying k_features
    unlearn_config = UnlearnConfig(
        data_type=target,
        learning_rate=model_config['unlearn']['learning_rate'],
        k_features=k_features,
        alpha=model_config['unlearn']['alpha'],
        save_model=False,  # Do not save models to save disk space
        beta=0.99,
        gamma=0.01,
        batch_size=4,
        lora_rank=4,
        verbose=target
    )
    
    # Handle feature extraction and unlearning data
    if n_samples_unlearning is None or n_samples_unlearning == n_samples_extraction:
        # Use same data for both
        forget_data_unlearn = forget_data_features
        retain_data_unlearn = retain_data_features
        data_config_unlearn = data_config_features
    else:
        # Pre-compute features with extraction data
        print(f"\n--- Extracting features ({n_samples_extraction if n_samples_extraction else 'all'} samples) ---")
        crisp.process_multi_texts_batch(
            text_target=forget_data_features,
            text_benign=retain_data_features,
            data_config=data_config_features,
            batch_size=unlearn_config.batch_size
        )
        
        # Load different data for unlearning
        print(f"\n--- Loading data for unlearning ({n_samples_unlearning if n_samples_unlearning else 'all'} samples) ---")
        forget_data_unlearn, retain_data_unlearn = load_data(
            target, retain, n_samples_unlearning, max_length
        )
        data_config_unlearn = create_data_config(target, retain, n_samples_unlearning, max_length)
    
    # Perform unlearning
    print(f"\n--- Performing Unlearning (k_features={k_features}) ---")
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
    
    # Calculate random features stats
    n_random_features = 0
    total_features_actual = k_features
    if supplement_with_random and isinstance(crisp, FeatureSupplementedCRISP):
        # Average across layers
        if crisp.n_random_features_per_layer:
            n_random_features = sum(crisp.n_random_features_per_layer.values()) / len(crisp.n_random_features_per_layer)
            total_features_actual = k_features + n_random_features
    
    # Compile results
    results = {
        "k_features": k_features,
        "n_random_features": float(n_random_features),
        "total_features": float(total_features_actual),
        "supplement_with_random": supplement_with_random,
        "n_samples_extraction": n_samples_extraction,
        "n_samples_unlearning": n_samples_unlearning,
        "target": target,
        "retain": retain,
        "model": model_config['model_card'],
        "max_length": max_length,
        "timestamp": timestamp,
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
    suffix_parts = []
    if supplement_with_random:
        suffix_parts.append(f"supp{supplement_with_random}")
    if n_samples_extraction:
        suffix_parts.append(f"ext{n_samples_extraction}")
    if n_samples_unlearning:
        suffix_parts.append(f"unl{n_samples_unlearning}")
    suffix = "_" + "_".join(suffix_parts) if suffix_parts else ""
    
    # Add timestamp to filename
    exp_filename = f"experiment_k{k_features}_{target}_{retain}_{model_config['model_name_short']}{suffix}_{timestamp.replace(' ', '_').replace(':', '-')}.json"
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
    
    suffix_parts = []
    if args.supplement_with_random:
        suffix_parts.append(f"supp{args.supplement_with_random}")
    if args.n_samples_extraction:
        suffix_parts.append(f"ext{args.n_samples_extraction}")
    if args.n_samples_unlearning:
        suffix_parts.append(f"unl{args.n_samples_unlearning}")
    suffix = "_" + "_".join(suffix_parts) if suffix_parts else ""
    
    summary_filename = f"summary_{args.target}_{args.retain}_{args.model.replace('.', '_')}{suffix}_{timestamp}.json"
    summary_path = os.path.join(output_dir, summary_filename)
    
    summary = {
        "experiment_info": {
            "target": args.target,
            "retain": args.retain,
            "model": args.model,
            "max_length": args.max_length,
            "feature_counts": args.feature_counts,
            "n_samples_extraction": args.n_samples_extraction,
            "n_samples_unlearning": args.n_samples_unlearning,
            "supplement_with_random": args.supplement_with_random,
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
    
    header = f"{'k_features':<12} {'Total Feat':<12} {'Random Feat':<12} {'Target Before':<15} {'Target After':<15} {'Drop %':<10} {'Retain Before':<15} {'Retain After':<15} {'Drop %':<10}"
    print(header)
    print("-" * len(header))
    
    for result in all_results:
        k_features = result['k_features']
        total_features = result.get('total_features', k_features)
        n_random = result.get('n_random_features', 0)
        
        target_key = "hp_accuracy" if args.target == "hp" else "wmdp_bio_accuracy"
        retain_key = "mmlu_accuracy"
        
        target_before = result['metrics_before'][target_key]
        target_after = result['metrics_after'][target_key]
        target_drop = result['target_accuracy_drop_percent']
        
        retain_before = result['metrics_before'][retain_key]
        retain_after = result['metrics_after'][retain_key]
        retain_drop = result['retain_accuracy_drop_percent']
        
        print(f"{k_features:<12} {total_features:<12.1f} {n_random:<12.1f} {target_before:<15.4f} {target_after:<15.4f} {target_drop:<10.2f} "
              f"{retain_before:<15.4f} {retain_after:<15.4f} {retain_drop:<10.2f}")
    
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
    
    # Validate supplement_with_random
    if args.supplement_with_random is not None:
        max_k = max(args.feature_counts)
        if args.supplement_with_random < max_k:
            print(f"Warning: --supplement-with-random ({args.supplement_with_random}) is less than max k_features ({max_k})")
            print("Random supplementation will only apply when k_features < supplement_with_random")
    
    # Check if k=0 is used and ensure supplement_with_random is specified
    if 0 in args.feature_counts:
        if args.supplement_with_random is None or args.supplement_with_random <= 0:
            print("ERROR: When k_features=0, --supplement-with-random must be specified with a positive value.")
            print("Example: --feature-counts 0 --supplement-with-random 10")
            sys.exit(1)
        print(f"\nNote: k_features=0 detected. Will use {args.supplement_with_random} purely random features.")
    
    print("\n" + "="*80)
    print("FEATURE VARIATION EXPERIMENT CONFIGURATION")
    print("="*80)
    print(f"Seed: {SEED}")
    print(f"Model: {args.model}")
    print(f"Target: {args.target}")
    print(f"Retain: {args.retain}")
    print(f"Feature counts (k_features): {args.feature_counts}")
    print(f"N samples (extraction): {args.n_samples_extraction if args.n_samples_extraction else 'all'}")
    print(f"N samples (unlearning): {args.n_samples_unlearning if args.n_samples_unlearning else 'all'}")
    print(f"Supplement with random: {args.supplement_with_random if args.supplement_with_random else 'disabled'}")
    print(f"Output directory: {args.output_dir}")
    print(f"GPU: {args.gpu}")
    print(f"Max length: {args.max_length}")
    print("="*80)
    
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
    print("STARTING EXPERIMENTS WITH VARYING k_features")
    print("="*80)
    
    # Run experiments for each k_features value
    all_results = []
    
    # Generate a shared timestamp for this run of experiments
    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    for k_features in args.feature_counts:
        # Check if results already exist
        suffix_parts = []
        if args.supplement_with_random:
            suffix_parts.append(f"supp{args.supplement_with_random}")
        if args.n_samples_extraction:
            suffix_parts.append(f"ext{args.n_samples_extraction}")
        if args.n_samples_unlearning:
            suffix_parts.append(f"unl{args.n_samples_unlearning}")
        suffix = "_" + "_".join(suffix_parts) if suffix_parts else ""
        
        # Pattern to match experiment files with any timestamp
        exp_pattern = f"experiment_k{k_features}_{args.target}_{args.retain}_{model_config['model_name_short']}{suffix}_*.json"
        exp_matches = glob.glob(os.path.join(args.output_dir, exp_pattern))
        
        if args.skip_existing and exp_matches:
            print(f"\n{'='*80}")
            print(f"Skipping k_features={k_features} (results already exist)")
            print(f"{'='*80}")
            # Load the most recent matching file
            exp_path = max(exp_matches, key=os.path.getmtime)
            with open(exp_path, 'r') as f:
                results = json.load(f)
            all_results.append(results)
            continue
        
        try:
            results = run_single_experiment(
                k_features=k_features,
                n_samples_extraction=args.n_samples_extraction,
                n_samples_unlearning=args.n_samples_unlearning,
                supplement_with_random=args.supplement_with_random,
                target=args.target,
                retain=args.retain,
                max_length=args.max_length,
                model_config=model_config,
                output_dir=args.output_dir,
                metrics_before=metrics_before,
                timestamp=run_timestamp,
            )
            all_results.append(results)
            
        except Exception as e:
            print(f"\n{'!'*80}")
            print(f"ERROR in experiment with k_features={k_features}: {str(e)}")
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
