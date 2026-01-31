#!/usr/bin/env python3
"""
Script to analyze and visualize feature variation experiment results.

This script loads experiment results from JSON files (feature variation experiments)
and generates plots and statistics for analysis.
"""

import argparse
import json
import os
import sys
import glob
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Get the parent directory (CRISP/crisp/)
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze feature variation experiment results"
    )
    
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Directory containing experiment results"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis_plots_features",
        help="Directory to save analysis plots (default: analysis_plots_features)"
    )
    
    parser.add_argument(
        "--summary-file",
        type=str,
        default=None,
        help="Specific summary JSON file to analyze (optional)"
    )
    
    return parser.parse_args()


def load_results(results_dir: str, summary_file: str = None) -> List[Dict[str, Any]]:
    """Load experiment results from JSON files.
    
    Returns:
        List of dictionaries, each containing 'data' and 'filepath' keys.
        If summary_file is provided, returns a single-item list.
    """
    if summary_file:
        # Load specific summary file
        with open(summary_file, 'r') as f:
            return [{'data': json.load(f), 'filepath': summary_file}]
    else:
        # Find all summary files in the directory
        summary_files = glob.glob(os.path.join(results_dir, "summary_*.json"))
        
        if not summary_files:
            raise FileNotFoundError(f"No summary files found in {results_dir}")
        
        # Sort by modification time (most recent first) for consistent ordering
        summary_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        # Load all summary files
        results_list = []
        for filepath in summary_files:
            with open(filepath, 'r') as f:
                results_list.append({'data': json.load(f), 'filepath': filepath})
        
        print(f"Found {len(results_list)} summary file(s)")
        for item in results_list:
            print(f"  - {os.path.basename(item['filepath'])}")
        
        return results_list


def create_accuracy_plot(results: List[Dict[str, Any]], target: str, model: str, output_dir: str, experiment_info: Dict[str, Any]):
    """Create plot showing accuracy vs k_features."""
    k_features = [r['k_features'] for r in results]
    
    # Determine target key based on target type
    if target == "hp":
        target_key = "hp_accuracy"
        target_label = "HP Accuracy"
    else:
        target_key = "wmdp_bio_accuracy"
        target_label = "WMDP Bio Accuracy"
    
    retain_key = "mmlu_accuracy"
    
    target_before = [r['metrics_before'][target_key] for r in results]
    target_after = [r['metrics_after'][target_key] for r in results]
    retain_before = [r['metrics_before'][retain_key] for r in results]
    retain_after = [r['metrics_after'][retain_key] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Target domain accuracy
    ax1.plot(k_features, target_before, 'o-', label='Before Unlearning', linewidth=2, markersize=8)
    ax1.plot(k_features, target_after, 's-', label='After Unlearning', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Features (k_features)', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title(f'{target_label} vs Number of Features', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Retain domain accuracy
    ax2.plot(k_features, retain_before, 'o-', label='Before Unlearning', linewidth=2, markersize=8)
    ax2.plot(k_features, retain_after, 's-', label='After Unlearning', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Features (k_features)', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title(f'MMLU Accuracy vs Number of Features', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Create filename suffix from experiment configuration
    suffix = get_filename_suffix(experiment_info)
    output_path = os.path.join(output_dir, f'accuracy_vs_k_features_{target}_{model}{suffix}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.close()


def create_drop_plot(results: List[Dict[str, Any]], target: str, model: str, output_dir: str, experiment_info: Dict[str, Any]):
    """Create plot showing accuracy drop vs k_features."""
    k_features = [r['k_features'] for r in results]
    target_drop_pct = [r['target_accuracy_drop_percent'] for r in results]
    retain_drop_pct = [r['retain_accuracy_drop_percent'] for r in results]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(k_features, target_drop_pct, 'o-', label='Target Domain Drop', 
            linewidth=2, markersize=8, color='red')
    ax.plot(k_features, retain_drop_pct, 's-', label='Retain Domain Drop', 
            linewidth=2, markersize=8, color='blue')
    ax.set_xlabel('Number of Features (k_features)', fontsize=12)
    ax.set_ylabel('Accuracy Drop (%)', fontsize=12)
    ax.set_title(f'Accuracy Drop vs Number of Features', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    suffix = get_filename_suffix(experiment_info)
    output_path = os.path.join(output_dir, f'accuracy_drop_vs_k_features_{target}_{model}{suffix}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.close()


def create_random_features_plot(results: List[Dict[str, Any]], target: str, model: str, output_dir: str, experiment_info: Dict[str, Any]):
    """Create plot showing effect of random features if supplementation was used."""
    # Check if any results have random features
    has_random = any(r.get('n_random_features', 0) > 0 for r in results)
    
    if not has_random:
        print("No random feature supplementation detected, skipping random features plot")
        return
    
    k_features = [r['k_features'] for r in results]
    n_random = [r.get('n_random_features', 0) for r in results]
    total_features = [r.get('total_features', r['k_features']) for r in results]
    target_drop_pct = [r['target_accuracy_drop_percent'] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Stacked bar chart showing learned vs random features
    width = 0.6
    ax1.bar(k_features, k_features, width, label='Learned Features', color='steelblue')
    ax1.bar(k_features, n_random, width, bottom=k_features, label='Random Features', color='coral')
    ax1.set_xlabel('k_features (Learned)', fontsize=12)
    ax1.set_ylabel('Total Number of Features', fontsize=12)
    ax1.set_title('Feature Composition: Learned vs Random', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Effect of random features on unlearning
    ax2.plot(n_random, target_drop_pct, 'o-', linewidth=2, markersize=8, color='purple')
    ax2.set_xlabel('Number of Random Features', fontsize=12)
    ax2.set_ylabel('Target Domain Accuracy Drop (%)', fontsize=12)
    ax2.set_title('Unlearning Effectiveness vs Random Features', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    suffix = get_filename_suffix(experiment_info)
    output_path = os.path.join(output_dir, f'random_features_analysis_{target}_{model}{suffix}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.close()


def create_tradeoff_plot(results: List[Dict[str, Any]], target: str, model: str, output_dir: str, experiment_info: Dict[str, Any]):
    """Create plot showing tradeoff between target and retain accuracy."""
    target_drop_pct = [r['target_accuracy_drop_percent'] for r in results]
    retain_drop_pct = [r['retain_accuracy_drop_percent'] for r in results]
    k_features = [r['k_features'] for r in results]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Use k_features for coloring
    scatter = ax.scatter(retain_drop_pct, target_drop_pct, c=k_features, 
                        s=200, cmap='viridis', edgecolors='black', linewidth=1.5)
    
    # Add annotations for each point
    for i, k in enumerate(k_features):
        ax.annotate(str(k), (retain_drop_pct[i], target_drop_pct[i]), 
                   fontsize=8, ha='center', va='center')
    
    ax.set_xlabel('Retain Domain Accuracy Drop (%)', fontsize=12)
    ax.set_ylabel('Target Domain Accuracy Drop (%)', fontsize=12)
    ax.set_title(f'Unlearning Tradeoff: Target vs Retain', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('k_features', fontsize=10)
    
    plt.tight_layout()
    suffix = get_filename_suffix(experiment_info)
    output_path = os.path.join(output_dir, f'tradeoff_plot_{target}_{model}{suffix}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.close()


def get_filename_suffix(experiment_info: Dict[str, Any]) -> str:
    """Generate filename suffix based on experiment configuration, including timestamp."""
    suffix_parts = []
    
    if experiment_info.get('supplement_with_random'):
        suffix_parts.append(f"supp{experiment_info['supplement_with_random']}")
    if experiment_info.get('n_samples_extraction'):
        suffix_parts.append(f"ext{experiment_info['n_samples_extraction']}")
    if experiment_info.get('n_samples_unlearning'):
        suffix_parts.append(f"unl{experiment_info['n_samples_unlearning']}")
    
    # Add timestamp if available (from experiment_info)
    if experiment_info.get('timestamp'):
        # Format timestamp for filename (replace spaces and colons)
        timestamp_str = experiment_info['timestamp'].replace(' ', '_').replace(':', '-')
        suffix_parts.append(timestamp_str)
    
    return "_" + "_".join(suffix_parts) if suffix_parts else ""


def create_statistics_table(results: List[Dict[str, Any]], target: str, model: str, output_dir: str, experiment_info: Dict[str, Any]):
    """Create and save statistics table."""
    # Determine target key
    if target == "hp":
        target_key = "hp_accuracy"
    else:
        target_key = "wmdp_bio_accuracy"
    
    retain_key = "mmlu_accuracy"
    
    # Create DataFrame
    data = []
    for r in results:
        row = {
            'k_features': r['k_features'],
        }
        
        # Add timestamp if available
        if 'timestamp' in r:
            row['Timestamp'] = r['timestamp']
        
        # Add random features info if available
        if r.get('n_random_features', 0) > 0:
            row['Random Features'] = f"{r.get('n_random_features', 0):.1f}"
            row['Total Features'] = f"{r.get('total_features', r['k_features']):.1f}"
        
        # Add sample sizes if they're fixed
        if experiment_info.get('n_samples_extraction'):
            row['N Samples (Ext)'] = experiment_info['n_samples_extraction']
        if experiment_info.get('n_samples_unlearning'):
            row['N Samples (Unl)'] = experiment_info['n_samples_unlearning']
        
        row.update({
            'Target Acc (Before)': f"{r['metrics_before'][target_key]:.4f}",
            'Target Acc (After)': f"{r['metrics_after'][target_key]:.4f}",
            'Target Drop (%)': f"{r['target_accuracy_drop_percent']:.2f}",
            'MMLU Acc (Before)': f"{r['metrics_before'][retain_key]:.4f}",
            'MMLU Acc (After)': f"{r['metrics_after'][retain_key]:.4f}",
            'MMLU Drop (%)': f"{r['retain_accuracy_drop_percent']:.2f}",
        })
        
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Save as CSV
    suffix = get_filename_suffix(experiment_info)
    csv_path = os.path.join(output_dir, f'statistics_table_{target}_{model}{suffix}.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved statistics table: {csv_path}")
    
    # Save as markdown
    md_path = os.path.join(output_dir, f'statistics_table_{target}_{model}{suffix}.md')
    with open(md_path, 'w') as f:
        f.write(df.to_markdown(index=False))
    print(f"Saved markdown table: {md_path}")
    
    return df


def print_summary_statistics(results: List[Dict[str, Any]], target: str, experiment_info: Dict[str, Any]):
    """Print summary statistics to console."""
    n_experiments = len(results)
    
    target_drops = [r['target_accuracy_drop_percent'] for r in results]
    retain_drops = [r['retain_accuracy_drop_percent'] for r in results]
    k_features_list = [r['k_features'] for r in results]
    
    # Extract timestamps if available
    timestamps = [r.get('timestamp', 'N/A') for r in results]
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS - FEATURE VARIATION EXPERIMENTS")
    print("="*80)
    print(f"Number of experiments: {n_experiments}")
    print(f"k_features tested: {k_features_list}")
    
    # Display timestamp range
    valid_timestamps = [ts for ts in timestamps if ts != 'N/A']
    if valid_timestamps:
        print(f"Experiment timespan: {valid_timestamps[0]} to {valid_timestamps[-1]}")
    
    print()
    
    # Sample size info
    if experiment_info.get('n_samples_extraction'):
        print(f"N samples (extraction): {experiment_info['n_samples_extraction']}")
    else:
        print("N samples (extraction): all available")
    
    if experiment_info.get('n_samples_unlearning'):
        print(f"N samples (unlearning): {experiment_info['n_samples_unlearning']}")
    else:
        print("N samples (unlearning): all available")
    
    # Random feature supplementation info
    if experiment_info.get('supplement_with_random'):
        print(f"Supplementing with random features to: {experiment_info['supplement_with_random']} total")
        avg_random = np.mean([r.get('n_random_features', 0) for r in results])
        print(f"  Average random features added: {avg_random:.1f}")
    else:
        print("Random feature supplementation: disabled")
    
    print()
    print("Target Domain Accuracy Drop:")
    print(f"  Mean: {np.mean(target_drops):.2f}%")
    print(f"  Std:  {np.std(target_drops):.2f}%")
    print(f"  Min:  {np.min(target_drops):.2f}% (k={k_features_list[np.argmin(target_drops)]})")
    print(f"  Max:  {np.max(target_drops):.2f}% (k={k_features_list[np.argmax(target_drops)]})")
    print()
    print("Retain Domain Accuracy Drop:")
    print(f"  Mean: {np.mean(retain_drops):.2f}%")
    print(f"  Std:  {np.std(retain_drops):.2f}%")
    print(f"  Min:  {np.min(retain_drops):.2f}% (k={k_features_list[np.argmin(retain_drops)]})")
    print(f"  Max:  {np.max(retain_drops):.2f}% (k={k_features_list[np.argmax(retain_drops)]})")
    print("="*80)


def process_single_summary(data: Dict[str, Any], output_dir: str, filepath: str):
    """Process a single summary file and generate all outputs."""
    results = data['results']
    experiment_info = data['experiment_info']
    target = experiment_info['target']
    model = experiment_info['model']
    
    # Sort results by k_features for consistent ordering
    results = sorted(results, key=lambda x: x['k_features'])
    
    # Extract model name for filename (handle paths like "google/gemma-2-2b")
    model_short = model.split('/')[-1].replace('.', '_')
    
    print(f"\n{'='*80}")
    print(f"Processing: {os.path.basename(filepath)}")
    print(f"{'='*80}")
    print(f"Loaded {len(results)} experiments")
    print(f"Target: {target}")
    print(f"Model: {model}")
    
    # Validate this is a feature variation experiment
    if 'feature_counts' not in experiment_info:
        print("WARNING: This appears to be a dataset size experiment, not a feature variation experiment.")
        print("Please use the original analyze_results.py for dataset size experiments.")
        return
    
    # Create plots
    print("\nGenerating plots...")
    create_accuracy_plot(results, target, model_short, output_dir, experiment_info)
    create_drop_plot(results, target, model_short, output_dir, experiment_info)
    create_random_features_plot(results, target, model_short, output_dir, experiment_info)
    create_tradeoff_plot(results, target, model_short, output_dir, experiment_info)
    
    # Create statistics table
    print("\nGenerating statistics...")
    df = create_statistics_table(results, target, model_short, output_dir, experiment_info)
    
    # Print summary
    print_summary_statistics(results, target, experiment_info)


def main():
    """Main execution function."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results
    print("Loading experiment results...")
    results_list = load_results(args.results_dir, args.summary_file)
    
    # Process each summary file separately
    for item in results_list:
        process_single_summary(item['data'], args.output_dir, item['filepath'])
    
    print(f"\n{'='*80}")
    print(f"All outputs saved to: {args.output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
