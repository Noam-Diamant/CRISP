#!/usr/bin/env python3
"""
Script to analyze and visualize unlearning experiment results.

This script loads experiment results from JSON files and generates
plots and statistics for analysis.
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
# This script doesn't need to change working directory since it only processes JSON files
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze unlearning experiment results"
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
        default="analysis_plots",
        help="Directory to save analysis plots (default: analysis_plots)"
    )
    
    parser.add_argument(
        "--summary-file",
        type=str,
        default=None,
        help="Specific summary JSON file to analyze (optional)"
    )
    
    return parser.parse_args()


def load_results(results_dir: str, summary_file: str = None) -> Dict[str, Any]:
    """Load experiment results from JSON files."""
    if summary_file:
        # Load specific summary file
        with open(summary_file, 'r') as f:
            return json.load(f)
    else:
        # Find all summary files in the directory
        summary_files = glob.glob(os.path.join(results_dir, "summary_*.json"))
        
        if not summary_files:
            raise FileNotFoundError(f"No summary files found in {results_dir}")
        
        # Use the most recent summary file
        summary_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        print(f"Loading results from: {summary_files[0]}")
        
        with open(summary_files[0], 'r') as f:
            return json.load(f)


def create_accuracy_plot(results: List[Dict[str, Any]], target: str, output_dir: str):
    """Create plot showing accuracy vs dataset size."""
    n_examples = [r['n_examples'] for r in results]
    
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
    ax1.plot(n_examples, target_before, 'o-', label='Before Unlearning', linewidth=2, markersize=8)
    ax1.plot(n_examples, target_after, 's-', label='After Unlearning', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Examples', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title(f'{target_label} vs Dataset Size', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    
    # Retain domain accuracy
    ax2.plot(n_examples, retain_before, 'o-', label='Before Unlearning', linewidth=2, markersize=8)
    ax2.plot(n_examples, retain_after, 's-', label='After Unlearning', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Examples', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('MMLU Accuracy vs Dataset Size', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'accuracy_vs_dataset_size_{target}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.close()


def create_drop_plot(results: List[Dict[str, Any]], target: str, output_dir: str):
    """Create plot showing accuracy drop vs dataset size."""
    n_examples = [r['n_examples'] for r in results]
    target_drop_pct = [r['target_accuracy_drop_percent'] for r in results]
    retain_drop_pct = [r['retain_accuracy_drop_percent'] for r in results]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(n_examples, target_drop_pct, 'o-', label='Target Domain Drop', 
            linewidth=2, markersize=8, color='red')
    ax.plot(n_examples, retain_drop_pct, 's-', label='Retain Domain Drop', 
            linewidth=2, markersize=8, color='blue')
    ax.set_xlabel('Number of Examples', fontsize=12)
    ax.set_ylabel('Accuracy Drop (%)', fontsize=12)
    ax.set_title('Accuracy Drop vs Dataset Size', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'accuracy_drop_vs_dataset_size_{target}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.close()


def create_tradeoff_plot(results: List[Dict[str, Any]], target: str, output_dir: str):
    """Create plot showing tradeoff between target and retain accuracy."""
    target_drop_pct = [r['target_accuracy_drop_percent'] for r in results]
    retain_drop_pct = [r['retain_accuracy_drop_percent'] for r in results]
    n_examples = [r['n_examples'] for r in results]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scatter = ax.scatter(retain_drop_pct, target_drop_pct, c=np.log10(n_examples), 
                        s=200, cmap='viridis', edgecolors='black', linewidth=1.5)
    
    # Add annotations for each point
    for i, n in enumerate(n_examples):
        ax.annotate(str(n), (retain_drop_pct[i], target_drop_pct[i]), 
                   fontsize=8, ha='center', va='center')
    
    ax.set_xlabel('Retain Domain Accuracy Drop (%)', fontsize=12)
    ax.set_ylabel('Target Domain Accuracy Drop (%)', fontsize=12)
    ax.set_title('Unlearning Tradeoff: Target vs Retain', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('log10(N Examples)', fontsize=10)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'tradeoff_plot_{target}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.close()


def create_statistics_table(results: List[Dict[str, Any]], target: str, output_dir: str):
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
        data.append({
            'N Examples': r['n_examples'],
            'Target Acc (Before)': f"{r['metrics_before'][target_key]:.4f}",
            'Target Acc (After)': f"{r['metrics_after'][target_key]:.4f}",
            'Target Drop (%)': f"{r['target_accuracy_drop_percent']:.2f}",
            'MMLU Acc (Before)': f"{r['metrics_before'][retain_key]:.4f}",
            'MMLU Acc (After)': f"{r['metrics_after'][retain_key]:.4f}",
            'MMLU Drop (%)': f"{r['retain_accuracy_drop_percent']:.2f}",
        })
    
    df = pd.DataFrame(data)
    
    # Save as CSV
    csv_path = os.path.join(output_dir, f'statistics_table_{target}.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved statistics table: {csv_path}")
    
    # Save as markdown
    md_path = os.path.join(output_dir, f'statistics_table_{target}.md')
    with open(md_path, 'w') as f:
        f.write(df.to_markdown(index=False))
    print(f"Saved markdown table: {md_path}")
    
    return df


def print_summary_statistics(results: List[Dict[str, Any]], target: str):
    """Print summary statistics to console."""
    n_experiments = len(results)
    
    target_drops = [r['target_accuracy_drop_percent'] for r in results]
    retain_drops = [r['retain_accuracy_drop_percent'] for r in results]
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Number of experiments: {n_experiments}")
    print(f"Dataset sizes: {[r['n_examples'] for r in results]}")
    print()
    print("Target Domain Accuracy Drop:")
    print(f"  Mean: {np.mean(target_drops):.2f}%")
    print(f"  Std:  {np.std(target_drops):.2f}%")
    print(f"  Min:  {np.min(target_drops):.2f}% (n={results[np.argmin(target_drops)]['n_examples']})")
    print(f"  Max:  {np.max(target_drops):.2f}% (n={results[np.argmax(target_drops)]['n_examples']})")
    print()
    print("Retain Domain Accuracy Drop:")
    print(f"  Mean: {np.mean(retain_drops):.2f}%")
    print(f"  Std:  {np.std(retain_drops):.2f}%")
    print(f"  Min:  {np.min(retain_drops):.2f}% (n={results[np.argmin(retain_drops)]['n_examples']})")
    print(f"  Max:  {np.max(retain_drops):.2f}% (n={results[np.argmax(retain_drops)]['n_examples']})")
    print("="*80)


def main():
    """Main execution function."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results
    print("Loading experiment results...")
    data = load_results(args.results_dir, args.summary_file)
    
    results = data['results']
    target = data['experiment_info']['target']
    
    print(f"Loaded {len(results)} experiments")
    print(f"Target: {target}")
    print(f"Model: {data['experiment_info']['model']}")
    
    # Create plots
    print("\nGenerating plots...")
    create_accuracy_plot(results, target, args.output_dir)
    create_drop_plot(results, target, args.output_dir)
    create_tradeoff_plot(results, target, args.output_dir)
    
    # Create statistics table
    print("\nGenerating statistics...")
    df = create_statistics_table(results, target, args.output_dir)
    
    # Print summary
    print_summary_statistics(results, target)
    
    print(f"\nAll outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
