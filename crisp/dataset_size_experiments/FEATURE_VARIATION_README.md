# Feature Variation Experiments

This document explains how to use the `run_feature_variation_experiments.py` script to run unlearning experiments with varying numbers of features (k_features).

## Overview

The script extends the functionality of `run_unlearning_experiments.py` by allowing you to:
1. **Vary the number of features (k_features)** used for unlearning
2. **Fix the number of samples** used for feature extraction and/or unlearning
3. **Supplement learned features with random features** to reach a target total

## Command Line Arguments

### Core Arguments

- `--model`: Model to use (`gemma-2-2b` or `llama-3.1-8b`, default: `gemma-2-2b`)
- `--target`: Target domain to unlearn (`hp` or `bio`, default: `hp`)
- `--retain`: Retain set type (`book` or `wiki`, default: `book`)
- `--output-dir`: Directory to save results (default: `experiment_results_features`)
- `--gpu`: GPU device ID (default: `0`)

### Feature Variation (NEW)

- `--feature-counts`: List of k_features values to test
  - Default: `1 2 3 5 7 10 15 20 25 30`
  - Example: `--feature-counts 1 5 10 20`

### Sample Control (NEW)

- `--n-samples-extraction`: Fixed number of samples for feature extraction
  - Default: `None` (uses all available data)
  - Example: `--n-samples-extraction 100`

- `--n-samples-unlearning`: Fixed number of samples for unlearning
  - Default: `None` (uses all available data)
  - Example: `--n-samples-unlearning 500`

### Random Feature Supplementation (NEW)

- `--supplement-with-random`: Total number of features to reach by adding random features
  - Default: `None` (no supplementation)
  - Example: `--supplement-with-random 10`
  - If `k_features=3` and `--supplement-with-random 10`, then 7 random features will be added

### Other Arguments

- `--max-length`: Maximum text length (default: `1000`)
- `--skip-existing`: Skip experiments for which results already exist

## Usage Examples

### Example 1: Basic Feature Variation

Vary k_features from 1 to 10 with default sample sizes:

```bash
python run_feature_variation_experiments.py \
  --model gemma-2-2b \
  --target hp \
  --retain book \
  --feature-counts 1 2 3 5 7 10 \
  --output-dir results_basic
```

### Example 2: Fixed Sample Sizes

Vary k_features while using a fixed number of samples:

```bash
python run_feature_variation_experiments.py \
  --model gemma-2-2b \
  --target hp \
  --retain book \
  --feature-counts 1 3 5 10 \
  --n-samples-extraction 100 \
  --n-samples-unlearning 100 \
  --output-dir results_fixed_samples
```

### Example 3: Random Feature Supplementation

Test the effect of supplementing learned features with random ones:

```bash
python run_feature_variation_experiments.py \
  --model gemma-2-2b \
  --target hp \
  --retain book \
  --feature-counts 1 2 3 5 7 9 \
  --supplement-with-random 10 \
  --output-dir results_with_random
```

In this example:
- When k_features=1, 9 random features are added (total=10)
- When k_features=3, 7 random features are added (total=10)
- When k_features=9, 1 random feature is added (total=10)

### Example 4: Complete Configuration

Combine all options:

```bash
python run_feature_variation_experiments.py \
  --model gemma-2-2b \
  --target hp \
  --retain book \
  --feature-counts 1 3 5 7 10 \
  --n-samples-extraction 200 \
  --n-samples-unlearning 500 \
  --supplement-with-random 15 \
  --output-dir results_complete \
  --gpu 0
```

### Example 5: WMDP Bio Target

Run experiments on the WMDP Bio target:

```bash
python run_feature_variation_experiments.py \
  --model gemma-2-2b \
  --target bio \
  --retain wiki \
  --feature-counts 5 10 15 20 \
  --n-samples-extraction 100 \
  --output-dir results_bio
```

## Output Format

### Individual Experiment Files

Each experiment produces a JSON file named:
```
experiment_k{k_features}_{target}_{retain}_{model}[_supp{N}][_ext{N}][_unl{N}].json
```

Example: `experiment_k5_hp_book_gemma_supp10_ext100_unl500.json`

**Contents:**
```json
{
  "k_features": 5,
  "n_random_features": 5.0,
  "total_features": 10.0,
  "supplement_with_random": 10,
  "n_samples_extraction": 100,
  "n_samples_unlearning": 500,
  "target": "hp",
  "retain": "book",
  "model": "gemma-2-2b",
  "timestamp": "2024-01-24T12:00:00",
  "metrics_before": {
    "hp_accuracy": 0.85,
    "mmlu_accuracy": 0.72
  },
  "metrics_after": {
    "hp_accuracy": 0.42,
    "mmlu_accuracy": 0.70
  },
  "target_accuracy_drop": 0.43,
  "target_accuracy_drop_percent": 50.59,
  "retain_accuracy_drop": 0.02,
  "retain_accuracy_drop_percent": 2.78,
  ...
}
```

### Summary File

A summary JSON file is created with all results:
```
summary_{target}_{retain}_{model}[_supp{N}][_ext{N}][_unl{N}]_{timestamp}.json
```

**Contents:**
```json
{
  "experiment_info": {
    "target": "hp",
    "retain": "book",
    "model": "gemma-2-2b",
    "feature_counts": [1, 2, 3, 5, 7, 10],
    "n_samples_extraction": 100,
    "n_samples_unlearning": 500,
    "supplement_with_random": 10,
    "timestamp": "20240124_120000"
  },
  "results": [...]
}
```

### Console Summary Table

After all experiments complete, a summary table is printed:

```
================================================================================
EXPERIMENT SUMMARY
================================================================================
k_features   Total Feat   Random Feat  Target Before   Target After    Drop %     Retain Before   Retain After    Drop %    
--------------------------------------------------------------------------------
1            10.0         9.0          0.8500          0.7200          15.29      0.7200          0.7150          0.69      
3            10.0         7.0          0.8500          0.5100          40.00      0.7200          0.7100          1.39      
5            10.0         5.0          0.8500          0.4200          50.59      0.7200          0.7000          2.78      
...
================================================================================
```

## Implementation Details

### Feature Supplementation

The script includes a `FeatureSupplementedCRISP` class that extends the base `CRISP` class to support random feature supplementation:

1. First, salient features are selected using the standard method
2. If supplementation is enabled, random features are added:
   - Random features are selected from the SAE's feature space
   - Already-selected salient features are excluded
   - Uses a seeded random generator for reproducibility

### Sample Size Control

The script allows independent control of sample sizes for:
- **Feature extraction**: Number of samples used to identify salient features
- **Unlearning**: Number of samples used during the LoRA training phase

This allows you to test scenarios like:
- Extract features from a large dataset, but unlearn on a smaller one
- Extract features from a small dataset, but unlearn on a larger one
- Keep both the same (default behavior)

## Comparison with Original Script

| Feature | `run_unlearning_experiments.py` | `run_feature_variation_experiments.py` |
|---------|--------------------------------|---------------------------------------|
| Varies | Dataset size (`n_examples`) | Number of features (`k_features`) |
| Fixed k_features | ❌ (uses default 10) | ✅ (varies as specified) |
| Fixed sample sizes | ❌ | ✅ (optional) |
| Random supplementation | ❌ | ✅ (optional) |
| Output format | Same structure | Same structure (extended) |

## Tips

1. **Start small**: Test with a few feature counts first to ensure everything works
2. **Use `--skip-existing`**: Resume interrupted experiments without re-running
3. **GPU memory**: Each experiment loads the model fresh, so memory usage is controlled
4. **Random supplementation**: Use when you want to isolate the effect of the number of features vs. which specific features are selected
5. **Sample control**: Use to test if feature quality (from more data) vs. unlearning effectiveness (from more data) matters more

## Example Shell Script

Create a shell script to run multiple experiments:

```bash
#!/bin/bash

# Run feature variation experiments for HP target
python run_feature_variation_experiments.py \
  --model gemma-2-2b \
  --target hp \
  --retain book \
  --feature-counts 1 2 3 5 7 10 15 20 \
  --n-samples-extraction 200 \
  --n-samples-unlearning 200 \
  --output-dir results_hp_200samples \
  --gpu 0 \
  --skip-existing

# Run with random supplementation
python run_feature_variation_experiments.py \
  --model gemma-2-2b \
  --target hp \
  --retain book \
  --feature-counts 1 2 3 5 7 10 \
  --n-samples-extraction 200 \
  --n-samples-unlearning 200 \
  --supplement-with-random 15 \
  --output-dir results_hp_200samples_supp15 \
  --gpu 0 \
  --skip-existing
```

## Analyzing Results

After running experiments, use `analyze_feature_results.py` to generate plots and statistics:

### Basic Usage

```bash
python analyze_feature_results.py \
  --results-dir results_hp_200samples \
  --output-dir analysis_plots
```

### Options

- `--results-dir`: Directory containing experiment results (required)
- `--output-dir`: Directory to save analysis plots (default: `analysis_plots_features`)
- `--summary-file`: Specific summary JSON file to analyze (optional, otherwise uses most recent)

### Generated Outputs

The analysis script creates:

1. **Accuracy vs k_features plot**: Shows target and retain accuracy before/after unlearning
2. **Accuracy drop plot**: Shows percentage drop in accuracy vs number of features
3. **Random features analysis** (if applicable): Shows effect of random feature supplementation
4. **Tradeoff plot**: Scatter plot showing target vs retain accuracy drops
5. **Statistics tables**: CSV and Markdown tables with detailed metrics
6. **Console summary**: Statistical summary printed to terminal

### Example

```bash
# Run experiments
python run_feature_variation_experiments.py \
  --feature-counts 1 3 5 7 10 \
  --supplement-with-random 10 \
  --output-dir my_results

# Analyze results
python analyze_feature_results.py \
  --results-dir my_results \
  --output-dir my_analysis
```

### Plot Examples

**Accuracy vs k_features**: See how unlearning effectiveness changes with number of features

**Random Features Analysis**: Understand the contribution of learned vs random features (only generated when `--supplement-with-random` is used)

**Tradeoff Plot**: Visualize the balance between target domain forgetting and retain domain preservation
