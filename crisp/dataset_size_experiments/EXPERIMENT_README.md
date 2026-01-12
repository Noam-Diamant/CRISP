# Unlearning Experiments Script

This document describes how to use the `run_unlearning_experiments.py` script to run systematic unlearning experiments with varying dataset sizes.

## Overview

The script performs unlearning experiments by:
1. Varying the number of examples used for both forget and retain sets
2. Evaluating metrics before and after unlearning for each dataset size
3. Saving processed features with suffixes indicating the number of examples
4. Storing all metrics in JSON format
5. NOT saving the unlearned models to conserve disk space

## Usage

### Basic Usage

```bash
# Default configuration (Gemma-2-2B, HP target, book retain set)
python run_unlearning_experiments.py

# Specify model
python run_unlearning_experiments.py --model llama-3.1-8b

# Specify target and retain sets
python run_unlearning_experiments.py --target bio --retain wiki

# Custom dataset sizes
python run_unlearning_experiments.py --dataset-sizes 10 50 100 500 1000
```

### Command Line Arguments

| Argument | Type | Default | Choices | Description |
|----------|------|---------|---------|-------------|
| `--model` | str | `gemma-2-2b` | `gemma-2-2b`, `llama-3.1-8b` | Model to use for experiments |
| `--target` | str | `hp` | `hp`, `bio` | Target domain to unlearn |
| `--retain` | str | `book` | `book`, `wiki` | Retain set (book for HP, wiki for bio) |
| `--dataset-sizes` | int+ | `[10, 25, 50, 100, 250, 500, 1000, 1500, 2500]` | Any positive integers | List of dataset sizes to experiment with |
| `--output-dir` | str | `experiment_results` | Any path | Directory to save experiment results |
| `--gpu` | str | `0` | Any GPU ID | GPU device ID to use |
| `--max-length` | int | `1000` | Any positive integer | Maximum length for text processing |
| `--skip-existing` | flag | False | N/A | Skip experiments with existing results |

## Examples

### Harry Potter Experiments with Gemma-2-2B

```bash
# Default HP experiment with books as retain set
python run_unlearning_experiments.py \
    --model gemma-2-2b \
    --target hp \
    --retain book \
    --dataset-sizes 10 25 50 100 250 500 1000 1500 2500 \
    --output-dir results_hp_gemma \
    --gpu 0
```

### WMDP Bio Experiments with Llama-3.1-8B

```bash
# Bio experiment with wiki as retain set
python run_unlearning_experiments.py \
    --model llama-3.1-8b \
    --target bio \
    --retain wiki \
    --dataset-sizes 10 25 50 100 250 500 1000 1500 2500 \
    --output-dir results_bio_llama \
    --gpu 1
```

### Custom Dataset Sizes

```bash
# Run experiments with specific dataset sizes
python run_unlearning_experiments.py \
    --dataset-sizes 50 100 200 400 800 1600 \
    --output-dir results_custom
```

### Resume Interrupted Experiments

```bash
# Skip experiments that have already completed
python run_unlearning_experiments.py \
    --skip-existing \
    --output-dir experiment_results
```

## Output Files

The script generates the following output files in the specified output directory:

### Individual Experiment Results

For each dataset size, a file is created:
```
experiment_n{N}_{target}_{retain}_{model}.json
```

Example: `experiment_n250_hp_book_gemma.json`

This file contains:
- Configuration details (n_examples, target, retain, model, etc.)
- Metrics before unlearning (target accuracy, MMLU accuracy)
- Metrics after unlearning
- Calculated improvements (accuracy drops in absolute and percentage)
- Timestamp and complete configuration

### Summary Results

A summary file aggregating all experiments:
```
summary_{target}_{retain}_{model}_{timestamp}.json
```

Example: `summary_hp_book_gemma-2-2b_20260112_143052.json`

This file contains:
- Experiment metadata
- Array of all individual experiment results
- Easy-to-parse format for analysis

### Cached Features

Processed features are saved with suffixes indicating the number of examples:
```
layer_{layer}_type_{config_type}_forget_{forget}_retain_{retain}_n_examples_{N}_max_len_{max_len}.pkl
```

Example: `layer_10_type_hpdata_forget_HarryPotter_books_1to7_retain_Tiny-Open-Domain-Books_n_examples_250_max_len_1000.pkl`

## Metrics Tracked

For each experiment, the following metrics are tracked:

### Harry Potter (HP) Target
- **HP Accuracy**: Multiple-choice accuracy on Harry Potter questions
- **MMLU Accuracy**: General knowledge accuracy (to verify retention)

### WMDP Bio Target
- **WMDP Bio Accuracy**: Multiple-choice accuracy on biology/biosecurity questions
- **MMLU Accuracy**: General knowledge accuracy (to verify retention)

### Derived Metrics
- **Target Accuracy Drop**: Absolute decrease in target domain accuracy
- **Target Accuracy Drop %**: Percentage decrease in target domain accuracy
- **Retain Accuracy Drop**: Absolute decrease in retained knowledge
- **Retain Accuracy Drop %**: Percentage decrease in retained knowledge

## Output Example

### Console Output

```
================================================================================
EXPERIMENT SUMMARY
================================================================================
N Examples   Target Acc Before  Target Acc After   Drop %     Retain Acc Before Retain Acc After  Drop %    
------------------------------------------------------------------------------------------------------------------------
10           0.6263             0.4521             27.82      0.4630             0.4601             0.63      
25           0.6263             0.3892             37.86      0.4630             0.4585             0.97      
50           0.6263             0.3215             48.67      0.4630             0.4550             1.73      
100          0.6263             0.2789             55.47      0.4630             0.4520             2.38      
250          0.6263             0.2389             61.87      0.4630             0.4470             3.46      
================================================================================
```

### JSON Output Structure

```json
{
  "n_examples": 250,
  "target": "hp",
  "retain": "book",
  "model": "google/gemma-2-2b",
  "metrics_before": {
    "hp_accuracy": 0.6263,
    "mmlu_accuracy": 0.4630
  },
  "metrics_after": {
    "hp_accuracy": 0.2389,
    "mmlu_accuracy": 0.4470
  },
  "target_accuracy_drop": 0.3874,
  "target_accuracy_drop_percent": 61.87,
  "retain_accuracy_drop": 0.0160,
  "retain_accuracy_drop_percent": 3.46
}
```

## Performance Considerations

### Memory Management
- Models are deleted after each experiment to free GPU memory
- `torch.cuda.empty_cache()` and `gc.collect()` are called between experiments
- Unlearned models are NOT saved to disk

### Disk Space
- Processed features are saved for each dataset size
- Each feature cache is approximately 10-100 MB depending on the dataset
- Plan for ~1-10 GB of disk space for the full experiment series
- Models are NOT saved (saving ~5-15 GB per experiment)

### Time Estimates
- Gemma-2-2B: ~10-20 minutes per experiment
- Llama-3.1-8B: ~20-40 minutes per experiment
- Full 9-size experiment series: 2-6 hours depending on model

## Troubleshooting

### Out of Memory Errors
```bash
# Use smaller batch sizes or fewer dataset sizes
python run_unlearning_experiments.py --dataset-sizes 10 50 100 500
```

### Resume After Crash
```bash
# Use --skip-existing to resume
python run_unlearning_experiments.py --skip-existing
```

### Check GPU Usage
```bash
# Monitor GPU during execution
watch -n 1 nvidia-smi
```

## Analysis

After running experiments, you can analyze results using:

```python
import json
import pandas as pd

# Load summary results
with open('experiment_results/summary_hp_book_gemma-2-2b_20260112_143052.json', 'r') as f:
    data = json.load(f)

# Convert to DataFrame
results = pd.DataFrame(data['results'])

# Plot accuracy vs dataset size
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(results['n_examples'], 
         [r['hp_accuracy'] for r in results['metrics_before']], 
         label='Before Unlearning', marker='o')
plt.plot(results['n_examples'], 
         [r['hp_accuracy'] for r in results['metrics_after']], 
         label='After Unlearning', marker='s')
plt.xlabel('Number of Examples')
plt.ylabel('HP Accuracy')
plt.title('Unlearning Performance vs Dataset Size')
plt.legend()
plt.grid(True)
plt.savefig('unlearning_performance.png')
```

## Notes

1. The script automatically validates retain set choices for the target domain
2. Processed features are cached with proper suffixes to avoid conflicts
3. Random seed is set for reproducibility (SEED=0)
4. The script continues with remaining experiments even if one fails
5. Both per-experiment and summary JSON files are created for easy analysis
