# Experiment Scripts Summary

This document provides a complete overview of the unlearning experiment scripts created for varying dataset sizes.

## Files Created

### Main Scripts

1. **`run_unlearning_experiments.py`** - Main experiment runner
   - Runs unlearning experiments with varying dataset sizes
   - Supports both Gemma-2-2B and Llama-3.1-8B models
   - Handles HP and Bio target domains
   - Saves metrics in JSON format
   - Does NOT save unlearned models (saves disk space)

2. **`analyze_results.py`** - Results analysis and visualization
   - Loads experiment results from JSON
   - Generates plots (accuracy, drops, tradeoffs)
   - Creates statistics tables (CSV and Markdown)
   - Prints summary statistics

3. **`test_experiment_setup.py`** - Setup verification
   - Tests basic setup (CUDA, HF token)
   - Tests data loading
   - Tests model initialization
   - Tests cache functionality
   - Runs minimal experiment to verify full pipeline

### Convenience Scripts

4. **`run_hp_experiments.sh`** - Quick HP experiments
   - Wrapper for HP experiments with default settings
   - Supports command-line model and GPU selection

5. **`run_bio_experiments.sh`** - Quick Bio experiments
   - Wrapper for Bio experiments with default settings
   - Supports command-line model and GPU selection

### Documentation

6. **`EXPERIMENT_README.md`** - Comprehensive documentation
   - Detailed usage instructions
   - All command-line arguments explained
   - Output format documentation
   - Performance considerations
   - Troubleshooting guide

7. **`QUICKSTART.md`** - Quick start guide
   - Step-by-step getting started
   - Common use cases
   - Expected outputs
   - Tips and troubleshooting

8. **`EXPERIMENT_SCRIPTS_SUMMARY.md`** - This file
   - Overview of all scripts
   - Quick reference
   - Workflow diagram

## Quick Reference

### Running Experiments

```bash
# Test setup first
./test_experiment_setup.py

# Run HP experiments (Gemma-2-2B)
./run_hp_experiments.sh

# Run HP experiments (Llama-3.1-8B)
./run_hp_experiments.sh --model llama-3.1-8b --gpu 1

# Run Bio experiments
./run_bio_experiments.sh

# Custom configuration
python run_unlearning_experiments.py \
    --model gemma-2-2b \
    --target hp \
    --retain book \
    --dataset-sizes 10 25 50 100 250 500 1000 1500 2500 \
    --output-dir results_hp \
    --gpu 0
```

### Analyzing Results

```bash
# Analyze HP results
python analyze_results.py \
    --results-dir experiment_results_hp \
    --output-dir analysis_hp

# Analyze Bio results
python analyze_results.py \
    --results-dir experiment_results_bio \
    --output-dir analysis_bio
```

## Workflow Diagram

```
1. Setup & Verification
   └─> test_experiment_setup.py
       ├─> Tests CUDA, HF token
       ├─> Tests data loading
       ├─> Tests model initialization
       └─> Runs minimal experiment

2. Run Experiments
   └─> run_unlearning_experiments.py (or convenience scripts)
       ├─> For each dataset size (10, 25, 50, ..., 2500):
       │   ├─> Load data (forget + retain)
       │   ├─> Initialize model
       │   ├─> Evaluate before unlearning
       │   ├─> Perform unlearning
       │   ├─> Evaluate after unlearning
       │   ├─> Save results (JSON)
       │   └─> Save processed features (with n_examples suffix)
       └─> Save summary results (JSON)

3. Analyze Results
   └─> analyze_results.py
       ├─> Load experiment results
       ├─> Generate plots
       │   ├─> Accuracy vs dataset size
       │   ├─> Accuracy drop vs dataset size
       │   └─> Tradeoff plot
       ├─> Create statistics tables (CSV, Markdown)
       └─> Print summary statistics
```

## Key Features

### 1. Varying Dataset Sizes
- Default: 10, 25, 50, 100, 250, 500, 1000, 1500, 2500 examples
- Customizable via `--dataset-sizes` argument
- Both forget and retain sets use the same number of examples

### 2. Cache Management
- Processed features are cached with n_examples suffix
- Format: `layer_{L}_type_{T}_forget_{F}_retain_{R}_n_examples_{N}_max_len_{M}.pkl`
- Example: `layer_10_type_hpdata_forget_HarryPotter_books_1to7_retain_Tiny-Open-Domain-Books_n_examples_250_max_len_1000.pkl`
- Prevents conflicts between different dataset sizes

### 3. Metrics Tracking
- **Before unlearning**: Target accuracy, MMLU accuracy
- **After unlearning**: Target accuracy, MMLU accuracy
- **Derived metrics**: Absolute and percentage drops for both target and retain

### 4. Model Support
- **Gemma-2-2B** (default)
  - Layers: 4, 6, 8, 10, 12, 14
  - SAE: JumpReLUSAE
  - Learning rate: 1e-5, alpha: 5
- **Llama-3.1-8B**
  - Layers: 4, 6, 8, ..., 28
  - SAE: TopkSae
  - Learning rate: 2e-5, alpha: 30

### 5. Target Domains
- **HP (Harry Potter)**
  - Forget: Harry Potter books
  - Retain: Tiny-Open-Domain-Books or WikiText
  - Eval: HP MCQ accuracy, MMLU accuracy
- **Bio (WMDP Bio)**
  - Forget: WMDP Bio corpus
  - Retain: WikiText or biology wiki
  - Eval: WMDP Bio MCQ accuracy, MMLU accuracy

### 6. Disk Space Management
- ✓ Processed features are saved (needed for analysis)
- ✗ Unlearned models are NOT saved (saves 5-15 GB per experiment)
- Estimated disk usage: 1-10 GB for full experiment series

## Output Structure

```
CRISP/crisp/dataset_size_experiments/
├── run_unlearning_experiments.py
├── analyze_results.py
├── test_experiment_setup.py
├── run_hp_experiments.sh
├── run_bio_experiments.sh
├── EXPERIMENT_README.md
├── QUICKSTART.md
├── EXPERIMENT_SCRIPTS_SUMMARY.md
├── experiment_results_hp/
│   ├── experiment_n10_hp_book_gemma.json       # Individual experiment results
│   ├── experiment_n25_hp_book_gemma.json
│   ├── ...
│   ├── experiment_n2500_hp_book_gemma.json
│   └── summary_hp_book_gemma-2-2b_20260112_143052.json  # Summary of all experiments
└── analysis_hp/
    ├── accuracy_vs_dataset_size_hp.png         # Plots
    ├── accuracy_drop_vs_dataset_size_hp.png
    ├── tradeoff_plot_hp.png
    ├── statistics_table_hp.csv                 # Tables
    └── statistics_table_hp.md

CRISP/crisp/crisp_cache/gemma_2_2b_processed_features/
├── layer_4_type_hpdata_..._n_examples_10_max_len_1000.pkl
├── layer_4_type_hpdata_..._n_examples_25_max_len_1000.pkl
└── ...
```

## JSON Output Format

### Individual Experiment Result

```json
{
  "n_examples": 250,
  "target": "hp",
  "retain": "book",
  "model": "google/gemma-2-2b",
  "max_length": 1000,
  "timestamp": "2026-01-12T14:30:52.123456",
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
  "retain_accuracy_drop_percent": 3.46,
  "unlearn_config": { ... },
  "data_config": { ... }
}
```

### Summary Results

```json
{
  "experiment_info": {
    "target": "hp",
    "retain": "book",
    "model": "gemma-2-2b",
    "max_length": 1000,
    "dataset_sizes": [10, 25, 50, 100, 250, 500, 1000, 1500, 2500],
    "timestamp": "20260112_143052"
  },
  "results": [
    { /* experiment 1 */ },
    { /* experiment 2 */ },
    ...
  ]
}
```

## Common Use Cases

### 1. Full HP Experiment Suite (Gemma)
```bash
./run_hp_experiments.sh
```

### 2. Full Bio Experiment Suite (Llama)
```bash
./run_bio_experiments.sh --model llama-3.1-8b
```

### 3. Quick Test with Small Dataset
```bash
python run_unlearning_experiments.py --dataset-sizes 10 50 100
```

### 4. Resume Interrupted Experiments
```bash
python run_unlearning_experiments.py --skip-existing
```

### 5. Parallel Execution on Multiple GPUs
```bash
# GPU 0: Small datasets
python run_unlearning_experiments.py --dataset-sizes 10 25 50 100 --gpu 0 &

# GPU 1: Medium datasets
python run_unlearning_experiments.py --dataset-sizes 250 500 1000 --gpu 1 &

# GPU 2: Large datasets
python run_unlearning_experiments.py --dataset-sizes 1500 2500 --gpu 2 &
```

## Troubleshooting Quick Reference

| Issue | Solution |
|-------|----------|
| Out of memory | Use fewer dataset sizes: `--dataset-sizes 10 50 100` |
| HF auth error | Set token: `export HF_TOKEN=<token>` |
| Slow execution | Run in parallel on multiple GPUs |
| Script crashes | Resume with: `--skip-existing` |
| Missing dependencies | Check imports in test script |

## Performance Estimates

| Configuration | Time per Experiment | Full Suite (9 sizes) |
|---------------|---------------------|----------------------|
| Gemma-2-2B | 10-20 minutes | 2-3 hours |
| Llama-3.1-8B | 20-40 minutes | 3-6 hours |

*Note: Times vary based on GPU, dataset size, and system load*

## Next Steps

1. **Verify Setup**: Run `./test_experiment_setup.py`
2. **Start Small**: Test with `--dataset-sizes 10 50 100`
3. **Run Full Suite**: Use convenience scripts or custom configuration
4. **Analyze Results**: Use `analyze_results.py` to generate plots
5. **Custom Analysis**: Load JSON files for deeper analysis

## Support

- **Detailed docs**: `EXPERIMENT_README.md`
- **Quick start**: `QUICKSTART.md`
- **Help options**:
  - `python run_unlearning_experiments.py --help`
  - `python analyze_results.py --help`
  - `python test_experiment_setup.py --help`
