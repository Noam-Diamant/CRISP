# Quick Start Guide: Unlearning Experiments

This guide will help you quickly run unlearning experiments with varying dataset sizes.

## Prerequisites

Ensure you have:
1. Python environment with all dependencies installed
2. Access to GPU(s)
3. HuggingFace token set in environment: `export HF_TOKEN=<your_token>`

## Quick Start

### 1. Run Harry Potter Experiments (Default)

```bash
cd /dsi/fetaya-lab/noam_diamant/projects/Unlearning_with_SAE/CRISP/crisp/dataset_size_experiments

# Using the convenience script
./run_hp_experiments.sh

# Or directly with Python
python run_unlearning_experiments.py
```

This will:
- Use Gemma-2-2B model (default)
- Target: Harry Potter knowledge
- Retain: Open domain books
- Dataset sizes: 10, 25, 50, 100, 250, 500, 1000, 1500, 2500
- Save results to `experiment_results_hp/`

### 2. Run Bio Experiments

```bash
# Using the convenience script
./run_bio_experiments.sh

# Or directly with Python
python run_unlearning_experiments.py --target bio --retain wiki
```

This will:
- Use Gemma-2-2B model (default)
- Target: WMDP Bio knowledge
- Retain: WikiText
- Dataset sizes: 10, 25, 50, 100, 250, 500, 1000, 1500, 2500
- Save results to `experiment_results_bio/`

### 3. Use Llama Model

```bash
# HP with Llama
./run_hp_experiments.sh --model llama-3.1-8b --gpu 1

# Bio with Llama
./run_bio_experiments.sh --model llama-3.1-8b --gpu 1
```

### 4. Custom Configuration

```bash
python run_unlearning_experiments.py \
    --model gemma-2-2b \
    --target hp \
    --retain book \
    --dataset-sizes 50 100 500 1000 \
    --output-dir my_custom_results \
    --gpu 2
```

### 5. Resume Interrupted Experiments

```bash
python run_unlearning_experiments.py --skip-existing
```

### 6. Analyze Results

```bash
python analyze_results.py --results-dir experiment_results_hp --output-dir analysis_hp
```

This will generate:
- Accuracy vs dataset size plots
- Accuracy drop plots
- Tradeoff analysis plots
- Statistics tables (CSV and Markdown)

## File Structure

After running experiments, you'll have:

```
dataset_size_experiments/
├── run_unlearning_experiments.py
├── analyze_results.py
├── test_experiment_setup.py
├── run_hp_experiments.sh
├── run_bio_experiments.sh
├── EXPERIMENT_README.md
├── QUICKSTART.md
├── EXPERIMENT_SCRIPTS_SUMMARY.md
├── experiment_results_hp/
│   ├── experiment_n10_hp_book_gemma.json
│   ├── experiment_n25_hp_book_gemma.json
│   ├── experiment_n50_hp_book_gemma.json
│   ├── ...
│   └── summary_hp_book_gemma-2-2b_20260112_143052.json
└── analysis_hp/
    ├── accuracy_vs_dataset_size_hp.png
    ├── accuracy_drop_vs_dataset_size_hp.png
    ├── tradeoff_plot_hp.png
    ├── statistics_table_hp.csv
    └── statistics_table_hp.md

crisp_cache/gemma_2_2b_processed_features/
├── layer_4_type_hpdata_forget_HarryPotter_books_1to7_retain_Tiny-Open-Domain-Books_n_examples_10_max_len_1000.pkl
├── layer_4_type_hpdata_forget_HarryPotter_books_1to7_retain_Tiny-Open-Domain-Books_n_examples_25_max_len_1000.pkl
└── ...
```

## Expected Output

### Console Output During Experiments

```
================================================================================
Running experiment with 250 examples
================================================================================

Initializing model: google/gemma-2-2b
Operating on layers: [4, 6, 8, 10, 12, 14]
Loading HP data with 250 examples, retain type: book

--- Evaluating Original Model ---
Evaluating Harry Potter accuracy...
Evaluating batches: 100%|██████████| 155/155 [00:09<00:00, 17.19it/s]
Overall accuracy for hp_mcq: 0.626

Evaluating MMLU accuracy...
Evaluating batches: 100%|██████████| 1756/1756 [02:24<00:00, 12.19it/s]
Overall accuracy for mmlu: 0.463

--- Performing Unlearning ---
[Unlearning process details...]

--- Evaluating After Unlearning ---
Evaluating Harry Potter accuracy...
Overall accuracy for hp_mcq: 0.289
HP Accuracy: 28.89% (drop of 53.85%)

Evaluating MMLU accuracy...
Overall accuracy for mmlu: 0.447
MMLU Accuracy: 44.70% (drop of 3.46%)

Saved experiment results to: experiment_results_hp/experiment_n250_hp_book_gemma.json
```

### Summary Table

```
================================================================================
EXPERIMENT SUMMARY
================================================================================
N Examples   Target Acc Before  Target Acc After   Drop %     Retain Acc Before Retain Acc After  Drop %    
------------------------------------------------------------------------------------------------------------------------
10           0.6263             0.5521             11.85      0.4630             0.4615             0.32      
25           0.6263             0.4892             21.89      0.4630             0.4601             0.63      
50           0.6263             0.4215             32.71      0.4630             0.4585             0.97      
100          0.6263             0.3489             44.31      0.4630             0.4560             1.51      
250          0.6263             0.2889             53.85      0.4630             0.4470             3.46      
500          0.6263             0.2589             58.65      0.4630             0.4420             4.54      
1000         0.6263             0.2389             61.87      0.4630             0.4370             5.62      
1500         0.6263             0.2289             63.47      0.4630             0.4330             6.48      
2500         0.6263             0.2189             65.06      0.4630             0.4280             7.56      
================================================================================
```

## Troubleshooting

### Issue: Out of Memory

**Solution**: Use fewer dataset sizes or smaller batches
```bash
python run_unlearning_experiments.py --dataset-sizes 10 50 100 500
```

### Issue: HuggingFace Authentication Error

**Solution**: Set your token
```bash
export HF_TOKEN=<your_huggingface_token>
```

### Issue: Experiments Taking Too Long

**Solution**: Run in parallel on multiple GPUs
```bash
# Terminal 1 (GPU 0)
python run_unlearning_experiments.py --dataset-sizes 10 25 50 100 --gpu 0

# Terminal 2 (GPU 1)
python run_unlearning_experiments.py --dataset-sizes 250 500 1000 --gpu 1

# Terminal 3 (GPU 2)
python run_unlearning_experiments.py --dataset-sizes 1500 2500 --gpu 2
```

### Issue: Script Crashes Mid-Experiment

**Solution**: Resume with skip-existing flag
```bash
python run_unlearning_experiments.py --skip-existing
```

## Tips for Efficient Experimentation

1. **Start Small**: Test with `--dataset-sizes 10 50 100` first to verify everything works

2. **Use Skip-Existing**: Always use `--skip-existing` when resuming to avoid re-running completed experiments

3. **Monitor Resources**: Watch GPU memory and disk space
   ```bash
   watch -n 1 nvidia-smi
   df -h
   ```

4. **Parallel Execution**: Run different configurations in parallel on different GPUs

5. **Early Analysis**: Analyze partial results before all experiments complete
   ```bash
   python analyze_results.py --results-dir experiment_results_hp
   ```

## Next Steps

1. **Visualize Results**: Use the analysis script to generate plots
2. **Compare Configurations**: Run experiments with different models and compare
3. **Deep Dive**: Examine individual experiment JSON files for detailed insights
4. **Custom Analysis**: Load JSON results in Python/R for custom analysis

## Support

For detailed documentation, see:
- `EXPERIMENT_README.md` - Full documentation
- `analyze_results.py --help` - Analysis options
- `run_unlearning_experiments.py --help` - All command-line options
