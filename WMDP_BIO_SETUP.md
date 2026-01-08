# WMDP-Bio Unlearning Pipeline Setup Guide

This guide explains how to use the CRISP unlearning pipeline with the WMDP-bio dataset for biosecurity knowledge removal.

## Overview

The WMDP-bio pipeline follows the same structure as the Harry Potter demo but is adapted for:
- **Forget Data**: WMDP-bio corpus (biosecurity knowledge)
- **Retain Data**: Wikipedia (general knowledge)
- **Goal**: Remove harmful biosecurity knowledge while preserving general capabilities

## Quick Start

### 1. Prerequisites

Ensure you have completed the basic CRISP setup:
```bash
conda activate crisp_env
cd CRISP/crisp
```

### 2. Set HuggingFace Token

You'll need access to the model:
```bash
export HF_TOKEN=your_huggingface_token
```

### 3. Option A: Use HuggingFace Fallback (Easiest)

The simplest approach is to let the pipeline load and preprocess WMDP-bio data on-the-fly from HuggingFace:

```bash
jupyter notebook demo_unlearn_wmdp_bio.ipynb
```

**Note**: You'll need access to `cais/wmdp-bio-forget-corpus`. Request access at:
https://huggingface.co/datasets/cais/wmdp-bio-forget-corpus

The data will be automatically downloaded and preprocessed when you run the notebook.

### 3. Option B: Pre-process Data Locally (Recommended for Repeated Use)

For better performance and repeated experiments, pre-process the data:

```bash
# Preprocess WMDP-bio forget corpus
python preprocess_wmdp_bio.py

# Prepare evaluation MCQ files (optional but recommended)
python prepare_wmdp_eval.py
```

This creates:
```
crisp/data/wmdp/bio/
  ├── bio_forget_dataset_cleaned.jsonl
  ├── bio_mcq.json
  ├── college_bio_mcq.json
  └── high_school_bio_mcq.json
```

### 4. Run the Demo Notebook

```bash
jupyter notebook demo_unlearn_wmdp_bio.ipynb
```

## What's Different from Harry Potter Demo?

| Aspect | Harry Potter | WMDP-Bio |
|--------|-------------|----------|
| **Forget Data** | Harry Potter books | WMDP-bio corpus |
| **Retain Data** | General books | Wikipedia |
| **Data Config** | `HPDataConfig` | `WMDPDataConfig` |
| **Load Function** | `load_hp_data()` | `load_wmdp_data()` |
| **Eval Type** | `"hp"` | `"wmdp_bio"` |
| **Text Generation** | `genenrate_hp_eval_text()` | `generate_bio_eval_text()` |
| **Data Type** | `"hp"` | `"wmdp_bio"` |

## Configuration

### Model Selection

The notebook uses **Gemma 2 2B** by default. To use Llama 3.1 8B:

```python
MODEL_CARD = LLAMA_3_1_8B  # Change from GEMMA_2_2B
```

### Hyperparameters (Gemma 2 2B)

```python
{
    "learning_rate": 1e-5,
    "k_features": 10,
    "alpha": 5,
    "sae_layers": [4, 6, 8, 10, 12, 14]
}
```

### Data Configuration

```python
data_config = WMDPDataConfig(
    n_examples=2500,        # Number of examples to use
    forget_type="bio",      # Target: biosecurity knowledge
    retain_type="wiki",     # Retain: general Wikipedia
    max_length=1000,        # Maximum text length
    min_length=1000         # Minimum text length
)
```

## Evaluation

The pipeline evaluates on:

1. **WMDP-Bio MCQs**: Biosecurity-specific questions
2. **College Biology**: General biology knowledge
3. **High School Biology**: Basic biology knowledge  
4. **MMLU**: General knowledge across domains

Expected results:
- ✓ **WMDP-bio accuracy decreases** (biosecurity knowledge removed)
- ✓ **MMLU accuracy maintained** (general knowledge preserved)

## Files Created/Modified

### New Files
- `crisp/demo_unlearn_wmdp_bio.ipynb` - Main demo notebook
- `crisp/preprocess_wmdp_bio.py` - Data preprocessing script
- `crisp/prepare_wmdp_eval.py` - Evaluation data preparation

### Modified Files
- `crisp/globals.py` - Added DATA_PATH
- `crisp/data.py` - Added HuggingFace fallback in load_wmdp_data()

### New Data Directory
```
crisp/data/wmdp/bio/
```

## Troubleshooting

### Issue: "Could not load WMDP-bio data"

**Solution 1**: Request access to the dataset
- Visit: https://huggingface.co/datasets/cais/wmdp-bio-forget-corpus
- Click "Request Access"
- Wait for approval

**Solution 2**: Use local files
- Run `python preprocess_wmdp_bio.py` if you have the raw data

### Issue: "WMDP-bio MCQ evaluation not available"

**Solution**: Run the evaluation preparation script
```bash
python prepare_wmdp_eval.py
```

### Issue: Out of memory

**Solutions**:
1. Reduce batch_size in data loading:
   ```python
   crisp.process_multi_texts_batch(..., batch_size=4)  # or smaller
   ```

2. Reduce n_examples:
   ```python
   data_config = WMDPDataConfig(n_examples=1000)  # instead of 2500
   ```

3. Use fewer SAE layers:
   ```python
   SAE_LAYERS = [8, 10, 12]  # instead of [4, 6, 8, 10, 12, 14]
   ```

### Issue: "HF_TOKEN environment variable not set"

**Solution**: Set your HuggingFace token
```bash
export HF_TOKEN=your_token_here
```

Or in the notebook:
```python
os.environ['HF_TOKEN'] = 'your_token_here'
```

## Advanced Usage

### Using Different Retain Data

The pipeline supports multiple retain data options:

```python
# Option 1: Wikipedia (default)
data = load_wmdp_data(target_type="bio", retain_type="wiki", ...)

# Option 2: Biology-specific Wikipedia
data = load_wmdp_data(target_type="bio", retain_type="wiki-bio", ...)

# Option 3: WMDP-bio retain corpus (if you have it)
data = load_wmdp_data(target_type="bio", retain_type="bio", ...)
```

### Hyperparameter Tuning

Experiment with different values:

```python
uconfig = UnlearnConfig(
    learning_rate=1e-5,     # Try: 5e-6, 1e-5, 2e-5
    k_features=10,          # Try: 5, 10, 15, 20
    alpha=5,                # Try: 1, 5, 10, 20
    beta=0.99,              # Retention loss weight
    gamma=0.01,             # Coherence loss weight
)
```

### Batch Processing

For processing large datasets:

```python
# Process in smaller batches
crisp.process_multi_texts_batch(
    text_target=data['forget'],
    text_benign=data['retain'],
    data_config=data_config,
    batch_size=4  # Adjust based on GPU memory
)
```

## Expected Runtime

On a single GPU (e.g., A100):
- **SAE Download**: 5-10 minutes (first time only)
- **Data Loading**: 2-5 minutes (with HuggingFace fallback) or <1 minute (with local files)
- **Feature Processing**: 20-40 minutes (2500 examples)
- **Unlearning**: 15-30 minutes (1 epoch)
- **Evaluation**: 5-10 minutes
- **Total**: ~1-1.5 hours (first run), ~45 minutes (subsequent runs)

## Citation

If you use this pipeline, please cite the CRISP paper:

```bibtex
@article{ashuach2025crisp,
  title={CRISP: Persistent Concept Unlearning via Sparse Autoencoders},
  author={Ashuach, Tomer and Arad, Dana and Mueller, Aaron and Tutek, Martin and Belinkov, Yonatan},
  journal={arXiv preprint arXiv:2508.13650},
  year={2025}
}
```

## Next Steps

1. **Run the demo**: Start with `demo_unlearn_wmdp_bio.ipynb`
2. **Analyze results**: Check WMDP-bio accuracy drop and MMLU preservation
3. **Inspect features**: Use Neuronpedia to understand which biosecurity concepts were targeted
4. **Experiment**: Try different hyperparameters, models, or retain data
5. **Evaluate comprehensively**: Test on additional biology benchmarks

## Support

For issues or questions:
- Check the main [CRISP README](README.md)
- Review the original [Harry Potter demo](crisp/demo_unlearn_hp.ipynb)
- Consult the [CRISP paper](https://arxiv.org/abs/2508.13650) appendix for implementation details

