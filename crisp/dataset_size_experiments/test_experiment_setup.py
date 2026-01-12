#!/usr/bin/env python3
"""
Test script to verify the experiment setup is working correctly.

This script performs a minimal test run to ensure all components are functioning
before running the full experiment suite.
"""

import os
import sys
import torch

# Get the parent directory (CRISP/crisp/) and change to it
# This is needed because eval.py and other modules use relative paths
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
os.chdir(parent_dir)

from globals import GEMMA_2_2B, set_seed, SEED
from crisp import CRISP, CRISPConfig
from unlearn import unlearn_lora, UnlearnConfig
from data import load_hp_data, HPDataConfig
from eval import get_mcq_accuracy


def test_basic_setup():
    """Test basic imports and GPU availability."""
    print("="*80)
    print("TEST 1: Basic Setup")
    print("="*80)
    
    # Check CUDA
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.device_count()} GPUs")
        print(f"  Current device: {torch.cuda.current_device()}")
        print(f"  Device name: {torch.cuda.get_device_name(0)}")
    else:
        print("✗ CUDA not available - will use CPU (very slow!)")
    
    # Check HF token
    if 'HF_TOKEN' in os.environ:
        print("✓ HuggingFace token found")
    else:
        print("✗ HF_TOKEN not set in environment")
        return False
    
    return True


def test_data_loading():
    """Test data loading functionality."""
    print("\n" + "="*80)
    print("TEST 2: Data Loading")
    print("="*80)
    
    try:
        print("Loading HP data with n_examples=10...")
        forget_data, retain_data = load_hp_data(benign='book', n_examples=10, max_len=1000)
        print(f"✓ Loaded {len(forget_data)} forget examples")
        print(f"✓ Loaded {len(retain_data)} retain examples")
        print(f"  Sample forget text (first 100 chars): {forget_data[0][:100]}")
        return True
    except Exception as e:
        print(f"✗ Data loading failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_model_initialization():
    """Test model initialization."""
    print("\n" + "="*80)
    print("TEST 3: Model Initialization")
    print("="*80)
    
    try:
        print("Initializing CRISP with Gemma-2-2B...")
        crisp_config = CRISPConfig(
            layers=[4, 6],  # Use only 2 layers for quick test
            model_name=GEMMA_2_2B,
            bf16=True
        )
        
        crisp = CRISP(config=crisp_config)
        print("✓ CRISP model initialized")
        print(f"  Model: {crisp.config.model_name}")
        print(f"  Layers: {crisp.config.layers}")
        print(f"  Device: {crisp.model.device}")
        
        # Clean up
        del crisp
        torch.cuda.empty_cache()
        return True
    except Exception as e:
        print(f"✗ Model initialization failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_cache_functionality():
    """Test cache filename generation."""
    print("\n" + "="*80)
    print("TEST 4: Cache Functionality")
    print("="*80)
    
    try:
        from utils import get_cache_filename
        
        # Test HPDataConfig cache naming
        data_config = HPDataConfig(
            retain_type='book',
            n_examples=100,
            max_length=1000,
            min_length=1000
        )
        
        filename = get_cache_filename(layer=4, data_config=data_config)
        print(f"✓ Cache filename generated: {filename}")
        
        # Verify n_examples is in the filename
        if "n_examples_100" in filename:
            print("✓ n_examples correctly included in cache filename")
        else:
            print("✗ n_examples NOT found in cache filename")
            return False
        
        return True
    except Exception as e:
        print(f"✗ Cache functionality test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_minimal_experiment():
    """Test a minimal experiment with very small data."""
    print("\n" + "="*80)
    print("TEST 5: Minimal Experiment (This will take a few minutes)")
    print("="*80)
    
    try:
        set_seed(SEED)
        
        # Initialize model with minimal layers
        print("Initializing model...")
        crisp_config = CRISPConfig(
            layers=[4, 6],  # Just 2 layers for speed
            model_name=GEMMA_2_2B,
            bf16=True
        )
        crisp = CRISP(config=crisp_config)
        
        # Load minimal data
        print("Loading data (n=5)...")
        forget_data, retain_data = load_hp_data(benign='book', n_examples=5, max_len=500)
        
        # Create configs
        data_config = HPDataConfig(
            retain_type='book',
            n_examples=5,
            max_length=500,
            min_length=500
        )
        
        unlearn_config = UnlearnConfig(
            data_type='hp',
            learning_rate=1e-5,
            k_features=5,  # Fewer features
            alpha=5,
            num_epochs=1,
            batch_size=1,
            save_model=False,
            verbose=None  # Reduce verbosity
        )
        
        # Test evaluation (just one dataset)
        print("Testing evaluation...")
        hp_acc = get_mcq_accuracy(crisp, type="hp", verbose=False)
        print(f"✓ HP accuracy: {hp_acc:.4f}")
        
        # Test unlearning (this is the main test)
        print("Testing unlearning process...")
        unlearn_lora(
            crisp=crisp,
            text_target=forget_data,
            text_benign=retain_data,
            config=unlearn_config,
            data_config=data_config
        )
        print("✓ Unlearning completed")
        
        # Test post-unlearning evaluation
        print("Testing post-unlearning evaluation...")
        hp_acc_after = get_mcq_accuracy(crisp, type="hp", verbose=False)
        print(f"✓ HP accuracy after: {hp_acc_after:.4f}")
        
        # Clean up
        del crisp
        torch.cuda.empty_cache()
        
        print("✓ Minimal experiment successful!")
        return True
        
    except Exception as e:
        print(f"✗ Minimal experiment failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "#"*80)
    print("# EXPERIMENT SETUP TEST")
    print("#"*80 + "\n")
    
    results = {
        "Basic Setup": test_basic_setup(),
        "Data Loading": test_data_loading(),
        "Model Initialization": test_model_initialization(),
        "Cache Functionality": test_cache_functionality(),
    }
    
    # Only run expensive test if others pass
    if all(results.values()):
        results["Minimal Experiment"] = test_minimal_experiment()
    else:
        print("\n⚠ Skipping minimal experiment test due to previous failures")
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:<30} {status}")
    
    print("="*80)
    
    if all(results.values()):
        print("\n🎉 All tests passed! You're ready to run experiments.")
        print("\nTo run experiments:")
        print("  ./run_hp_experiments.sh")
        print("  or")
        print("  python run_unlearning_experiments.py")
        return 0
    else:
        print("\n❌ Some tests failed. Please fix the issues before running experiments.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
