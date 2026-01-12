#!/bin/bash
# Script to run Harry Potter unlearning experiments with varying dataset sizes

# Default parameters
MODEL="gemma-2-2b"
TARGET="hp"
RETAIN="book"
GPU="0"
OUTPUT_DIR="experiment_results_hp"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --gpu)
            GPU="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--model gemma-2-2b|llama-3.1-8b] [--gpu 0] [--output-dir DIR]"
            exit 1
            ;;
    esac
done

echo "======================================"
echo "Harry Potter Unlearning Experiments"
echo "======================================"
echo "Model: $MODEL"
echo "GPU: $GPU"
echo "Output: $OUTPUT_DIR"
echo "======================================"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

python "$SCRIPT_DIR/run_unlearning_experiments.py" \
    --model $MODEL \
    --target $TARGET \
    --retain $RETAIN \
    --dataset-sizes 10 25 50 100 250 500 1000 1500 2500 \
    --output-dir $OUTPUT_DIR \
    --gpu $GPU \
    --max-length 1000 \
    --skip-existing

echo ""
echo "======================================"
echo "Experiments completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "======================================"
