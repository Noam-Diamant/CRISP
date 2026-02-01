#!/bin/bash -l
#SBATCH --job-name=dataset_size_experiments_vary_feature_extraction_%j     
#SBATCH --output=dataset_size_experiments_vary_feature_extraction_%j.out        
#SBATCH --error=dataset_size_experiments_vary_feature_extraction_%j.err         
#SBATCH --partition=H200-12h            
#SBATCH --gres=gpu:1                   
#SBATCH --mem=80G  


# activate conda
source /usr/bin/conda.sh

# activate environment
conda activate crisp_env

echo "Starting dataset size experiments on node $SLURM_NODELIST"

#python layer_gradient_analysis.py --model_name HuggingFaceH4/zephyr-7b-beta --layers "7" --save_outputs --num_samples 10000 --max_length 512 --batch_size 8 --aggregate_by hessian_diagonal --normalize_mode all --aggregation_mode both --device cuda:0
# python -u "run_unlearning_experiments.py" \
#     --model gemma-2-2b \
#     --target hp \
#     --retain book \
#     --dataset-sizes 10 25 50 100 250 500 1000 1500 2500 \
#     --output-dir experiment_results_hp \
#     --gpu 0 \
#     --max-length 1000 \
#     --vary-dataset feature_extraction \
#     --skip-existing \
#     > logs/experiment_log_gemma_2_2b_vary_feature_extraction.txt 2>&1

#python -u run_feature_variation_experiments.py --feature-counts  1 2 3 4 5 6 7 8 9 10 --n-samples-extraction 10 --n-samples-unlearning 2500 --gpu 0 > logs/experiment_log_gemma_2_2b_vary_featurs_counts.txt 2>&1

python -u run_feature_variation_experiments.py --model llama-3.1-8b --feature-counts  0 1 2 3 4 5 6 --n-samples-extraction 10 --n-samples-unlearning 2500 --supplement-with-random 10 --gpu 0 > logs/experiment_log_llama-3.1-8b_vary_featurs_counts_with_supp_0_and_10_features_alpha_abs.txt


#bash run_hp_experiments.sh --vary-dataset feature_extraction #--model gemma-2-2b --gpu 0 --output-dir results/hp_experiments/gemma-2-2b_vary_both

echo "Job finished."