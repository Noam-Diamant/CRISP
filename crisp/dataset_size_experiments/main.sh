#!/bin/bash -l
#SBATCH --job-name=important_acts_ftrs_job_%j     
#SBATCH --output=important_acts_ftrs_job_%j.out        
#SBATCH --error=important_acts_ftrs_job_%j.err         
#SBATCH --partition=H200-12h            
#SBATCH --gres=gpu:1                   
#SBATCH --mem=45G  


# activate conda
source /usr/bin/conda.sh

# activate environment
conda activate crisp_env

echo "Starting dataset size experiments on node $SLURM_NODELIST"

#python layer_gradient_analysis.py --model_name HuggingFaceH4/zephyr-7b-beta --layers "7" --save_outputs --num_samples 10000 --max_length 512 --batch_size 8 --aggregate_by hessian_diagonal --normalize_mode all --aggregation_mode both --device cuda:0
bash run_hp_experiments.sh #--model gemma-2-2b --gpu 0 --output-dir results/hp_experiments/gemma-2-2b_vary_both

echo "Job finished."