#!/usr/local_rwth/bin/zsh 

### SBATCH Section

#SBATCH --job-name=VQA-ReGAT
#SBATCH --output=%j-bs64-lr0.005-eval-output.log
#SBATCH --error=%j-bs64-lr0.005-eval-error.log
#SBATCH --nodes=1
#SBATCH --gres=gpu:volta:2
#SBATCH --mem-per-gpu=32G
#SBATCH --time=00:10:00

module load CUDA/11.8.0
source /rwthfs/rz/cluster/home/gl671475/miniconda3/bin/activate
conda activate regat_common

export WANDB_API_KEY=32abfb91875c02ffd9d1f8d294baad4f5eabbdf7

# Extracts the TID of the user
USERNAME=$(whoami)

REGAT_LOCAL_CODE_PATH="/rwthfs/rz/cluster/home/${USERNAME}/vqa_regat"
REGAT_DATA_AND_MODELS_PATH="/rwthfs/rz/cluster/hpcwork/lect0099" 
REGAT_SAVED_MODELS_PATH="${REGAT_DATA_AND_MODELS_PATH}/saved_models/${USERNAME}"
REGAT_EXPERIMENT_NAME=$(find ${REGAT_SAVED_MODELS_PATH} -name "*bs_64_lr_0.005_ep_20" | sed 's/.*\///')

# By default, run.sh operates in code directory
cd ${REGAT_LOCAL_CODE_PATH}

### Program Section

python3 eval.py \
    --output_folder "${REGAT_SAVED_MODELS_PATH}/bs-lr/${REGAT_EXPERIMENT_NAME}" \
    --data_folder "${REGAT_DATA_AND_MODELS_PATH}/data" \
    --checkpoint 19 \
    --split "val" \
    --save_answers

# REGAT_SAVE_EXPERIMENT_EVAL_PATH=$(find ${REGAT_SAVED_MODELS_PATH} -type d -name "${SLURM_JOB_ID}*" -print -quit)  
mv "sbatch-scripts/${USERNAME}/${SLURM_JOB_ID}-"*"eval"*".log" \
    ${REGAT_SAVED_MODELS_PATH}/bs-lr/${REGAT_EXPERIMENT_NAME}/eval

# If running this script by itself (i.e. not using the central run.sh to submit jobs), switch to
# mv "sbatch-scripts/bs-lr/${USERNAME}/${SLURM_JOB_ID}-"*"eval"*".log" \
#   ${REGAT_SAVED_MODELS_PATH}/bs-lr/<exp-name>/eval