#!/usr/local_rwth/bin/zsh 

### SBATCH Section

#SBATCH --job-name=VQA-ReGAT
#SBATCH --output=%j-pretrained-ban-spatial-eval-output.log
#SBATCH --error=%j-pretrained-ban-spatial-eval-error.log
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
REGAT_SAVED_MODELS_PATH="${REGAT_DATA_AND_MODELS_PATH}/saved_models/${USERNAME}/train"

# By default, run.sh operates in code directory
cd ${REGAT_LOCAL_CODE_PATH}

### Program Section
python3 eval.py \
    --output_folder ${REGAT_DATA_AND_MODELS_PATH}/pretrained_models/regat_spatial/ban_1_spatial_vqa_1687 \
    --data_folder ${REGAT_DATA_AND_MODELS_PATH}/data
                
