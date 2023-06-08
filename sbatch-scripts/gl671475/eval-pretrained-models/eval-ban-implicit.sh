#!/usr/local_rwth/bin/zsh 

### SBATCH Section

#SBATCH --job-name=VQA-ReGAT
#SBATCH --output=%j-eval-ban-implicit.log
#SBATCH --error=%j-eval-ban-implicit.log
#SBATCH --nodes=1
#SBATCH --gres=gpu:volta:2
#SBATCH --mem-per-gpu=32G
#SBATCH --time=01:00:00
#SBATCH --account=lect0099

module load CUDA/11.8.0
source /rwthfs/rz/cluster/home/gl671475/miniconda3/bin/activate
conda activate regat_common

# Extracts the TID of the user
USERNAME=$(whoami)

REGAT_LOCAL_CODE_PATH="/rwthfs/rz/cluster/home/${USERNAME}/vqa_regat"
REGAT_DATA_AND_MODELS_PATH="/rwthfs/rz/cluster/hpcwork/lect0099" 
REGAT_SAVE_MODELS_TRAIN_PATH="${REGAT_DATA_AND_MODELS_PATH}/saved_models/${USERNAME}"

# By default, run.sh operates in code directory
cd ${REGAT_LOCAL_CODE_PATH}

### Program Section
python3 eval.py \
    --output_folder "${REGAT_DATA_AND_MODELS_PATH}/pretrained_models/regat_implicit/ban_1_implicit_vqa_196" \
    --data_folder "${REGAT_DATA_AND_MODELS_PATH}/data" \
    --save_logits \
    --save_answers