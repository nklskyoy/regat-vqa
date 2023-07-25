#!/usr/local_rwth/bin/zsh 

### SBATCH Section
#SBATCH --job-name=VQA-ReGAT
#SBATCH --output=%j-output.log
#SBATCH --error=%j-error.log
#SBATCH --nodes=1
#SBATCH --gres=gpu:volta:2
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --account=lect0099

export WANDB_API_KEY=4eeb4bff5fc5678d495fd6ac0a3c0a8bfab3ac3c

module load CUDA/11.8.0
source /home/hf201627/anaconda3/bin/activate regat

# Extracts the TID of the user
USERNAME=$(whoami)

REGAT_LOCAL_CODE_PATH="/rwthfs/rz/cluster/home/${USERNAME}/vqa_regat"
REGAT_DATA_AND_MODELS_PATH="/rwthfs/rz/cluster/hpcwork/lect0099" 

REGAT_SAVE_MODELS_TRAIN_PATH="${REGAT_DATA_AND_MODELS_PATH}/saved_models/${USERNAME}/train"
REGAT_SAVE_MODELS_EVAL_PATH="${REGAT_DATA_AND_MODELS_PATH}/saved_models/${USERNAME}/eval"
echo "Save models to: ${REGAT_SAVE_MODELS_TRAIN_PATH}"
echo $1
# By default, run.sh operates in code directory
# cd ${REGAT_LOCAL_CODE_PATH}
#${SLURM_JOB_ID}
### Program Section
python3 main.py \
    --config config/ban_vqa.json  \
    --epochs 30 \
    --output "${REGAT_SAVE_MODELS_TRAIN_PATH}" \ 
    --init_lr $1 \
    --peak_lr $2 \
    --final_lr 0.1 \
    --begin_constant 15 \
    --begin_decay 100 \
    --lr_scheduler "custom" \
    --name "fat_concat?opt=adamax;init_lr=$1;peak_lr=$2;final_lr=$3;begin_constant=$4;begin_decay=$5" \ 
    --wandb 



