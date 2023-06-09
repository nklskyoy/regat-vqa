#!/usr/local_rwth/bin/zsh 

### SBATCH Section

#SBATCH --job-name=VQA-ReGAT
#SBATCH --output=%j-bs_192_default_lr_3e-4_5e-2_1e-3_opt_Adam_wd_1e-2_ep_20-output.log
#SBATCH --error=%j-bs_192_default_lr_3e-4_5e-2_1e-3_opt_Adam_wd_1e-2_ep_20-error.log
#SBATCH --nodes=1
#SBATCH --gres=gpu:volta:2
#SBATCH --mem-per-gpu=32G
#SBATCH --time=08:00:00

module load CUDA/11.8.0
source /rwthfs/rz/cluster/home/ji111636/anaconda3/bin/activate
conda activate regat_common

export WANDB_API_KEY=523e78ddbd43440344461776e162217f75387b8e

# Extracts the TID of the user
USERNAME=$(whoami)

REGAT_LOCAL_CODE_PATH="/rwthfs/rz/cluster/home/${USERNAME}/vqa_regat"
REGAT_DATA_AND_MODELS_PATH="/rwthfs/rz/cluster/hpcwork/lect0099" 
REGAT_SAVE_MODELS_TRAIN_PATH="${REGAT_DATA_AND_MODELS_PATH}/saved_models/${USERNAME}"

# By default, run.sh operates in code directory
cd ${REGAT_LOCAL_CODE_PATH}

### Program Section
python3 main.py \
    --config config/ban_vqa.json \
    --relation_type spatial \
    --epochs 20 \
    --batch_size 192 \
    --optimizer "Adam" \
    --weight_decay 0.01 \
    --lr_scheduler "default" \
    --base_lr 0.0003 \
    --peak_lr 0.05 \
    --final_lr 0.001 \
    --name "bs_192_default_lr_3e-4_5e-2_1e-3_opt_Adam_wd_1e-2_ep_20" \
    --wandb "run" \
    --job_id ${SLURM_JOB_ID} \
    --output ${REGAT_SAVE_MODELS_TRAIN_PATH}

REGAT_SAVE_EXPERIMENT_TRAIN_PATH=$(find ${REGAT_SAVE_MODELS_TRAIN_PATH} -type d -name "${SLURM_JOB_ID}*" -print -quit)  
mv "./sbatch-scripts/opt-lrs/${USERNAME}/${SLURM_JOB_ID}-"*".log" ${REGAT_SAVE_EXPERIMENT_TRAIN_PATH}

# If running this script by itself (i.e. not using the central run.sh to submit jobs), switch to
# mv "sbatch-scripts/bs-lr/${USERNAME}/${SLURM_JOB_ID}-"*".log" ${REGAT_SAVE_EXPERIMENT_TRAIN_PATH}
