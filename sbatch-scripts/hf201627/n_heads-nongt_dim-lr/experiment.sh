#!/usr/local_rwth/bin/zsh 

### SBATCH Section

#SBATCH --job-name=VQA-ReGAT
#SBATCH --output=%jbs_192;opt_Adam;nongt_dim=$dim;n_heads=$nh-output.log
#SBATCH --error=%jbs_192;opt_Adam;nongt_dim=$dim;n_heads=$nh-error.log
#SBATCH --nodes=1
#SBATCH --gres=gpu:volta:2
#SBATCH --mem-per-gpu=32G
#SBATCH --time=08:00:00
#SBATCH --account=lect0099

module load CUDA/11.8.0
source /home/hf201627/anaconda3/bin/activate regat

export WANDB_API_KEY=4eeb4bff5fc5678d495fd6ac0a3c0a8bfab3ac3c

# Extracts the TID of the user
USERNAME=$(whoami)

REGAT_LOCAL_CODE_PATH="/rwthfs/rz/cluster/home/${USERNAME}/dev/lab2/vqa_regat"
REGAT_DATA_AND_MODELS_PATH="/rwthfs/rz/cluster/hpcwork/lect0099" 
REGAT_SAVE_MODELS_TRAIN_PATH="${REGAT_DATA_AND_MODELS_PATH}/saved_models/${USERNAME}"

# By default, run.sh operates in code directory
cd ${REGAT_LOCAL_CODE_PATH}

### Program Section

# Debug log
echo "dim:$1"
echo "nh:$2"

# freaky rUn!
python3 main.py \
    --config config/ban_vqa.json \
    --relation_type spatial \
    --epochs 20 \
    --batch_size 192 \
    --num_heads $2 \
    --nongt_dim $1 \
    --optimizer "Adam" \
    --lr_scheduler "custom" \
    --base_lr 0.001 \
    --name "bs_192;opt_Adam;nongt_dim=$1;n_heads=$2" \
    --wandb "run" \
    --job_id ${SLURM_JOB_ID} \
    --output ${REGAT_SAVE_MODELS_TRAIN_PATH}

#REGAT_SAVE_EXPERIMENT_TRAIN_PATH=$(find ${REGAT_SAVE_MODELS_TRAIN_PATH} -type d -name "${SLURM_JOB_ID}*" -print -quit)  
#mv "./sbatch-scripts/${USERNAME}/${SLURM_JOB_ID}-"*".log" ${REGAT_SAVE_EXPERIMENT_TRAIN_PATH}
