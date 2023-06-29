#!/usr/local_rwth/bin/zsh 

### SBATCH Section

#SBATCH --job-name=VQA-ReGAT
#SBATCH --output=%j-output.log
#SBATCH --error=%j-error.log
#SBATCH --nodes=1
#SBATCH --gres=gpu:volta:2
#SBATCH --mem=64G
#SBATCH --time=01:00:00

export WANDB_API_KEY=4eeb4bff5fc5678d495fd6ac0a3c0a8bfab3ac3c

module load CUDA/11.8.0
source /home/hf201627/anaconda3/bin/activate regat

# Extracts the TID of the user
USERNAME=$(whoami)

REGAT_LOCAL_CODE_PATH="/rwthfs/rz/cluster/home/${USERNAME}/vqa_regat"
REGAT_DATA_AND_MODELS_PATH="/rwthfs/rz/cluster/hpcwork/lect0099" 

REGAT_SAVE_MODELS_TRAIN_PATH="${REGAT_DATA_AND_MODELS_PATH}/saved_models/${USERNAME}/train"
REGAT_SAVE_MODELS_EVAL_PATH="${REGAT_DATA_AND_MODELS_PATH}/saved_models/${USERNAME}/eval"


# By default, run.sh operates in code directory
# cd ${REGAT_LOCAL_CODE_PATH}

### Program Section

# (Uncomment for training)
python3 fat.py --config config/ban_vqa-o.json  --epochs 1 --name "fat_adamax_0.001_wd_0" --job_id ${SLURM_JOB_ID} --output ${REGAT_SAVE_MODELS_TRAIN_PATH}
REGAT_SAVE_EXPERIMENT_TRAIN_PATH=$(find ${REGAT_SAVE_MODELS_TRAIN_PATH} -type d -name "${SLURM_JOB_ID}*" -print -quit)  
#mv "./sbatch-scripts/${SLURM_JOB_ID}-"*".log" ${REGAT_SAVE_EXPERIMENT_TRAIN_PATH}

# (Uncomment for evaluation)
# python3 eval.py --data_folder ${REGAT_DATA_AND_MODELS_PATH}/data --output_folder ${REGAT_DATA_AND_MODELS_PATH}/pretrained_models/regat_spatial/ban_1_spatial_vqa_1687

# (DON'T UNCOMMENT)
# REGAT_SAVE_EXPERIMENT_EVAL_PATH=$(find ${REGAT_SAVE_MODELS_EVAL_PATH} -type d -name "${SLURM_JOB_ID}*" -print -quit)  
# mv "./sbatch-scripts/${SLURM_JOB_ID}-output.log" ${REGAT_SAVE_EXPERIMENT_EVAL_PATH}