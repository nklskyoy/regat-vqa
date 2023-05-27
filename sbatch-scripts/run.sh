#!/usr/local_rwth/bin/zsh 

### SBATCH Section

#SBATCH --job-name=VQA-ReGAT
#SBATCH --output=%j-output.log
#SBATCH --error=%j-error.log
#SBATCH --nodes=1
#SBATCH --gres=gpu:volta:2
#SBATCH --mem=64G
#SBATCH --time=01:00:00

module load CUDA/11.8.0
source /rwthfs/rz/cluster/home/gl671475/miniconda3/bin/activate
conda activate regat_common

export WANDB_API_KEY=32abfb91875c02ffd9d1f8d294baad4f5eabbdf7

# Extracts the TID of the user
USERNAME=$(whoami)

REGAT_LOCAL_CODE_PATH="/rwthfs/rz/cluster/home/${USERNAME}/vqa_regat"
REGAT_DATA_AND_MODELS_PATH="/rwthfs/rz/cluster/hpcwork/lect0099" 

REGAT_SAVE_MODELS_TRAIN_PATH="${REGAT_DATA_AND_MODELS_PATH}/saved_models/${USERNAME}/train"
REGAT_SAVE_MODELS_EVAL_PATH="${REGAT_DATA_AND_MODELS_PATH}/saved_models/${USERNAME}/eval"


# By default, run.sh operates in code directory
cd ${REGAT_LOCAL_CODE_PATH}

# Program Section

# (Uncomment for training)
python3 main.py --config config/ban_vqa.json --relation_type spatial  --epochs 1 --name "adamax_0.001_wd_0" --job_id ${SLURM_JOB_ID} --output ${REGAT_SAVE_MODELS_TRAIN_PATH}
REGAT_SAVE_EXPERIMENT_TRAIN_PATH=$(find ${REGAT_SAVE_MODELS_TRAIN_PATH} -type d -name "${SLURM_JOB_ID}*" -print -quit)  
mv "./sbatch-scripts/${SLURM_JOB_ID}-"*".log" ${REGAT_SAVE_EXPERIMENT_TRAIN_PATH}

# (Uncomment for evaluation)
# python3 eval.py --data_folder ${REGAT_DATA_AND_MODELS_PATH}/data --output_folder ${REGAT_DATA_AND_MODELS_PATH}/pretrained_models/regat_spatial/ban_1_spatial_vqa_1687

# (DON'T UNCOMMENT)
# REGAT_SAVE_EXPERIMENT_EVAL_PATH=$(find ${REGAT_SAVE_MODELS_EVAL_PATH} -type d -name "${SLURM_JOB_ID}*" -print -quit)  
# mv "./sbatch-scripts/${SLURM_JOB_ID}-output.log" ${REGAT_SAVE_EXPERIMENT_EVAL_PATH}
