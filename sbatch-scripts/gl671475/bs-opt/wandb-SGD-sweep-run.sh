#!/usr/local_rwth/bin/zsh 

### SBATCH Section

#SBATCH --job-name=VQA-ReGAT
#SBATCH --output=%j-wandb-sweep-SGD-output.log
#SBATCH --error=%j-wandb-sweep-SGD-error.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=1-4
#SBATCH --gres=gpu:volta:2
#SBATCH --mem-per-gpu=32G
#SBATCH --time=10:00:00
#SBATCH --account=lect0099

module load CUDA/11.8.0
source /rwthfs/rz/cluster/home/gl671475/miniconda3/bin/activate
conda activate regat_common

export WANDB_API_KEY=32abfb91875c02ffd9d1f8d294baad4f5eabbdf7

# Extracts the TID of the user
USERNAME=$(whoami)

REGAT_LOCAL_CODE_PATH="/rwthfs/rz/cluster/home/${USERNAME}/vqa_regat"

# By default, run.sh operates in code directory
cd ${REGAT_LOCAL_CODE_PATH}

###  Program Section

# DESC: Launching a test wandb-sweep
wandb agent lect0099/VQA_ReGAT/twvlhbkl