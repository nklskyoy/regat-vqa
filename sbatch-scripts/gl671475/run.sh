#!/usr/local_rwth/bin/zsh 

### SBATCH Section

#SBATCH --job-name=VQA-ReGAT
#SBATCH --output=%j-output.log
#SBATCH --error=%j-error.log
#SBATCH --nodes=1
#SBATCH --array=1-9
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:volta:2
#SBATCH --mem-per-gpu=32G
#SBATCH --time=02:00:00

# Extracts the TID of the user
USERNAME=$(whoami)

REGAT_LOCAL_CODE_PATH="/rwthfs/rz/cluster/home/${USERNAME}/vqa_regat/sbatch-scripts/${USERNAME}"

# By default, run.sh operates in code directory
cd ${REGAT_LOCAL_CODE_PATH}

# Program Section

bs_list=(128 192 256)
lr_list=(0.001 0.005 0.01)

for bs in "${bs_list[@]}"; do
  for lr in "${lr_list[@]}"; do
    srun bs-lr/bs${bs}-lr${lr}-run.sh
  done
done
