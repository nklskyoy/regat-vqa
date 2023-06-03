#!/usr/local_rwth/bin/zsh 

### SBATCH Section

#SBATCH --job-name=VQA-ReGAT
#SBATCH --output=%j-output.log
#SBATCH --error=%j-error.log
#SBATCH --nodes=1
#SBATCH --time=15:00

# Extracts the TID of the user
USERNAME=$(whoami)

REGAT_LOCAL_CODE_PATH="/rwthfs/rz/cluster/home/${USERNAME}/vqa_regat/sbatch-scripts/${USERNAME}"

# By default, run.sh operates in code directory
cd ${REGAT_LOCAL_CODE_PATH}

###  Program Section

# DESC: Launched initial experiments for batch_size and learning_rates

bs_list=(256)
lr_list=(0.001)

for bs in "${bs_list[@]}"; do
  for lr in "${lr_list[@]}"; do
    sbatch bs-lr/bs${bs}-lr${lr}-run.sh
  done
done

# bs_list = (64 192)
# opt_list = ("Adam" "SGDM")

# for bs in "${bs_list[@]}"; do
#  for opt in "${opt_list[@]}"; do
#    sbatch bs-opt/bs${bs}-opt-${opt}
#  done 
# done 
