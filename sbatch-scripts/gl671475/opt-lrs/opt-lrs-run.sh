#!/usr/local_rwth/bin/zsh 

### SBATCH Section

#SBATCH --job-name=VQA-ReGAT
#SBATCH --output=%j-output.log
#SBATCH --error=%j-error.log
#SBATCH --nodes=1
#SBATCH --time=10:00
#SBATCH --account=lect0099

# Extracts the TID of the user
USERNAME=$(whoami)

REGAT_LOCAL_CODE_PATH="/rwthfs/rz/cluster/home/${USERNAME}/vqa_regat/sbatch-scripts/${USERNAME}"

# By default, run.sh operates in code directory
cd ${REGAT_LOCAL_CODE_PATH}

###  Program Section
sbatch opt-lrs/bs64-opt-Adam-schedule-custom-1e-4_2e-3_5e-5-epochs-5_15_20-run.sh
sbatch opt-lrs/bs64-opt-Adam-schedule-OCLR-1e-4_2e-3_5e-5-run.sh
sbatch opt-lrs/bs64-opt-AdamW-wd-1e-2-schedule-custom-1e-4_2e-3_5e-5-epochs-5_15_20-run.sh
sbatch opt-lrs/bs64-opt-AdamW-wd-1e-2-schedule-OCLR-1e-4_2e-3_5e-5-run.sh