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
sbatch experiments/bs_192_opt_Adam_schedule_custom-2e-4_4e-3_4e-5_ep_5_15_20-run.sh
sbatch experiments/bs_192_opt_Adam_schedule_OCLR-2e-4_5e-3_5e-5_ep_20-run.sh
sbatch experiments/bs_192_opt_Adam_wd_1e-2_schedule_custom-3e-4_6e-3_5e-5_ep_5_15_20-run.sh
sbatch experiments/bs_192_opt_Adam_wd_1e-2_schedule_OCLR-3e-4_6e-3_5e-5_ep_20-run.sh