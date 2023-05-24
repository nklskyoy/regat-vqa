#!/usr/local_rwth/bin/zsh 

### SBATCH Section

#SBATCH --job-name=VQA-ReGAT
#SBATCH --output=%j-test.out
#SBATCH --error=%j-test.err
#SBATCH --nodes=2
#SBATCH --gres=gpu:volta:2
#SBATCH --mem=64G
#SBATCH --time=08:00:00

module load CUDA/11.8.0
source /rwthfs/rz/cluster/home/hf201627/miniconda3/bin/activate
conda activate regat-origin

# By default, run.sh operates in code directory
#cd ${REGAT_CODE_PATH}

# Program Section
python3 main.py --config config/butd_vqa-o.json --epochs 1
# python3 eval.py --output_folder pretrained_models/regat_implicit/ban_1_implicit_vqa_196




