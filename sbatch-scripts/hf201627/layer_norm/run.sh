#!/usr/local_rwth/bin/zsh 

BASE_DIR="/rwthfs/rz/cluster/home/hf201627/dev/lab2/vqa_regat/sbatch-scripts/hf201627/layer_norm"

tail -n +2 "${BASE_DIR}/hyperparam.csv" | while IFS=";" read -r init_lr peak_lr final_lr begin_constant begin_decay layer_norm
do
        sbatch ${BASE_DIR}/experiment.sh \
            "$init_lr"\
            "$peak_lr"\
            "$final_lr"\
            "$begin_constant"\
            "$begin_decay"\
            "$layer_norm"
done
