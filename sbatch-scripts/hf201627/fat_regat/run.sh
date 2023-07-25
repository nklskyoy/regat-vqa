#!/usr/local_rwth/bin/zsh 

BASE_DIR="/rwthfs/rz/cluster/home/hf201627/dev/lab2/vqa_regat/sbatch-scripts/hf201627/fat_regat"

tail -n +2 "${BASE_DIR}/hyperparam.csv" | while IFS=";" read -r init_lr peak_lr final_lr begin_constant begin_decay
do
        echo "init_lr: $init_lr"
        echo "peak_lr: $peak_lr"
        echo "final_lr: $final_lr"
        echo "begin_constant: $begin_constant"
        echo "begin_decay: $begin_decay"

        sbatch ${BASE_DIR}/experiment.sh \
            "$init_lr"\
            "$peak_lr"\
            "$final_lr"\
            "$begin_constant"\
            "$begin_decay"
done
