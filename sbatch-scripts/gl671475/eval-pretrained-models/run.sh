#!/usr/local_rwth/bin/zsh 


tail -n +2 config.csv | while IFS=";" read -r output_folder
do
    sbatch eval.sh \
        ${output_folder}
done
