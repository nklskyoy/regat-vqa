#!/usr/local_rwth/bin/zsh 

nongt_dim=(10 20 30)
num_heads=(8 16 32)


#nongt_dim=(20)
#num_heads=(32)

BASE_DIR=/rwthfs/rz/cluster/home/${USERNAME}/dev/lab2/vqa_regat/sbatch-scripts/hf201627/n_heads-nongt_dim-lr/

###  Program Section

for dim in "${nongt_dim[@]}"; do
  for nh in "${num_heads[@]}"; do
    echo $dim
    sbatch $BASE_DIR/experiment.sh \
        $dim \
        $nh
  done
done