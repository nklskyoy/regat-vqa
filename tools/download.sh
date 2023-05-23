## Copyright (c) Microsoft Corporation.
## Licensed under the MIT license.

## This code is modified by Linjie Li from Hengyuan Hu's repository.
## https://github.com/hengyuan-hu/bottom-up-attention-vqa
## GNU General Public License v3.0

## Script for downloading data

# VQA Questions
wget -P $DATA_ROOT https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip
unzip $DATA_ROOT/v2_Questions_Train_mscoco.zip -d $DATA_ROOT/Questions
rm $DATA_ROOT/v2_Questions_Train_mscoco.zip

wget -P $DATA_ROOT https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip
unzip $DATA_ROOT/v2_Questions_Val_mscoco.zip -d $DATA_ROOT/Questions
rm $DATA_ROOT/v2_Questions_Val_mscoco.zip

wget -P $DATA_ROOT https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip
unzip $DATA_ROOT/v2_Questions_Test_mscoco.zip -d $DATA_ROOT/Questions
rm $DATA_ROOT/v2_Questions_Test_mscoco.zip

# VQA Annotations
wget -P $DATA_ROOT https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip
wget -P $DATA_ROOT https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip
unzip $DATA_ROOT/v2_Annotations_Train_mscoco.zip -d $DATA_ROOT/Answers
rm $DATA_ROOT/v2_Annotations_Train_mscoco.zip
unzip $DATA_ROOT/v2_Annotations_Val_mscoco.zip -d $DATA_ROOT/Answers
rm $DATA_ROOT/v2_Annotations_Val_mscoco.zip

# VQA cp-v2 Questions
mkdir $DATA_ROOT/cp_v2_questions
wget -P $DATA_ROOT/cp_v2_questions https://computing.ece.vt.edu/~aish/vqacp/vqacp_v2_train_questions.json 
wget -P $DATA_ROOT/cp_v2_questions https://computing.ece.vt.edu/~aish/vqacp/vqacp_v2_test_questions.json 

# VQA cp-v2 Annotations
mkdir $DATA_ROOT/cp_v2_annotations
wget -P $DATA_ROOT/cp_v2_annotations https://computing.ece.vt.edu/~aish/vqacp/vqacp_v2_train_annotations.json 
wget -P $DATA_ROOT/cp_v2_annotations https://computing.ece.vt.edu/~aish/vqacp/vqacp_v2_train_annotations.json

# Visual Genome Annotations
mkdir $DATA_ROOT/visualGenome
wget -P $DATA_ROOT/visualGenome https://convaisharables.blob.core.windows.net/vqa-regat/data/visualGenome/image_data.json
wget -P $DATA_ROOT/visualGenome https://convaisharables.blob.core.windows.net/vqa-regat/data/visualGenome/question_answers.json

# GloVe Vectors and dictionary
wget -P $DATA_ROOT https://convaisharables.blob.core.windows.net/vqa-regat/data/glove.zip
unzip $DATA_ROOT/glove.zip -d $DATA_ROOT/glove
rm $DATA_ROOT/glove.zip

# Image Features
# adaptive
# WARNING: This may take a while
mkdir $DATA_ROOT/Bottom-up-features-adaptive
wget -P $DATA_ROOT/Bottom-up-features-adaptive https://convaisharables.blob.core.windows.net/vqa-regat/data/Bottom-up-features-adaptive/train.hdf5
wget -P $DATA_ROOT/Bottom-up-features-adaptive https://convaisharables.blob.core.windows.net/vqa-regat/data/Bottom-up-features-adaptive/val.hdf5
wget -P $DATA_ROOT/Bottom-up-features-adaptive https://convaisharables.blob.core.windows.net/vqa-regat/data/Bottom-up-features-adaptive/test2015.hdf5

# fixed
# WARNING: This may take a while
mkdir $DATA_ROOT/Bottom-up-features-fixed
wget -P $DATA_ROOT/Bottom-up-features-fixed https://convaisharables.blob.core.windows.net/vqa-regat/data/Bottom-up-features-fixed/train36.hdf5
wget -P $DATA_ROOT/Bottom-up-features-fixed https://convaisharables.blob.core.windows.net/vqa-regat/data/Bottom-up-features-fixed/val36.hdf5
wget -P $DATA_ROOT/Bottom-up-features-fixed https://convaisharables.blob.core.windows.net/vqa-regat/data/Bottom-up-features-fixed/test2015_36.hdf5

# imgids
wget -P $DATA_ROOT/ https://convaisharables.blob.core.windows.net/vqa-regat/data/imgids.zip
unzip $DATA_ROOT/imgids.zip -d $DATA_ROOT/imgids
rm $DATA_ROOT/imgids.zip

# Download Pickle caches for the pretrained model
# and extract pkl files under $DATA_ROOT/cache/.
wget -P $DATA_ROOT https://convaisharables.blob.core.windows.net/vqa-regat/data/cache.zip
unzip $DATA_ROOT/cache.zip -d $DATA_ROOT/cache
rm $DATA_ROOT/cache.zip

# Download pretrained models
# and extract files under pretrained_models.
#wget https://convaisharables.blob.core.windows.net/vqa-regat/pretrained_models.zip
#unzip pretrained_models.zip -d pretrained_models/
#rm pretrained_models.zip
