# %%

import os
from os.path import join, exists
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, random_split
import random
import json

from dataset import Dictionary, VQAFeatureDataset, VisualGenomeFeatureDataset
from dataset import tfidf_from_questions
from dataset_cp_v2 import VQA_cp_Dataset, Image_Feature_Loader
from model.regat import build_regat
from config.parser import parse_with_config
from train import train
import utils
from utils import trim_collate

from main import parse_args

import matplotlib.pyplot as plt

# %% 
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 
# %%

data_folder = '/hpcwork/lect0099/data'
relation_type = 'implicit'
adaptive = True
imp_pos_emb_dim = 64


# %% 
dictionary = Dictionary.load_from_file(
                join(data_folder, 'glove/dictionary.pkl'))
val_dset = VQAFeatureDataset(
        'val', dictionary, relation_type, adaptive=adaptive,
        pos_emb_dim=imp_pos_emb_dim, dataroot= data_folder)
train_dset = VQAFeatureDataset(
        'train', dictionary, relation_type,
        adaptive=adaptive, pos_emb_dim=imp_pos_emb_dim,
        dataroot=data_folder)
# %%
