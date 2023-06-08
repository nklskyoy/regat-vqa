
from  dataset import Dictionary, VQAFeatureDataset, VisualGenomeFeatureDataset
import matplotlib.pyplot as plt; plt.ioff()
import numpy as np
from PIL import Image
from os.path import join, exists
import matplotlib.patches as patches


vqa_data_folder = '/hpcwork/lect0099/data'
coco_data_folder = '/work/lect0099/coco-2014/train2014/train2014'
coco_img_prefix = 'COCO_train2014'
coco_img_idx_len = 12 

adaptive = True
imp_pos_emb_dim = 64
relation_type = "implicit"


dictionary = Dictionary.load_from_file(
                join(vqa_data_folder, 'glove/dictionary.pkl'))

train_dset = VQAFeatureDataset(
        'train', dictionary, relation_type,
        adaptive=adaptive, pos_emb_dim=imp_pos_emb_dim,
        dataroot=vqa_data_folder)


def plot_with_bb(idx, axs):
    # https://stackoverflow.com/questions/37435369/matplotlib-how-to-draw-a-rectangle-on-image
    img_idx=  str(train_dset[idx][5])
    img_idx = img_idx.zfill(len('000000324104'))
    bb = train_dset[idx][6]

    img = np.asarray(
        Image.open( join(coco_data_folder, 
                         "{pref}_{id}.jpg".format(pref=coco_img_prefix,id=img_idx) )
                ))
    for b in bb:
        x = [b[0], b[0], b[2], b[2], b[0]]
        y = [b[1], b[3], b[3], b[1], b[1]]
        axs.plot(x, y, color="red")
    axs.imshow(img)
    






id = 73

fig, ax = plt.subplots()
plot_with_bb(id, ax)
plt.show()

