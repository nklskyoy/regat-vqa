#%%
import h5py
import numpy as np
import os
import torch

# %%
dataroot='/hpcwork/lect0099/data'
prefix = '36'
adaptive = False
name = 'train'
h5_dataroot = dataroot+"/Bottom-up-features-adaptive"\
            if adaptive else dataroot+"/Bottom-up-features-fixed"
h5_path=  os.path.join(h5_dataroot, '%s%s.hdf5' %
                               (name, '' if adaptive else prefix))
# %%
semantic_adj_matrix = None
with h5py.File(h5_path, 'r') as hf:
    semantic_adj_matrix = np.array(hf.get('semantic_adj_matrix'))
    np.array(hf.get('image_features'))

# %%
result = []
i = 0
adj_matrix = torch.from_numpy(semantic_adj_matrix).double()
#for i in range(1, label_num+1):
index = torch.nonzero((adj_matrix == 1).view(-1).data).squeeze()

# %%


curr_result = torch.zeros(
    adj_matrix.shape[0], adj_matrix.shape[1], adj_matrix.shape[2])
curr_result = curr_result.view(-1)
curr_result[index] += 1
result.append(curr_result.view(
    (adj_matrix.shape[0], adj_matrix.shape[1],
        adj_matrix.shape[2], 1)))
result = torch.cat(result, dim=3)

# %%
label_num = 10
result = []
device =  'cpu'
for i in range(1, label_num+1):
    index = torch.nonzero((adj_matrix == i).view(-1).data).squeeze().to(device)
    curr_result = torch.zeros(
        adj_matrix.shape[0], adj_matrix.shape[1], adj_matrix.shape[2]).to(device)
    curr_result = curr_result.view(-1)
    curr_result[index] += 1
    result.append(curr_result.view(
        (adj_matrix.shape[0], adj_matrix.shape[1],
            adj_matrix.shape[2], 1)))
result = torch.cat(result, dim=3)
# %%
