import numpy as np
from sklearn.metrics import DistanceMetric
from datasets.act_dataset import ACTDataset
import scipy.io as sio
import pickle
import torch

TOP_K = 128

config = {
    "dataset": {
      "name": "act",
      "path": "/opt/tiger/dataset/cvact",            
      "same_area": True,              
      "print_bool": False            
    },
    "model": {
      "embed_dim": 1024,
      "head_dim": 1024,
      "n_query": 1,
      "n_head": 16,
      "n_attn": 2
    },
    "training": {
        "epochs": 40,
        "batch_size": 16,
        "ref_batch_size": 128,
        "shuffle_batch_size": 128,
        "learning_rate": 5e-4,
        "weight_decay": 0,
        "img_height": 256,
        "img_width": 256,
        "multi_layer_token": True,
        "use_conv": False,
        "fix_logit": True,
        "logit_scale": 0.07,
        "use_res": True,
        "img_size": 384,
        "model_path": "/opt/tiger/dino/GeoDiffuse/weights/pretrained/cvact/convnext_base.fb_in22k_ft_in1k_384/weights_e36_90.8149.pth"
      },
    "run_settings": {
      "model_type": "infonce",
      "verbose": False
    },
    "weights":{
      "conv": "/opt/tiger/dino/GeoDiffuse/weights/pretrained/cvact/convnext_base.fb_in22k_ft_in1k_384/weights_e36_90.8149.pth"
    }
}

# sanity check
# dataset = ACTDataset(config, mode='train')
# print("train_ids[0]:", dataset.train_ids[0])
# print("train_idsnum[0]:", dataset.train_idsnum[0])
# print("idx2numidx lookup:", dataset.idx2numidx[dataset.train_ids[0]])
# print("reverse lookup:", dataset.numidx2idx[dataset.train_idsnum[0]])

dataset = ACTDataset(config, mode='train')

anuData = sio.loadmat('./datasets/ACT_data.mat')

utm = anuData["utm"]
ids = anuData['panoIds']

print("Example from .mat:", ids[0], type(ids[0]), type(ids[0][0]))
print("Example from dataset:", dataset.train_ids[0], type(dataset.train_ids[0]))

idx2numidx = dataset.idx2numidx


train_ids_set = set(dataset.train_ids)
train_idsnum_list = []
 

utm_coords = dict()
utm_coords_list = []

print("check", "zzzyqrKa07ol9Doq35rB9w" in train_ids_set)

for i, idx in enumerate(ids):
    print(idx) 
    break
    idx = str(idx)
    
    if idx in train_ids_set:
        coordinates = (float(utm[i][0]), float(utm[i][1]))
        utm_coords[idx] = coordinates
        utm_coords_list.append(coordinates) 
        train_idsnum_list.append(idx2numidx[idx])
    
    
print("Length Train Ids:", len(utm_coords_list))

train_idsnum_lookup = np.array(train_idsnum_list)
    

print("Length of gps coords : " +str(len(utm_coords_list)))
print("Calculation...")

dist = DistanceMetric.get_metric("euclidean")
dm = dist.pairwise(utm_coords_list, utm_coords_list)
print("Distance Matrix:", dm.shape)


dm_torch = torch.from_numpy(dm)
dm_torch = dm_torch.fill_diagonal_(dm.max())


values, ids = torch.topk(dm_torch, k=TOP_K, dim=1, largest=False)

values_near_numpy = values.numpy()
ids_near_numpy = ids.numpy()

near_neighbors = dict()

for i, idnum in enumerate(train_idsnum_list):
    
    near_neighbors[idnum] = train_idsnum_lookup[ids_near_numpy[i]].tolist()

print("Saving...") 
with open("./datasets/cvact_gps_dict.pkl", "wb") as f:
    pickle.dump(near_neighbors, f)