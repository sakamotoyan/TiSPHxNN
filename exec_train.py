import os

import taichi as ti

from Autoencoders.ConvAE_1 import *
from torch.utils.data import DataLoader

feature_vector_size=128
res = 256

attr_name_1 = 'velocity'
attr_name_2 = 'strainRate2vorticity'
attr_name_3 = 'density'
dataset_file_path_1 = os.path.join('output', 'organized')
dataset_file_path_2 = os.path.join('output', 'organized')   
dataset_file_path_3 = os.path.join('output', 'organized')
platform = 'cuda'

ti.init(arch=ti.cuda)

model = TrainConvAutoencoder_1(feature_vector_size, res, attr_name_1, dataset_file_path_1, attr_name_2, dataset_file_path_2, attr_name_3, dataset_file_path_3, platform)
model.train(1)
# for input, target in model.data_loader:
#     print(input.shape)
#     print(target[0].sum())
#     break


# train.train(1)