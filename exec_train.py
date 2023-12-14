import os
import torch
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
model_path = os.path.join('output', 'dict')
former_model_file_path = os.path.join('output', 'dict', 'epochs_338.pth')
platform = 'cuda'

ti.init(arch=ti.cuda)
torch.autograd.set_detect_anomaly(True)
torch.set_printoptions(threshold=5000, edgeitems=10, linewidth=200)

model = TrainConvAutoencoder_1(feature_vector_size, res, attr_name_1, dataset_file_path_1, attr_name_2, dataset_file_path_2, attr_name_3, dataset_file_path_3, platform)
# model.train(num_epochs=200, network_model_path=model_path, former_model_file_path=former_model_file_path)
model.test(num_epochs=200, network_model_path=model_path, former_model_file_path=former_model_file_path)
