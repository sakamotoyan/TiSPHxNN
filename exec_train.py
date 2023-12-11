import os
from Autoencoders.ConvAE_1 import *
from torch.utils.data import DataLoader

feature_vector_size=128

attr_name_1 = 'velocity'
attr_name_2 = 'strainRate2vorticity'
dataset_file_path_1 = os.path.join('output', 'organized')
dataset_file_path_2 = os.path.join('output', 'organized')   
platform = 'cuda'

train = TrainConvAutoencoder_1(feature_vector_size, attr_name_1, dataset_file_path_1, attr_name_2, dataset_file_path_2, platform)

for input_velocity, input_vorticity in train.data_loader:
    print(input_velocity.shape)
    print(input_vorticity.shape)
    break


# train.train(1)