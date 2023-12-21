import os
import taichi as ti

from Autoencoders.ConvAE_1 import *
from Dataset_processing import *

feature_vector_size = 512
res = 256

attr_name_1 = 'velocity'
attr_name_2 = 'strainRate2vorticity'
attr_name_3 = 'density'
platform = 'cuda'

'''
Training process:
'''
main_folder_path = './dataset_train'
model_path  = os.path.join(main_folder_path, 'model')

dataset_file_path_1 = os.path.join(main_folder_path, 'dataset')
dataset_file_path_2 = os.path.join(main_folder_path, 'dataset')
dataset_file_path_3 = os.path.join(main_folder_path, 'dataset')

model = TrainConvAutoencoder_1(feature_vector_size, res, attr_name_1, dataset_file_path_1, 
                               attr_name_2, dataset_file_path_2, attr_name_3, dataset_file_path_3, platform)

model.train_velocityBased(num_epochs=4000, network_model_path=model_path,  former_model_file_path=None)
# model.train_vorticityBased(num_epochs=4000, network_model_path=model_path, former_model_file_path=None)
# model.train_histBased(     num_epochs=4000, network_model_path=model_path, former_model_file_path=None)