import os
import taichi as ti

from Autoencoders.ConvAE_1 import *
from Dataset_processing import *

strategy_list = ['skip_bottleneck', 'whole']
model_path = os.path.join('/workspace/models/model_512')
model_file_list = [None, os.path.join(model_path,'epochs_39999.pth')]

if_freeze_parameters = False
if_crop = True
strategy = strategy_list[1]
model_file = model_file_list[0]
exclude_threshold = None

res = 256

attr_name_1 = 'velocity'
attr_name_2 = 'strainRate2vorticity'
attr_name_3 = 'density'
platform = 'cuda'

'''
Training process:
'''
main_folder_path = '../dataset_train'
output_path = os.path.join(main_folder_path, 'test_output')

dataset_file_path_1 = os.path.join(main_folder_path, 'dataset')
dataset_file_path_2 = os.path.join(main_folder_path, 'dataset')
dataset_file_path_3 = os.path.join(main_folder_path, 'dataset')

model = TrainConvAutoencoder_1(res, attr_name_1, dataset_file_path_1, 
                               attr_name_2, dataset_file_path_2, attr_name_3, dataset_file_path_3, platform, False)

# model.train_velocityBased (num_epochs=8000, network_model_path=model_path,  former_model_file_path=None)
# model.output_bottleneck(   model_file_path='./model/epochs_299.pth', output_path=output_path)
model.train_vorticityBased(num_epochs=40000, network_model_path=model_path, strategy=strategy, former_model_file_path=model_file, save_step=100, freeze_param=if_freeze_parameters, crop=if_crop, exclude_threshold=exclude_threshold)
# model.train_histBased(     num_epochs=8000, network_model_path=model_path, former_model_file_path=None)

# datavis_1darray(output_path, output_path, 'bottleneck', 0, 659)