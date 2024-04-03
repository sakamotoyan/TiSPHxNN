import os

from Autoencoders.ConvAE_velocity import *
from Dataset_processing import *
from util import *

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--featureVectorSize', type=int, default=512)
args = parser.parse_args()

model_path = os.path.join('./model')
model_file_list = [None, os.path.join(model_path,'epochs_699.pth')]
main_folder_path = '../dataset_train'

crop_boundary = True
crop = True
model_file = model_file_list[0]
submodule_type = 1
bottleneck_type = 1

res = 256

print(f'new network type 0{submodule_type}')

exclude_threshold = None

attr_name_1 = 'velocity'
attr_name_2 = 'strainRate2vorticity'
attr_name_3 = 'density'
platform = select_device()

'''
Training process:
'''
output_path = os.path.join(main_folder_path, 'test_output')

dataset_file_path_1 = os.path.join(main_folder_path, 'dataset')
dataset_file_path_2 = os.path.join(main_folder_path, 'dataset')
dataset_file_path_3 = os.path.join(main_folder_path, 'dataset')

network = ConvAutoencoder(submodule_type=submodule_type, bottleneck_type=bottleneck_type, type='train', feature_vector_size=args.featureVectorSize)
print("feature_vector_size: ", args.featureVectorSize)

model = TrainConvAutoencoder(res, attr_name_1, dataset_file_path_1, 
                                  attr_name_2, dataset_file_path_2, 
                                  attr_name_3, dataset_file_path_3, 
                                  platform=platform, lr=5e-5, network=network)

# model.train_velocityBased (num_epochs=8000, network_model_path=model_path,  former_model_file_path=None)
# model.output_bottleneck(   model_file_path='./model/epochs_299.pth', output_path=output_path)
model.train(num_epochs=40000, crop=crop,
            network_model_path=model_path, former_model_file_path=model_file, save_step=10, crop_boundary=crop_boundary, exclude_threshold=exclude_threshold)
# model.train_histBased(     num_epochs=8000, network_model_path=model_path, former_model_file_path=None)

# datavis_1darray(output_path, output_path, 'bottleneck', 0, 659)