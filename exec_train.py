import os
import taichi as ti

from Autoencoders.ConvAE_1 import *
from Dataset_processing import *

feature_vector_size=128
res = 256

attr_name_1 = 'velocity'
attr_name_2 = 'strainRate2vorticity'
attr_name_3 = 'density'
platform = 'cuda'

# ti.init(arch=ti.cuda)

'''
Training process:
'''
model_path_vel  = os.path.join('output', 'dict_vel')
model_path_vort = os.path.join('output', 'dict_vort')
model_path_hist = os.path.join('output', 'dict_hist')

dataset_file_path_1 = os.path.join('output', 'organized_train')
dataset_file_path_2 = os.path.join('output', 'organized_train')   
dataset_file_path_3 = os.path.join('output', 'organized_train')
model = TrainConvAutoencoder_1(feature_vector_size, res, attr_name_1, dataset_file_path_1, 
                               attr_name_2, dataset_file_path_2, attr_name_3, dataset_file_path_3, platform)

model.train_velocityBased( num_epochs=2000, network_model_path=model_path_vel,  former_model_file_path=None)
model.train_vorticityBased(num_epochs=2000, network_model_path=model_path_vort, former_model_file_path=None)
model.train_histBased(     num_epochs=2000, network_model_path=model_path_hist, former_model_file_path=None)


'''
Testing process:
'''
dataset_file_path_1 = os.path.join('output', 'organized_test')
dataset_file_path_2 = os.path.join('output', 'organized_test')
dataset_file_path_3 = os.path.join('output', 'organized_test')
model = TrainConvAutoencoder_1(feature_vector_size, res, attr_name_1, dataset_file_path_1, 
                               attr_name_2, dataset_file_path_2, attr_name_3, dataset_file_path_3, platform)


vel_model_file_path  = os.path.join('output', 'dict_vel',  'epochs_599.pth')
test_result_vel_path  = os.path.join('output', 'network_testing_vel')
model.test(vel_model_file_path,  test_result_vel_path)


# vort_model_file_path = os.path.join('output', 'dict_vort', 'epochs_999.pth')
# test_result_vort_path = os.path.join('output', 'network_testing_vort')
# model.test(vort_model_file_path, test_result_vort_path)

# hist_model_file_path = os.path.join('output', 'dict_hist', 'epochs_999.pth')
# test_result_hist_path = os.path.join('output', 'network_testing_hist')
# model.test(hist_model_file_path, test_result_hist_path)

# # Visualization
scivis_R2toR1(test_result_vel_path,  test_result_vel_path,  0, 150, 'output_vorticity')
scivis_R2toR1(test_result_vel_path,  test_result_vel_path,  0, 150, 'input_vorticity')
datavis_hist( test_result_vel_path,  test_result_vel_path,  'output_vorticity_hist', 0, 150)
datavis_hist( test_result_vel_path,  test_result_vel_path,  'input_vorticity_hist',  0, 150)

# scivis_R2toR1(test_result_vort_path, test_result_vort_path, 0, 150, 'input_vorticity')
# scivis_R2toR1(test_result_vort_path, test_result_vort_path, 0, 150, 'output_vorticity')
# datavis_hist( test_result_vort_path, test_result_vort_path, 'input_vorticity_hist',  0, 150)
# datavis_hist( test_result_vort_path, test_result_vort_path, 'output_vorticity_hist', 0, 150)

# scivis_R2toR1(test_result_hist_path, test_result_hist_path, 0, 150, 'input_vorticity')
# scivis_R2toR1(test_result_hist_path, test_result_hist_path, 0, 150, 'output_vorticity')
# datavis_hist( test_result_hist_path, test_result_hist_path, 'input_vorticity_hist',  0, 150)
# datavis_hist( test_result_hist_path, test_result_hist_path, 'output_vorticity_hist', 0, 150)
