import os
import taichi as ti

from Autoencoders.ConvAE_1 import *
from Dataset_processing import *

platform = 'cuda'
feature_vector_size = 1024
res = 256

model_epoch = 5499

main_folder_path = '../dataset_train'
model_folder_path = os.path.join('./model')
vel_model_file_path   = os.path.join(model_folder_path,  f'epochs_{model_epoch}.pth')
test_result_vel_path  = os.path.join('./output')
vis_path              = os.path.join('./output')

attr_name_1 = 'velocity'
attr_name_2 = 'strainRate2vorticity'
attr_name_3 = 'density'
'''
Testing process:
'''
dataset_file_path_1 = os.path.join(main_folder_path, 'dataset')
dataset_file_path_2 = os.path.join(main_folder_path, 'dataset')
dataset_file_path_3 = os.path.join(main_folder_path, 'dataset')
# model = TrainConvAutoencoder_1(res, attr_name_1, dataset_file_path_1, 
#                                attr_name_2, dataset_file_path_2, attr_name_3, dataset_file_path_3, platform, False)

# model.test(vel_model_file_path,  test_result_vel_path)

number_of_frames = 3990
# scivis_R2toR1(test_result_vel_path,  test_result_vel_path,  0, number_of_frames, 'input_vorticity',  stride=1)
scivis_R2toR1(test_result_vel_path,  vis_path,  0, number_of_frames, 'output_vorticity', stride=5)
# scivis_R2toR2(test_result_vel_path,  test_result_vel_path,  0, number_of_frames, 'input_velocity')
# scivis_R2toR2(test_result_vel_path,  vis_path,  0, number_of_frames, 'output_velocity')
# datavis_hist( test_result_vel_path,  test_result_vel_path,  'input_vorticity_hist',  0, number_of_frames)
# datavis_hist( test_result_vel_path,  vis_path,  'output_vorticity_hist', 0, number_of_frames)

# scivis_R2toR1(test_result_vel_path,  test_result_vel_path,  0, 750, 'input_vorticity')



# vort_model_file_path = os.path.join('output', 'dict_vort', 'epochs_999.pth')
# test_result_vort_path = os.path.join('output', 'network_testing_vort')
# model.test(vort_model_file_path, test_result_vort_path)

# hist_model_file_path = os.path.join('output', 'dict_hist', 'epochs_199.pth')
# test_result_hist_path = os.path.join('output', 'network_testing_hist')
# model.test(hist_model_file_path, test_result_hist_path)

''' 
Visualization 
'''
# scivis_R2toR1(test_result_vort_path, test_result_vort_path, 0, 700, 'input_vorticity')
# scivis_R2toR1(test_result_vort_path, test_result_vort_path, 0, 700, 'output_vorticity')
# datavis_hist( test_result_vort_path, test_result_vort_path, 'input_vorticity_hist',  0, 700)
# datavis_hist( test_result_vort_path, test_result_vort_path, 'output_vorticity_hist', 0, 700)

# scivis_R2toR1(test_result_hist_path, test_result_hist_path, 0, 750, 'input_vorticity')
# scivis_R2toR1(test_result_hist_path, test_result_hist_path, 0, 750, 'output_vorticity')
# datavis_hist( test_result_hist_path, test_result_hist_path, 'input_vorticity_hist',  0, 150)
# datavis_hist( test_result_hist_path, test_result_hist_path, 'output_vorticity_hist', 0, 150)
