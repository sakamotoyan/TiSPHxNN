from Dataset_processing import *

raw_path = './output/raw/'
raw_test_path = './output/raw_test/'
raw_train_path = './output/raw_train/'

organized_path = './output/organized/'
organized_test_path = './output/organized_test/'
organized_train_path = './output/organized_train/'

sciVis_path = './output/sciVis/'
dataVis_path = './output/dataVis/'
hist_path = './output/hist/'
model_path = './output/dict/'
model_output_path = './output/network/'



ti.init(arch=ti.gpu)

# clear_folder(raw_path)
# clear_folder(raw_test_path)
# clear_folder(raw_train_path)
# clear_folder(organized_path)
# clear_folder(organized_test_path)
# clear_folder(organized_train_path)

# clear_folder(sciVis_path)
# clear_folder(dataVis_path)
# clear_folder(hist_path)

# concatDataset('\\Users\\xuyan\\Documents\\GitHub\\output', 
#               ['raw_t1', 'raw_t2', 'raw_t3'],
#               ['node_index', 'vel', 'pos', 'sensed_density', 'strainRate'], 
#               raw_path)
# concatDataset('\\Users\\xuyan\\Documents\\GitHub\\output',
#                 ['raw_t1', 'raw_t2'],
#                 ['node_index', 'vel', 'pos', 'sensed_density', 'strainRate'], 
#                 raw_train_path)
# concatDataset('\\Users\\xuyan\\Documents\\GitHub\\output',
#                 ['raw_t3'],
#                 ['node_index', 'vel', 'pos', 'sensed_density', 'strainRate'], 
#                 raw_test_path)

# start_index = 0
# end_index = 950
# gridExport_density(raw_path, organized_path, start_index, end_index)
# gridExport_vel(raw_path, organized_path, start_index, end_index)
# gridExport_strainRate(raw_path, organized_path, start_index, end_index)
# process_strainRate_to(organized_path, organized_path, start_index, end_index, to='vorticity', use_density_mask=True)
# process_vel_to_strainRate(organized_path, organized_path, start_index, end_index, 7.0/258, True, further_to='vorticity')
# process_minus(organized_path, organized_path, start_index, end_index, 'vel2vorticity', 'strainRate2vorticity')

# start_index = 0
# end_index = 798
# gridExport_density(raw_train_path, organized_train_path, start_index, end_index)
# gridExport_vel(raw_train_path, organized_train_path, start_index, end_index)
# gridExport_strainRate(raw_train_path, organized_train_path, start_index, end_index)
# process_strainRate_to(organized_train_path, organized_train_path, start_index, end_index, to='vorticity', use_density_mask=True)
# process_vel_to_strainRate(organized_train_path, organized_train_path, start_index, end_index, 7.0/258, True, further_to='vorticity')
# process_minus(organized_train_path, organized_train_path, start_index, end_index, 'vel2vorticity', 'strainRate2vorticity')

# start_index = 0
# end_index = 185
# gridExport_density(raw_test_path, organized_test_path, start_index, end_index)
# gridExport_vel(raw_test_path, organized_test_path, start_index, end_index)
# gridExport_strainRate(raw_test_path, organized_test_path, start_index, end_index)
# process_strainRate_to(organized_test_path, organized_test_path, start_index, end_index, to='vorticity', use_density_mask=True)
# process_vel_to_strainRate(organized_test_path, organized_test_path, start_index, end_index, 7.0/258, True, further_to='vorticity')
# process_minus(organized_test_path, organized_test_path, start_index, end_index, 'vel2vorticity', 'strainRate2vorticity')

start_index = 0
end_index = 798
scivis_R2toR1(organized_train_path, organized_train_path, start_index, end_index, 'density')
scivis_R2toR1(organized_train_path, organized_train_path, start_index, end_index, 'strainRate2vorticity')
scivis_R2toR1(organized_train_path, organized_train_path, start_index, end_index, 'vel2vorticity')
scivis_R2toR1(organized_train_path, organized_train_path, start_index, end_index, 'vel2vorticityMINUSstrainRate2vorticity')
datavis_hist_R2toR1(organized_train_path, organized_train_path, start_index, end_index, 'strainRate2vorticity', -100, 100)
datavis_hist_R2toR1(organized_train_path, organized_train_path, start_index, end_index, 'vel2vorticity', -100, 100)
datavis_hist_R2toR1(organized_train_path, organized_train_path, start_index, end_index, 'vel2vorticityMINUSstrainRate2vorticity', -10, 10)

# process_abs(organized_path, organized_path, start_index, end_index, 'vel2vorticity')
# process_abs(organized_path, organized_path, start_index, end_index, 'strainRate2vorticity')
# process_abs(organized_path, organized_path, start_index, end_index, 'vel2vorticityMINUSstrainRate2vorticity')

# process_hist(organized_path, hist_path, start_index, end_index, 'abs_strainRate2vorticity', 0, 300)



# analyze_filenames_in_folder(raw_path, 'npy')
# analyze_filenames_in_folder(organized_path, 'npy')
# analyze_filenames_in_folder(sciVis_path, 'png')
# analyze_filenames_in_folder(dataVis_path, 'png')
# analyze_filenames_in_folder(hist_path, 'npy')



