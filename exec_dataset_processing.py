from Dataset_processing import *

raw_path = './output/raw/'
organized_path = './output/organized/'
sciVis_path = './output/sciVis/'
dataVis_path = './output/dataVis/'
hist_path = './output/hist/'

start_index = 0
end_index = 136
res = 256

ti.init(arch=ti.gpu)

# clear_folder(organized_path)
# clear_folder(sciVis_path)
# clear_folder(dataVis_path)

# gridExport_density(raw_path, organized_path, start_index, end_index)
# gridExport_vel(raw_path, organized_path, start_index, end_index)
# gridExport_strainRate(raw_path, organized_path, start_index, end_index)
# process_strainRate_to(organized_path, organized_path, start_index, end_index, to='vorticity', use_density_mask=True)
# process_vel_to_strainRate(organized_path, organized_path, start_index, end_index, 7.0/256, True, further_to='vorticity')
# process_minus(organized_path, organized_path, start_index, end_index, 'vel2vorticity', 'strainRate2vorticity')

# process_abs(organized_path, organized_path, start_index, end_index, 'vel2vorticity')
# process_abs(organized_path, organized_path, start_index, end_index, 'strainRate2vorticity')
# process_abs(organized_path, organized_path, start_index, end_index, 'vel2vorticityMINUSstrainRate2vorticity')

# scivis_R2toR1(organized_path, sciVis_path, start_index, end_index, 'density')
# scivis_R2toR1(organized_path, sciVis_path, start_index, end_index, 'strainRate2vorticity')
# scivis_R2toR1(organized_path, sciVis_path, start_index, end_index, 'vel2vorticity')
# scivis_R2toR1(organized_path, sciVis_path, start_index, end_index, 'vel2vorticityMINUSstrainRate2vorticity')



datavis_hist_R2toR1(organized_path, dataVis_path, start_index, end_index, 'abs_strainRate2vorticity', -100, 100)
# datavis_hist_R2toR1(organized_path, dataVis_path, start_index, end_index, 'vel2vorticity', -100, 100)
# datavis_hist_R2toR1(organized_path, dataVis_path, start_index, end_index, 'vel2vorticityMINUSstrainRate2vorticity', -10, 10)

process_hist(organized_path, hist_path, start_index, end_index, 'abs_strainRate2vorticity', 0, 300)

# analyze_filenames_in_folder(raw_path, 'npy')
# analyze_filenames_in_folder(organized_path, 'npy')
# analyze_filenames_in_folder(sciVis_path, 'png')
# analyze_filenames_in_folder(dataVis_path, 'png')
# analyze_filenames_in_folder(hist_path, 'npy')
