from ti_sph import *
import numpy as np
from PIL import Image

start_index = 0
end_index = 16

dm_density = Grid_Data_manager('./output', './output_organised')
dm_density.read_data(attr='sensed_density',start_index=start_index,end_index=end_index,channel_num=1)
dm_density.reshape_data_to_2d(index_attr='node_index')
data_density = dm_density.export_single_frame_data('density')

dm_vel = Grid_Data_manager('./output', './output_organised')
dm_vel.read_data(attr='vel',start_index=start_index,end_index=end_index,channel_num=2)
dm_vel.reshape_data_to_2d(index_attr='node_index')
# dm_vel.processed_data = dm_vel.processed_data*dm_density.processed_data
data_vel = dm_vel.export_single_frame_data('velocity')

dm_momentum = Grid_Data_manager('./output', './output_organised')
dm_momentum.read_data(attr='vel',start_index=start_index,end_index=end_index,channel_num=2)
dm_momentum.reshape_data_to_2d(index_attr='node_index')
dm_momentum.processed_data = dm_momentum.processed_data*dm_density.processed_data
data_momentum = dm_momentum.export_single_frame_data('momentum')

for i in range(end_index-start_index+1):
    frame_data = data_density[i,0]
    frame_data = np.flipud(np.transpose(frame_data))
    normalized_array = ((frame_data - frame_data.min()) * (255 - 0) / (frame_data.max() - frame_data.min())).astype(np.uint8)
    img = Image.fromarray(normalized_array, 'L')
    img.save(f'./output_organised/density_{i+start_index}.jpg')

for i in range(end_index-start_index+1):
    frame_data = data_vel[i,1]
    frame_data = np.flipud(np.transpose(frame_data))
    normalized_array = ((frame_data - frame_data.min()) * (255 - 0) / (frame_data.max() - frame_data.min())).astype(np.uint8)
    img = Image.fromarray(normalized_array, 'L')
    img.save(f'./output_organised/vel_y_{i+start_index}.jpg')

# output momentum image
for i in range(end_index-start_index+1):
    frame_data = data_momentum[i,1]
    frame_data = np.flipud(np.transpose(frame_data))
    normalized_array = ((frame_data - frame_data.min()) * (255 - 0) / (frame_data.max() - frame_data.min())).astype(np.uint8)
    img = Image.fromarray(normalized_array, 'L')
    img.save(f'./output_organised/momentum_y_{i+start_index}.jpg')