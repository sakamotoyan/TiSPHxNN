from ti_sph import *
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as colors


start_index = 0
end_index = 10

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

# HSV visualization of velocity field
g_speed = np.sqrt(dm_vel.processed_data[:,0]**2 + dm_vel.processed_data[:,1]**2)
g_speed_min = g_speed.min()
g_speed_max = g_speed.max()
for i in range(end_index-start_index+1):
    # Step 0
    # Get the velocity vector at each frame
    v = dm_vel.processed_data[i]
    v[0]=np.flipud(np.transpose(v[0]))
    v[1]=np.flipud(np.transpose(v[1]))
    # Step 1
    # Calculate speed and direction (angle) from x and y components of the velocity
    speed = np.sqrt(v[0]**2 + v[1]**2)
    angle = (np.arctan2(v[1], v[0]) + np.pi) / (2. * np.pi)
    # Step 2
    # Create HSV representation
    hsv = np.zeros((v.shape[1],v.shape[2],3))
    hsv[..., 0] = angle
    hsv[..., 1] = 1.0  # Set saturation to maximum
    # hsv[..., 2] = (speed - speed.min()) / (speed.max() - speed.min())  # SINGLE_FRAME Normalize speed to range [0,1]
    hsv[..., 2] = (speed - g_speed_min) / (g_speed_max - g_speed_min)  # GLOBAL Normalize speed to range [0,1]
    # Step 3
    # Convert HSV to RGB
    rgb = colors.hsv_to_rgb(hsv)
    plt.imsave(f'./output_organised/vel_hsv_{i+start_index}.jpg', rgb)
