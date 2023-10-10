from ti_sph import *
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as colors


start_index = 0
end_index = 30
res = 128

dm_density = Grid_Data_manager('./output', './output_organised')
dm_density.read_data(attr='sensed_density',start_index=start_index,end_index=end_index)
dm_density.reshape_data_to_2d(index_attr='node_index')
data_density = dm_density.export_single_frame_data('density', from_zero=True)

dm_vel = Grid_Data_manager('./output', './output_organised')
dm_vel.read_data(attr='vel',start_index=start_index,end_index=end_index)
dm_vel.reshape_data_to_2d(index_attr='node_index')
dm_vel.export_single_frame_data('velocity', from_zero=True)

dm_strainRate = Grid_Data_manager('./output', './output_organised')
dm_strainRate.read_data(attr='strainRate',start_index=start_index,end_index=end_index)
dm_strainRate.reshape_data_to_2d(index_attr='node_index')
dm_strainRate.export_single_frame_data('strainRate', from_zero=True)

for i in range(end_index-start_index):
    frame_data = data_density[i]
    frame_data = np.flipud(np.transpose(frame_data))
    normalized_array = ((frame_data - frame_data.min()) * (255 - 0) / (frame_data.max() - frame_data.min())).astype(np.uint8)
    img = Image.fromarray(normalized_array, 'L')
    img.save(f'./output_organised/density_{i}.jpg') 

# HSV visualization of velocity field
# dm_vel.processed_data.shape = (frame_num, channel_num(2, for x and y each), vel_x, vel_y)
g_speed = np.sqrt(np.array(dm_vel.processed_data)[:,:,:,0]**2 + np.array(dm_vel.processed_data)[:,:,:,1]**2)
g_speed_min = g_speed.min()
g_speed_max = g_speed.max()
for i in range(end_index-start_index):
    # Step 0
    # Get the velocity vector at each frame
    v = dm_vel.processed_data[i]
    v_x=np.flipud(np.transpose(v[:,:,0]))
    v_y=np.flipud(np.transpose(v[:,:,1]))
    # Step 1
    # Calculate speed and direction (angle) from x and y components of the velocity
    speed = np.sqrt(v_x**2 + v_y**2)
    angle = (np.arctan2(v_x, v_y) + np.pi) / (2. * np.pi)
    # Step 2
    # Create HSV representation
    hsv = np.zeros((v.shape[0],v.shape[1],3))
    hsv[..., 0] = angle
    hsv[..., 1] = 1.0  # Set saturation to maximum
    # hsv[..., 2] = (speed - speed.min()) / (speed.max() - speed.min())  # SINGLE_FRAME Normalize speed to range [0,1]
    hsv[..., 2] = (speed - g_speed_min) / (g_speed_max - g_speed_min)  # GLOBAL Normalize speed to range [0,1]
    # Step 3
    # Convert HSV to RGB
    rgb = colors.hsv_to_rgb(hsv)
    # plt.imsave(f'./output_organised/vel_hsv_{i+start_index}.jpg', rgb) # if not from_zero, then i+start_index
    plt.imsave(f'./output_organised/vel_hsv_{i}.png', rgb)             # if from_zero, then i


# HSV visualization of velocity field
# dm_vel.processed_data.shape = (frame_num, channel_num(2, for x and y each), vel_x, vel_y)
x1, y1 = 0,0
x2, y2 = 0,0
g_strainRate = (np.array(dm_strainRate.processed_data)[:,:,:,x1,y1] + np.array(dm_strainRate.processed_data)[:,:,:,x2,y2])
g_speed_min = g_strainRate.min()
g_speed_max = g_strainRate.max()
print(g_speed_min, g_speed_max)
for i in range(end_index-start_index):
    v = np.array(dm_strainRate.processed_data)[i,:,:,:,:]

    v_x=np.flipud(np.transpose(v[:,:,x1,y1]))
    v_y=np.flipud(np.transpose(v[:,:,x2,y2]))
    v_val = v_x + v_y

    length = np.abs(v_val).max() - np.abs(v_val).min()
    normalized_array = ((np.abs(v_val)-np.abs(v_val).min()) * (255 - 0) / length).astype(np.uint8)
    img = Image.fromarray(normalized_array, 'L')
    img.save(f'./output_organised/strainRate_x_{i}.png')

# for i in range(end_index-start_index):
#     # Step 0
#     # Get the velocity vector at each frame
#     v = np.array(dm_strainRate.processed_data)[i,:,:,:,:]
#     v_x=np.flipud(np.transpose(v[:,:,0,0]))
#     v_y=np.flipud(np.transpose(v[:,:,1,1]))
#     # Step 1
#     # Calculate speed and direction (angle) from x and y components of the velocity
#     speed = np.sqrt(v_x**2 + v_y**2)
#     angle = (np.arctan2(v_x, v_y) + np.pi) / (2. * np.pi)
#     # Step 2
#     # Create HSV representation
#     hsv = np.zeros((v.shape[0],v.shape[1],3))
#     hsv[..., 0] = angle
#     hsv[..., 1] = 1.0  # Set saturation to maximum
#     # hsv[..., 2] = (speed - speed.min()) / (speed.max() - speed.min())  # SINGLE_FRAME Normalize speed to range [0,1]
#     hsv[..., 2] = (speed - g_speed_min) / (g_speed_max - g_speed_min)  # GLOBAL Normalize speed to range [0,1]
#     # Step 3
#     # Convert HSV to RGB
#     rgb = colors.hsv_to_rgb(hsv)
#     # plt.imsave(f'./output_organised/vel_hsv_{i+start_index}.jpg', rgb) # if not from_zero, then i+start_index
#     plt.imsave(f'./output_organised/strainRate_x_{i}.png', rgb)             # if from_zero, then i