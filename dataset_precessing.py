from ti_sph import *
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as colors


start_index = 0
end_index = 1089
res = 128

input_path = './output/t1r'
output_path = './output_organised'

def augment(a, steepness = 5, mildrange = 0.2):
    if a < mildrange:
        linear_val = a
        root_val = a**(1/steepness) 
        blend_factor = 3*(a/mildrange)**2 - 2*(a/mildrange)**3
        return linear_val * (1 - blend_factor) + root_val * blend_factor
    else:
        return a**(1/steepness)

def plot_func(func, range_min, range_max, steps=1000):
    x = np.linspace(range_min, range_max, steps)
    y = np.zeros_like(x)
    for i in range(len(x)):
        y[i] = func(x[i])
    plt.plot(x,y)
    plt.show()

def process_density(export_data=False):
    dm_density = Grid_Data_manager(input_path, output_path)
    dm_density.read_data(attr='sensed_density',start_index=start_index,end_index=end_index)
    dm_density.reshape_data_to_2d(index_attr='node_index')
    if export_data:
        dm_density.export_single_frame_data('density', from_zero=True)

    for i in range(end_index-start_index):
        frame_data = dm_density.processed_data[i]
        frame_data = np.flipud(np.transpose(frame_data))
        normalized_array = ((frame_data - frame_data.min()) * (255 - 0) / (frame_data.max() - frame_data.min())).astype(np.uint8)
        img = Image.fromarray(normalized_array, 'L')
        img.save(f'./output_organised/density_{i}.png')

def process_vel(export_data=False):
    dm_vel = Grid_Data_manager(input_path, output_path)
    dm_vel.read_data(attr='vel',start_index=start_index,end_index=end_index)
    dm_vel.reshape_data_to_2d(index_attr='node_index')
    if export_data:
        dm_vel.export_single_frame_data('velocity', from_zero=True)

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
        plt.imsave(f'./output_organised/vel_hsv_{i}.png', rgb)
        plt.close()             # if from_zero, then i

def process_strainRate(export_data=False, use_density_mask=False, vis_data='compression', background_color='black'):
    dm_strainRate = Grid_Data_manager(input_path, output_path)
    dm_strainRate.read_data(attr='strainRate',start_index=start_index,end_index=end_index)
    dm_strainRate.reshape_data_to_2d(index_attr='node_index')
    if export_data:
        dm_strainRate.export_single_frame_data('strainRate', from_zero=True)

    dm_density = Grid_Data_manager(input_path, output_path)
    dm_density.read_data(attr='sensed_density',start_index=start_index,end_index=end_index)
    dm_density.reshape_data_to_2d(index_attr='node_index')
    max_density = np.array(dm_density.processed_data).max()

    if vis_data == 'compression':
        x1, y1 = 1,1
        x2, y2 = 0,0
        measured_value = (np.array(dm_strainRate.processed_data)[:,:,:,x1,y1] + np.array(dm_strainRate.processed_data)[:,:,:,x2,y2])
    elif vis_data == 'rotation':
        x1, y1 = 1,0
        x2, y2 = 0,1
        measured_value = (np.array(dm_strainRate.processed_data)[:,:,:,x1,y1] - np.array(dm_strainRate.processed_data)[:,:,:,x2,y2])
    elif vis_data == 'shear':
        x1, y1 = 1,0
        x2, y2 = 0,1
        measured_value = (np.array(dm_strainRate.processed_data)[:,:,:,x1,y1] + np.array(dm_strainRate.processed_data)[:,:,:,x2,y2])
    
    min_measured_value = measured_value.min()
    max_measured_value = measured_value.max()

    print(min_measured_value, max_measured_value)

    for i in range(end_index-start_index):
        val = np.flipud(np.transpose(measured_value[i,...]))
        if use_density_mask:
            density_mask = np.flipud(np.transpose(dm_density.processed_data[i]))/max_density
        else:
            density_mask = np.ones_like(val)

        rgb = np.zeros((val.shape[0],val.shape[1],3))

        for x in range(val.shape[0]):
            for y in range(val.shape[1]):
                if val[x,y] > 0:
                    normalised_val = augment(val[x,y]/max_measured_value)
                    rgb[x,y,0] = 1 * density_mask[x,y]
                    rgb[x,y,1] = (1 - normalised_val) * density_mask[x,y]
                    rgb[x,y,2] = (1 - normalised_val) * density_mask[x,y]
                else:
                    normalised_val = augment(val[x,y]/min_measured_value)
                    rgb[x,y,0] = (1 - normalised_val) * density_mask[x,y]
                    rgb[x,y,1] = (1 - normalised_val) * density_mask[x,y]
                    rgb[x,y,2] = 1 * density_mask[x,y]

        plt.imsave(f'./output_organised/strainRate_{vis_data}_{i}.png', rgb)
        plt.close()          

process_strainRate(export_data=False, use_density_mask=False, vis_data='shear')
process_strainRate(export_data=False, use_density_mask=False, vis_data='compression')
process_strainRate(export_data=False, use_density_mask=False, vis_data='rotation')
process_vel(export_data=False)
process_density(export_data=False)