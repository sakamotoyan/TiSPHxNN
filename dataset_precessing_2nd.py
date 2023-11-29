from ti_sph import *
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as colors

input_path = './output_organised/'
output_path = './output_organised/'

start_index = 0
end_index = 136

# dm_vel = Grid_Data(input_path, 'velocity', start_index, end_index)
# print(dm_vel.get_3x3_data(dm_vel.read_single_frame_data(11), 1, 1))
# print(Grid_Data.Soble_Y)

# print((Grid_Data.Soble_X * dm_vel.get_3x3_data(dm_vel.read_single_frame_data(11), 1, 1)[...,dm_vel.X_COMP]))

# print(dm_vel.Sobel_conv(dm_vel.read_single_frame_data(11)).shape)

def augment(a, steepness = 5, mildrange = 0.2):
    if a < mildrange:
        linear_val = a
        root_val = a**(1/steepness) 
        blend_factor = 3*(a/mildrange)**2 - 2*(a/mildrange)**3
        return linear_val * (1 - blend_factor) + root_val * blend_factor
    else:
        return a**(1/steepness)

def process_vel2strainRate(start_index, end_index, export_data=False, use_density_mask=False, vis_data='rotation'):
    data = []
    dm_vel = Grid_Data(input_path, 'velocity', start_index, end_index)
    
    for i in range(start_index, end_index):
        np_vel_data = dm_vel.read_single_frame_data(i)
        data.append(dm_vel.Sobel_conv(np_vel_data))
    
    if vis_data == 'rotation':
        x1, y1 = 1,0
        x2, y2 = 0,1
        measured_value = (np.array(data)[:,:,:,x1,y1] - np.array(data)[:,:,:,x2,y2])

    min_measured_value = measured_value.min()
    max_measured_value = measured_value.max()

    print(min_measured_value, max_measured_value)

    for i in range(end_index-start_index):
        # val = np.flipud(np.transpose(measured_value[i,...]))
        val = measured_value[i,...]
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

        plt.imsave(f'./output_organised/vel2strainRate_{vis_data}_{i}.png', rgb)
        plt.close()     

process_vel2strainRate(start_index, end_index)