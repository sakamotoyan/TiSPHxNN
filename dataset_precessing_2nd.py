from ti_sph import *
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import taichi as ti
import os

input_path = './output_organised/'
output_path = './output_organised/'

start_index = 0
end_index = 136

ti.init(arch=ti.gpu)

# dm_vel = Grid_Data(input_path, 'velocity', start_index, end_index)
# print(dm_vel.get_3x3_data(dm_vel.read_single_frame_data(11), 1, 1))
# print(Grid_Data.Soble_Y)

# print((Grid_Data.Soble_X * dm_vel.get_3x3_data(dm_vel.read_single_frame_data(11), 1, 1)[...,dm_vel.X_COMP]))

# print(dm_vel.Sobel_conv(dm_vel.read_single_frame_data(11)).shape)



def process_strainRate_to(start_index, end_index, attr_name='strainRate', to='vorticity', use_density_mask=False):
    data = []
    dm_strainRate = Grid_Data(input_path, attr_name, start_index, end_index)
    dm_density = Grid_Data(input_path, 'density', start_index, end_index)

    ti_conved_density_map = ti.field(dtype=ti.f32, shape=(dm_density.shape_x, dm_density.shape_y))
    ti_density_grad_px = ti.field(dtype=ti.f32, shape=(dm_density.shape_x-2, dm_density.shape_y-2))
    ti_density_grad_py = ti.field(dtype=ti.f32, shape=(dm_density.shape_x-2, dm_density.shape_y-2))
    ti_density_grad_mag = ti.field(dtype=ti.f32, shape=(dm_density.shape_x-2, dm_density.shape_y-2))

    for i in range(start_index, end_index):
        np_strainRate_data = dm_strainRate.read_single_frame_data(i)
        if to == 'vorticity':
            np_vorticity_data = np_strainRate_data[...,1,0] - np_strainRate_data[...,0,1]
            data.append(np_vorticity_data)
    
    for i in range(start_index, end_index):
        np.save(os.path.join(output_path, f'{attr_name}2{to}_{i}.npy'), data[i-start_index])
        
    # np_data = np.array(data)
    # min_data = np_data.min()
    # max_data = np_data.max()

    # print(min_data, max_data)

    # for i in range(end_index-start_index):
    #     val = np_data[i,...]
    #     density_mask = np.ones_like(val)

    #     rgb = np.zeros((val.shape[0],val.shape[1],3))

    #     for x in range(val.shape[0]):
    #         for y in range(val.shape[1]):
    #             if val[x,y] > 0:
    #                 normalised_val = augment(val[x,y]/max_data)
    #                 rgb[x,y,0] = 1 * density_mask[x,y]
    #                 rgb[x,y,1] = (1 - normalised_val) * density_mask[x,y]
    #                 rgb[x,y,2] = (1 - normalised_val) * density_mask[x,y]
    #             else:
    #                 normalised_val = augment(val[x,y]/min_data)
    #                 rgb[x,y,0] = (1 - normalised_val) * density_mask[x,y]
    #                 rgb[x,y,1] = (1 - normalised_val) * density_mask[x,y]
    #                 rgb[x,y,2] = 1 * density_mask[x,y]

    #     output_rgb = np.flip(np.transpose(rgb,(1,0,2)),0)
    #     plt.imsave(os.path.join(output_path, f'{attr_name}2{to}_{i}.png'), output_rgb)
    #     plt.close()


def process_vel_to_strainRate(start_index, end_index, use_density_mask=False, further_to='vorticity'):
    data = []
    dm_vel = Grid_Data(input_path, 'velocity', start_index, end_index)
    dm_density = Grid_Data(input_path, 'density', start_index, end_index)

    ti_conved_density_map = ti.field(dtype=ti.f32, shape=(dm_density.shape_x, dm_density.shape_y))
    ti_density_grad_px = ti.field(dtype=ti.f32, shape=(dm_density.shape_x-2, dm_density.shape_y-2))
    ti_density_grad_py = ti.field(dtype=ti.f32, shape=(dm_density.shape_x-2, dm_density.shape_y-2))
    ti_density_grad_mag = ti.field(dtype=ti.f32, shape=(dm_density.shape_x-2, dm_density.shape_y-2))
    
    ti_conved_val = ti.field(dtype=ti.f32, shape=(dm_vel.shape_x, dm_vel.shape_y))
    ti_conv_val_pupx = ti.field(dtype=ti.f32, shape=(dm_vel.shape_x-2, dm_vel.shape_y-2))
    ti_conv_val_pupy = ti.field(dtype=ti.f32, shape=(dm_vel.shape_x-2, dm_vel.shape_y-2))
    ti_conv_val_pvpx = ti.field(dtype=ti.f32, shape=(dm_vel.shape_x-2, dm_vel.shape_y-2))
    ti_conv_val_pvpy = ti.field(dtype=ti.f32, shape=(dm_vel.shape_x-2, dm_vel.shape_y-2))

    for i in range(start_index, end_index):
        np_vel_data = dm_vel.read_single_frame_data(i)

        np_conv_val = np.zeros((dm_vel.shape_x-2, dm_vel.shape_y-2,2,2))
        
        ti_conved_val.from_numpy(np_vel_data[...,Grid_Data.VAL_X])
        dm_vel.Sobel_conv_ti(ti_conved_val, ti_conv_val_pupx, Grid_Data.PARTIAL_X)
        dm_vel.Sobel_conv_ti(ti_conved_val, ti_conv_val_pupy, Grid_Data.PARTIAL_Y)
        ti_conved_val.from_numpy(np_vel_data[...,Grid_Data.VAL_Y])
        dm_vel.Sobel_conv_ti(ti_conved_val, ti_conv_val_pvpx, Grid_Data.PARTIAL_X)
        dm_vel.Sobel_conv_ti(ti_conved_val, ti_conv_val_pvpy, Grid_Data.PARTIAL_Y)

        if use_density_mask:
            np_density_data = dm_density.read_single_frame_data(i)
            ti_conved_density_map.from_numpy(np_density_data)
            dm_density.Sobel_conv_ti(ti_conved_density_map, ti_density_grad_px, Grid_Data.PARTIAL_X)
            dm_density.Sobel_conv_ti(ti_conved_density_map, ti_density_grad_py, Grid_Data.PARTIAL_Y)
            ker_entry_wise_grad_mag(ti_density_grad_px, ti_density_grad_py, ti_density_grad_mag)

            ker_normalize(ti_density_grad_mag)
            ker_invBinary(ti_density_grad_mag, 0.2)        
            ker_entry_wise_productEqual(ti_density_grad_mag, ti_conv_val_pupx)
            ker_entry_wise_productEqual(ti_density_grad_mag, ti_conv_val_pupy)
            ker_entry_wise_productEqual(ti_density_grad_mag, ti_conv_val_pvpx)
            ker_entry_wise_productEqual(ti_density_grad_mag, ti_conv_val_pvpy)

            ti_density_grad_px.from_numpy(np_density_data[1:-1,1:-1])
            ker_normalize(ti_density_grad_px)
            ker_binary(ti_density_grad_px, 0.5)
            ker_entry_wise_productEqual(ti_density_grad_px, ti_conv_val_pupx)
            ker_entry_wise_productEqual(ti_density_grad_px, ti_conv_val_pupy)
            ker_entry_wise_productEqual(ti_density_grad_px, ti_conv_val_pvpx)
            ker_entry_wise_productEqual(ti_density_grad_px, ti_conv_val_pvpy)

        np_conv_val[...,Grid_Data.VAL_X,Grid_Data.PARTIAL_X] = ti_conv_val_pupx.to_numpy()
        np_conv_val[...,Grid_Data.VAL_X,Grid_Data.PARTIAL_Y] = ti_conv_val_pupy.to_numpy()
        np_conv_val[...,Grid_Data.VAL_Y,Grid_Data.PARTIAL_X] = ti_conv_val_pvpx.to_numpy()
        np_conv_val[...,Grid_Data.VAL_Y,Grid_Data.PARTIAL_Y] = ti_conv_val_pvpy.to_numpy()

        data.append(np_conv_val)

    for i in range(start_index, end_index):
        np.save(os.path.join(output_path, f'vel2strainRate_{i}.npy'), data[i-start_index])
    
    if further_to == 'vorticity':
        x1, y1 = 1,0
        x2, y2 = 0,1
        measured_value = (np.array(data)[:,:,:,x1,y1] - np.array(data)[:,:,:,x2,y2])
        for i in range(start_index, end_index):
            np.save(os.path.join(output_path, f'vel2{further_to}_{i}.npy'), measured_value[i-start_index])

    # min_measured_value = measured_value.min()
    # max_measured_value = measured_value.max()

    # print(min_measured_value, max_measured_value)
    
    # for i in range(end_index-start_index):
    #     val = measured_value[i,...]
    #     density_mask = np.ones_like(val)

    #     rgb = np.zeros((val.shape[0],val.shape[1],3))

    #     for x in range(val.shape[0]):
    #         for y in range(val.shape[1]):
    #             if val[x,y] > 0:
    #                 normalised_val = augment(val[x,y]/max_measured_value)
    #                 rgb[x,y,0] = 1 * density_mask[x,y]
    #                 rgb[x,y,1] = (1 - normalised_val) * density_mask[x,y]
    #                 rgb[x,y,2] = (1 - normalised_val) * density_mask[x,y]
    #             else:
    #                 normalised_val = augment(val[x,y]/min_measured_value)
    #                 rgb[x,y,0] = (1 - normalised_val) * density_mask[x,y]
    #                 rgb[x,y,1] = (1 - normalised_val) * density_mask[x,y]
    #                 rgb[x,y,2] = 1 * density_mask[x,y]

    #     output_rgb = np.flip(np.transpose(rgb,(1,0,2)),0)
    #     plt.imsave(f'./output_organised/vel2strainRate_{vis_data}_{i}.png', output_rgb)
    #     plt.close()     


def augment(a, steepness = 5, mildrange = 0.2):
    if a < mildrange:
        linear_val = a
        root_val = a**(1/steepness) 
        blend_factor = 3*(a/mildrange)**2 - 2*(a/mildrange)**3
        return linear_val * (1 - blend_factor) + root_val * blend_factor
    else:
        return a**(1/steepness)
    
def vis1d(attr_name, start_index, end_index):
    data = []
    for i in range(start_index, end_index):
        data.append(np.load(os.path.join(input_path, f'{attr_name}_{i}.npy')))
    np_data = np.array(data)
    min_data = np_data.min()
    max_data = np_data.max()

    print(min_data, max_data)

    for i in range(end_index-start_index):
        val = np_data[i,...]
        density_mask = np.ones_like(val)

        rgb = np.zeros((val.shape[0],val.shape[1],3))

        for x in range(val.shape[0]):
            for y in range(val.shape[1]):
                if val[x,y] > 0:
                    normalised_val = augment(val[x,y]/max_data)
                    rgb[x,y,0] = 1 * density_mask[x,y]
                    rgb[x,y,1] = (1 - normalised_val) * density_mask[x,y]
                    rgb[x,y,2] = (1 - normalised_val) * density_mask[x,y]
                else:
                    normalised_val = augment(val[x,y]/min_data)
                    rgb[x,y,0] = (1 - normalised_val) * density_mask[x,y]
                    rgb[x,y,1] = (1 - normalised_val) * density_mask[x,y]
                    rgb[x,y,2] = 1 * density_mask[x,y]

        output_rgb = np.flip(np.transpose(rgb,(1,0,2)),0)
        plt.imsave(os.path.join(output_path, f'{attr_name}_{i}.png'), output_rgb)
        plt.close()


process_strainRate_to(start_index, end_index, to='vorticity', use_density_mask=False)
process_vel_to_strainRate(start_index, end_index, False, further_to='vorticity')
vis1d('vel2vorticity', start_index, end_index)
vis1d('strainRate2vorticity', start_index, end_index)
vis1d('density', start_index, end_index)