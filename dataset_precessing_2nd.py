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

    ti_vorticity = ti.field(dtype=ti.f32, shape=(dm_strainRate.shape_x-2, dm_strainRate.shape_y-2))

    ti_density_data = ti.field(dtype=ti.f32, shape=(dm_density.shape_x, dm_density.shape_y))
    ti_density_grad_px = ti.field(dtype=ti.f32, shape=(dm_density.shape_x-2, dm_density.shape_y-2))
    ti_density_grad_py = ti.field(dtype=ti.f32, shape=(dm_density.shape_x-2, dm_density.shape_y-2))
    ti_density_grad_map = ti.field(dtype=ti.f32, shape=(dm_density.shape_x-2, dm_density.shape_y-2))
    ti_density_map = ti.static(ti_density_grad_px)

    for i in range(start_index, end_index):
        np_strainRate_data = dm_strainRate.read_single_frame_data(i)[1:-1,1:-1]
        if to == 'vorticity':
            np_vorticity_data = np_strainRate_data[...,1,0] - np_strainRate_data[...,0,1]
            ti_vorticity.from_numpy(np_vorticity_data)

            if use_density_mask:
                np_density_data = dm_density.read_single_frame_data(i)
                ti_density_data.from_numpy(np_density_data)
                dm_density.Sobel_conv_ti(1, ti_density_data, ti_density_grad_px, Grid_Data.PARTIAL_X)
                dm_density.Sobel_conv_ti(1, ti_density_data, ti_density_grad_py, Grid_Data.PARTIAL_Y)
                ker_entry_wise_grad_mag(ti_density_grad_px, ti_density_grad_py, ti_density_grad_map)

                ker_normalize(ti_density_grad_map)
                ker_invBinary(ti_density_grad_map, 0.2)        
                ker_entry_wise_productEqual(ti_density_grad_map, ti_vorticity)

                ti_density_map.from_numpy(np_density_data[1:-1,1:-1])
                ker_normalize(ti_density_map)
                ker_binary(ti_density_map, 0.5)
                ker_entry_wise_productEqual(ti_density_map, ti_vorticity)

            data.append(ti_vorticity.to_numpy())
    
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


def process_vel_to_strainRate(start_index, end_index, cell_size, use_density_mask=False, further_to='vorticity'):
    data = []
    dm_vel = Grid_Data(input_path, 'velocity', start_index, end_index)
    dm_density = Grid_Data(input_path, 'density', start_index, end_index)

    ti_density_data = ti.field(dtype=ti.f32, shape=(dm_density.shape_x, dm_density.shape_y))
    ti_density_grad_px = ti.field(dtype=ti.f32, shape=(dm_density.shape_x-2, dm_density.shape_y-2))
    ti_density_grad_py = ti.field(dtype=ti.f32, shape=(dm_density.shape_x-2, dm_density.shape_y-2))
    ti_density_grad_map = ti.field(dtype=ti.f32, shape=(dm_density.shape_x-2, dm_density.shape_y-2))
    ti_density_map = ti.static(ti_density_grad_px)
    
    ti_conved_val = ti.field(dtype=ti.f32, shape=(dm_vel.shape_x, dm_vel.shape_y))
    ti_conv_val_pupx = ti.field(dtype=ti.f32, shape=(dm_vel.shape_x-2, dm_vel.shape_y-2))
    ti_conv_val_pupy = ti.field(dtype=ti.f32, shape=(dm_vel.shape_x-2, dm_vel.shape_y-2))
    ti_conv_val_pvpx = ti.field(dtype=ti.f32, shape=(dm_vel.shape_x-2, dm_vel.shape_y-2))
    ti_conv_val_pvpy = ti.field(dtype=ti.f32, shape=(dm_vel.shape_x-2, dm_vel.shape_y-2))

    for i in range(start_index, end_index):
        np_vel_data = dm_vel.read_single_frame_data(i)

        np_conv_val = np.zeros((dm_vel.shape_x-2, dm_vel.shape_y-2,2,2))
        
        ti_conved_val.from_numpy(np_vel_data[...,Grid_Data.VAL_X])
        dm_vel.Sobel_conv_ti(cell_size, ti_conved_val, ti_conv_val_pupx, Grid_Data.PARTIAL_X)
        dm_vel.Sobel_conv_ti(cell_size, ti_conved_val, ti_conv_val_pupy, Grid_Data.PARTIAL_Y)
        ti_conved_val.from_numpy(np_vel_data[...,Grid_Data.VAL_Y])
        dm_vel.Sobel_conv_ti(cell_size, ti_conved_val, ti_conv_val_pvpx, Grid_Data.PARTIAL_X)
        dm_vel.Sobel_conv_ti(cell_size, ti_conved_val, ti_conv_val_pvpy, Grid_Data.PARTIAL_Y)

        if use_density_mask:
            np_density_data = dm_density.read_single_frame_data(i)
            ti_density_data.from_numpy(np_density_data)
            dm_density.Sobel_conv_ti(cell_size, ti_density_data, ti_density_grad_px, Grid_Data.PARTIAL_X)
            dm_density.Sobel_conv_ti(cell_size, ti_density_data, ti_density_grad_py, Grid_Data.PARTIAL_Y)
            ker_entry_wise_grad_mag(ti_density_grad_px, ti_density_grad_py, ti_density_grad_map)

            ker_normalize(ti_density_grad_map)
            ker_invBinary(ti_density_grad_map, 0.2)        
            ker_entry_wise_productEqual(ti_density_grad_map, ti_conv_val_pupx)
            ker_entry_wise_productEqual(ti_density_grad_map, ti_conv_val_pupy)
            ker_entry_wise_productEqual(ti_density_grad_map, ti_conv_val_pvpx)
            ker_entry_wise_productEqual(ti_density_grad_map, ti_conv_val_pvpy)

            ti_density_map.from_numpy(np_density_data[1:-1,1:-1])
            ker_normalize(ti_density_map)
            ker_binary(ti_density_map, 0.5)
            ker_entry_wise_productEqual(ti_density_map, ti_conv_val_pupx)
            ker_entry_wise_productEqual(ti_density_map, ti_conv_val_pupy)
            ker_entry_wise_productEqual(ti_density_map, ti_conv_val_pvpx)
            ker_entry_wise_productEqual(ti_density_map, ti_conv_val_pvpy)

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

def process_minus(attr_name_1, attr_name_2, start_index, end_index):
    data_1 = []
    data_2 = []
    for i in range(start_index, end_index):
        data_1.append(np.load(os.path.join(input_path, f'{attr_name_1}_{i}.npy')))
        data_2.append(np.load(os.path.join(input_path, f'{attr_name_2}_{i}.npy')))
    np_data_1 = np.array(data_1)
    np_data_2 = np.array(data_2)
    np_data = np_data_1 - np_data_2
    for i in range(start_index, end_index):
        np.save(os.path.join(output_path, f'{attr_name_1}_minus_{attr_name_2}_{i}.npy'), np_data[i-start_index])

def augment(a, steepness = 5, mildrange = 0.2):
    if a < mildrange:
        linear_val = a
        root_val = a**(1/steepness) 
        blend_factor = 3*(a/mildrange)**2 - 2*(a/mildrange)**3
        return linear_val * (1 - blend_factor) + root_val * blend_factor
    else:
        return a**(1/steepness)
    
def scivis_R2toR1(attr_name, start_index, end_index):
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
        plt.imsave(os.path.join(output_path, f'sci_{attr_name}_{i}.png'), output_rgb)
        plt.close()

def datavis_hist_R2toR1(attr_name, start_index, end_index, range_min, range_max, bins=128):
    data = []
    for i in range(start_index, end_index):
        data = np.load(os.path.join(input_path, f'{attr_name}_{i}.npy'))

        hist, bins = np.histogram(data, bins=128, range=(range_min, range_max))
        plt.figure(figsize=(10, 6))
        plt.ylim(0, data.shape[0]*data.shape[1])
        plt.bar(bins[:-1], hist, width=np.diff(bins), edgecolor='black')
        plt.title('Histogram of Data')
        plt.xlabel('Vorticity value')
        plt.ylabel('Number of nodes')
        plt.grid(True)
        plt.savefig(os.path.join(output_path, f'hist_{attr_name}_{i}.png'))
        plt.close()


# process_strainRate_to(start_index, end_index, to='vorticity', use_density_mask=True)
# process_vel_to_strainRate(start_index, end_index, 7.0/256, True, further_to='vorticity')

# process_minus('vel2vorticity', 'strainRate2vorticity', start_index, end_index)

# scivis_R2toR1('strainRate2vorticity', start_index, end_index)
# scivis_R2toR1('vel2vorticity', start_index, end_index)
# scivis_R2toR1('vel2vorticity_minus_strainRate2vorticity', start_index, end_index)

# datavis_hist_R2toR1('strainRate2vorticity', start_index, end_index, -300, 300)
# datavis_hist_R2toR1('vel2vorticity', start_index, end_index, -300, 300)
datavis_hist_R2toR1('vel2vorticity_minus_strainRate2vorticity', start_index, end_index, -10, 10)
