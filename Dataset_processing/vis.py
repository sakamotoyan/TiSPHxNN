import matplotlib.pyplot as plt
import numpy as np
from matplotlib import image
import os

def augment(a, steepness = 5, mildrange = 0.2):
    if a < mildrange:
        linear_val = a
        root_val = a**(1/steepness) 
        blend_factor = 3*(a/mildrange)**2 - 2*(a/mildrange)**3
        return linear_val * (1 - blend_factor) + root_val * blend_factor
    else:
        return a**(1/steepness)
augment_vectorized = np.vectorize(augment)


def scivis_R2toR1(input_path, output_path, start_index, end_index, attr_name):
    data = []
    for i in range(start_index, end_index):
        data.append(np.load(os.path.join(input_path, f'{attr_name}_{i}.npy')))
    np_data = np.array(data)
    min_data = np_data.min()
    max_data = np_data.max()

    print(min_data, max_data)
    rgb = np.zeros((np_data.shape[1],np_data.shape[2],3))

    for i in range(end_index-start_index):
        val = np_data[i,...]
        rgb.fill(0)

        normalised_val_positive = augment_vectorized(np.clip(val, 0, None) / max_data)
        normalised_val_negative = augment_vectorized(np.clip(val, None, 0) / min_data)
        rgb[:,:,0] = np.where(val > 0, 1, 1 - normalised_val_negative)
        rgb[:,:,1] = 1 - np.where(val > 0, normalised_val_positive, normalised_val_negative)
        rgb[:,:,2] = np.where(val > 0, 1 - normalised_val_positive, 1)

        output_rgb = np.flip(np.transpose(rgb,(1,0,2)),0)
        image.imsave(os.path.join(output_path, f'sci_{attr_name}_{i}.png'), output_rgb)


def datavis_hist_R2toR1(input_path, output_path, start_index, end_index, attr_name, range_min, range_max, bins=128):
    data = []
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_yscale('log')
    ax.set_title('Histogram of Data')
    ax.set_xlabel('Vorticity value')
    ax.set_ylabel('Number of nodes')
    ax.grid(True)
    file_paths = [os.path.join(output_path, f'hist_{attr_name}_{i}.png') for i in range(start_index, end_index)]

    for i, file_path in zip(range(start_index, end_index), file_paths):
        
        data = np.load(os.path.join(input_path, f'{attr_name}_{i}.npy'))

        hist, bins = np.histogram(data, bins=bins, range=(range_min, range_max))
        ax.bar(bins[:-1], hist, width=np.diff(bins), edgecolor='black')

        # Save the figure
        fig.savefig(file_path)
        ax.clear()  # Clear the axes for the next iteration