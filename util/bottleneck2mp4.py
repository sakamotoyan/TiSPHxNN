import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def inspect(path, attr_name):
    files = os.listdir(path)
    relevant_files = [f for f in files if f.startswith(attr_name) and f.endswith(".npy")]
    indices = [int(f.split('_')[-1].split('.')[0]) for f in relevant_files]
    indices.sort()
    start_number = min(indices) if indices else None
    end_number = max(indices) if indices else None
    return len(relevant_files)

def load_arrays(path, attr_name, n_frames):
    frames = []
    for i in range(n_frames):
        # Assuming the files are named 'frame0.npy', 'frame1.npy', ..., 'frame127.npy'
        filename = os.path.join(path, f'{attr_name}_{i}.npy')
        frame = np.load(filename)
        frames.append(frame)
    return frames

output_path = '../output'
input_path = output_path
attr_name = 'bottleneck'

np_arrays = []
n_frames = inspect(input_path, attr_name)

for i in range(n_frames):
    np_arrays.append(np.load(os.path.join(input_path,f'{attr_name}_{i}.npy')).squeeze())

np_arrays = np.array(np_arrays)

global_min = np_arrays.min()
global_max = np_arrays.max()
abs_global_max = max(abs(global_min), abs(global_max))
print(global_min, global_max)


positive_mask = np_arrays > 0
negative_mask = np_arrays < 0
np_arrays[positive_mask] = np_arrays[positive_mask] / np_arrays[positive_mask].max()
np_arrays[negative_mask] = np_arrays[negative_mask] / np.abs(np_arrays[negative_mask]).max()
np_normalized_arrays = np_arrays

for i in range(np_normalized_arrays.shape[0]):
    plt.figure(figsize=(10, 6))  # Set the figure size as desired
    # bar chart
    plt.bar(np.arange(len(np_normalized_arrays[i])), np_normalized_arrays[i])
    # plt.plot(np_normalized_arrays[i])  # Plot the frame data; adjust indexing as per your data structure
    plt.xlabel('Index')
    plt.ylabel('Normalized Value')
    plt.title(f'Frame {i + 1}')
    plt.ylim([-1, 1])
    plt.savefig(os.path.join(output_path, f'frame_bottleneck_{i + 1}.png'), dpi=200)  # Save the figure as PNG
    plt.close()  # Close the figure to free memory