from torch.utils.data import Dataset
from torch.utils.data import ConcatDataset
from settings import *
import numpy as np
import torch

### DATASET GENERATION
def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data) - num_forecast_steps  # subtract 1 to avoid index error

    def __getitem__(self, idx):
        sample_current = self.data[idx]  # current frame
        next_frames = []
        for i in range(num_forecast_steps):
            next_frames.append(self.data[idx + i + 1])  # next frame
        return sample_current, next_frames  # returns current frame and next frame

numpy_data_list_channel_1_density = []
numpy_data_list_channel_2n3_velocity = []

def get_dataset():

    for i in range(begin_index, end_index+1):
        numpy_data_channel_1_density = np.load(os.path.join(training_data_path,f'density_{i}.npy'))[:,0:res,0:res]
        numpy_data_channel_2n3_velocity = np.load(os.path.join(training_data_path,f'velocity_{i}.npy'))[:,0:res,0:res]
        numpy_data_list_channel_1_density.append(numpy_data_channel_1_density)
        numpy_data_list_channel_2n3_velocity.append(numpy_data_channel_2n3_velocity)

    stacked_data_channel_1_density = np.stack(numpy_data_list_channel_1_density)
    stacked_data_channel_2n3_velocity = np.stack(numpy_data_list_channel_2n3_velocity)
    stacked_data_channel_1_density = normalize_data(stacked_data_channel_1_density)
    stacked_data_channel_2n3_velocity = normalize_data(stacked_data_channel_2n3_velocity)

    print(stacked_data_channel_1_density.shape)
    print(stacked_data_channel_2n3_velocity.shape)
    stacked_data = np.concatenate([stacked_data_channel_1_density, stacked_data_channel_2n3_velocity], axis=1)
    tensor_data = torch.tensor(stacked_data, dtype=torch.float32)
    dataset = CustomDataset(tensor_data)

    return dataset

def get_dataset2():
    numpy_data_list_channel_1_density = []
    numpy_data_list_channel_2n3_velocity = []
    for i in range(0, 2000):
        numpy_data_channel_1_density = np.load(os.path.join(training_data_path,f'density_{i}.npy'))[:,0:res,0:res]
        numpy_data_channel_2n3_velocity = np.load(os.path.join(training_data_path,f'velocity_{i}.npy'))[:,0:res,0:res]
        numpy_data_list_channel_1_density.append(numpy_data_channel_1_density)
        numpy_data_list_channel_2n3_velocity.append(numpy_data_channel_2n3_velocity)

    stacked_data_channel_1_density = np.stack(numpy_data_list_channel_1_density)
    stacked_data_channel_2n3_velocity = np.stack(numpy_data_list_channel_2n3_velocity)
    stacked_data_channel_1_density = normalize_data(stacked_data_channel_1_density)
    stacked_data_channel_2n3_velocity = normalize_data(stacked_data_channel_2n3_velocity)

    print(stacked_data_channel_1_density.shape)
    print(stacked_data_channel_2n3_velocity.shape)
    stacked_data = np.concatenate([stacked_data_channel_1_density, stacked_data_channel_2n3_velocity], axis=1)
    tensor_data = torch.tensor(stacked_data, dtype=torch.float32)
    dataset_1 = CustomDataset(tensor_data)

    numpy_data_list_channel_1_density = []
    numpy_data_list_channel_2n3_velocity = []
    for i in range(2001, 4000):
        numpy_data_channel_1_density = np.load(os.path.join(training_data_path,f'density_{i}.npy'))[:,0:res,0:res]
        numpy_data_channel_2n3_velocity = np.load(os.path.join(training_data_path,f'velocity_{i}.npy'))[:,0:res,0:res]
        numpy_data_list_channel_1_density.append(numpy_data_channel_1_density)
        numpy_data_list_channel_2n3_velocity.append(numpy_data_channel_2n3_velocity)

    stacked_data_channel_1_density = np.stack(numpy_data_list_channel_1_density)
    stacked_data_channel_2n3_velocity = np.stack(numpy_data_list_channel_2n3_velocity)
    stacked_data_channel_1_density = normalize_data(stacked_data_channel_1_density)
    stacked_data_channel_2n3_velocity = normalize_data(stacked_data_channel_2n3_velocity)

    print(stacked_data_channel_1_density.shape)
    print(stacked_data_channel_2n3_velocity.shape)
    stacked_data = np.concatenate([stacked_data_channel_1_density, stacked_data_channel_2n3_velocity], axis=1)
    tensor_data = torch.tensor(stacked_data, dtype=torch.float32)
    dataset_2 = CustomDataset(tensor_data)

    return ConcatDataset([dataset_1, dataset_2])