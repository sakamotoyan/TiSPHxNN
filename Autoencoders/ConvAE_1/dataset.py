import os
from torch.utils.data import Dataset
from torch.utils.data import ConcatDataset
import numpy as np
import torch

class DatasetConvAutoencoder_1(Dataset):
    def __init__(self, attr_name_1, dataset_file_path_1, attr_name_2, dataset_file_path_2):
        self.dataset_velocity = attr_name_1
        self.dataset_vorticity = attr_name_2
        self.dataset_velocity_file_path = dataset_file_path_1
        self.dataset_vorticity_file_path = dataset_file_path_2

        self.len_velocity, self.start_number_velocity, self.end_number_velocity = self.check_files(self.dataset_velocity_file_path, self.dataset_velocity)
        self.len_vorticity, self.start_number_vorticity, self.end_number_vorticity = self.check_files(self.dataset_vorticity_file_path, self.dataset_vorticity)
        self.len = self.len_velocity
        self.start_idx = self.start_number_velocity

        if self.len_velocity != self.len_vorticity or self.start_number_velocity != self.start_number_vorticity or self.end_number_velocity != self.end_number_vorticity:
            raise Exception(f"Attributes {self.dataset_velocity} and {self.dataset_vorticity} have different number of files or different number ranges.\
                             {self.len_velocity}, {self.len_vorticity}, {self.start_number_velocity}, {self.start_number_vorticity}, {self.end_number_velocity}, {self.end_number_vorticity}")
        
        self.min_value_velocity = self.min_value(self.dataset_velocity, self.dataset_velocity_file_path)
        self.max_value_velocity = self.max_value(self.dataset_velocity, self.dataset_velocity_file_path)
        self.min_value_vorticity = self.min_value(self.dataset_vorticity, self.dataset_vorticity_file_path)
        self.max_value_vorticity = self.max_value(self.dataset_vorticity, self.dataset_vorticity_file_path)


    def check_files(self, folder_path, attr_name):
        # List all files in the directory
        files = os.listdir(folder_path)

        # Filter out the relevant files
        relevant_files = [f for f in files if f.startswith(attr_name) and f.endswith(".npy")]

        # Extract indices and sort them
        indices = [int(f.split('_')[-1].split('.')[0]) for f in relevant_files]
        indices.sort()
        start_number = min(indices) if indices else None
        end_number = max(indices) if indices else None

        # Check for missing files
        missing_files = []
        for i in range(max(indices) + 1):
            if i not in indices:
                missing_files.append(f"{attr_name}_{i}.npy")

        print(f"check_files(): In {folder_path}, found {len(relevant_files)} files for attribute {attr_name}, ranging [{start_number}, {end_number}].")
        if len(missing_files) > 0:
            raise Exception(f"Missing files for attribute {attr_name}: {missing_files}")     

        return len(relevant_files), start_number, end_number
    
    def min_value(self, attr_name, attr_file_path):
        min_value = None
        for i in range(self.start_idx, self.start_idx + self.len):
            data = np.load(os.path.join(attr_file_path,f'{attr_name}_{i}.npy'))
            if i == self.start_idx:
                min_value = np.min(data)
            else:
                min_value = min(min_value, np.min(data))
        print(f"min_value(): min_value for {attr_name} is {min_value}")
        return min_value

    def max_value(self, attr_name, attr_file_path):
        max_value = None
        for i in range(self.start_idx, self.start_idx + self.len):
            data = np.load(os.path.join(attr_file_path,f'{attr_name}_{i}.npy'))
            if i == self.start_idx:
                max_value = np.max(data)
            else:
                max_value = max(max_value, np.max(data))
        print(f"max_value(): max_value for {attr_name} is {max_value}")
        return max_value

    def __len__(self):
        return self.len  # subtract 1 to avoid index error

    def __getitem__(self, idx):
        np_velocity = np.load(os.path.join(self.dataset_velocity_file_path,f'{self.dataset_velocity}_{idx+self.start_idx}.npy'))
        velocity = torch.tensor(2*((np_velocity)), dtype=torch.float32)
        vorticity = torch.tensor(np.load(os.path.join(self.dataset_vorticity_file_path,f'{self.dataset_vorticity}_{idx+self.start_idx}.npy')), dtype=torch.float32)
        return velocity, vorticity