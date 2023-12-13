import os
from torch.utils.data import Dataset
from torch.utils.data import ConcatDataset
import numpy as np
import torch

class DatasetConvAutoencoder_1(Dataset):
    def __init__(self, clipped_res, dataset_velocity, dataset_velocity_file_path, dataset_vorticity, dataset_vorticity_file_path, dataset_density, dataset_density_file_path):
        self.clipped_res = clipped_res
        self.dataset_velocity = dataset_velocity
        self.dataset_vorticity = dataset_vorticity
        self.dataset_density = dataset_density
        self.dataset_velocity_file_path = dataset_velocity_file_path
        self.dataset_vorticity_file_path = dataset_vorticity_file_path
        self.dataset_density_file_path = dataset_density_file_path

        self.len_velocity, self.start_number_velocity, self.end_number_velocity = self.check_files(self.dataset_velocity_file_path, self.dataset_velocity)
        self.len_vorticity, self.start_number_vorticity, self.end_number_vorticity = self.check_files(self.dataset_vorticity_file_path, self.dataset_vorticity)
        self.len_density, self.start_number_density, self.end_number_density = self.check_files(self.dataset_density_file_path, self.dataset_density)
        self.len = self.len_velocity
        self.start_idx = self.start_number_velocity

        if self.len_velocity != self.len_vorticity != self.len_density or \
            self.start_number_velocity != self.start_number_vorticity != self.start_number_density or \
            self.end_number_velocity != self.end_number_vorticity != self.end_number_density:
            raise Exception(f"Attributes {self.dataset_velocity} and {self.dataset_vorticity} have different number of files or different number ranges.\
                             {self.len_velocity}, {self.len_vorticity}, {self.start_number_velocity}, {self.start_number_vorticity}, {self.end_number_velocity}, {self.end_number_vorticity}")
        
        self.min_value_velocity_norm  = self.min_norm_value(self.dataset_velocity, self.dataset_velocity_file_path)
        self.max_value_velocity_norm  = self.max_norm_value(self.dataset_velocity, self.dataset_velocity_file_path)
        self.min_value_vorticity = self.min_value(self.dataset_vorticity, self.dataset_vorticity_file_path)
        self.max_value_vorticity = self.max_value(self.dataset_vorticity, self.dataset_vorticity_file_path)

    def __len__(self):
        return self.len  # subtract 1 to avoid index error

    def __getitem__(self, idx):
        np_velocity  = np.load(os.path.join(self.dataset_velocity_file_path, f'{self.dataset_velocity}_{idx+self.start_idx}.npy' ))[:self.clipped_res, :self.clipped_res ,:2]
        np_vorticity = np.load(os.path.join(self.dataset_vorticity_file_path,f'{self.dataset_vorticity}_{idx+self.start_idx}.npy'))[:self.clipped_res-2, :self.clipped_res-2]
        np_density   = np.load(os.path.join(self.dataset_density_file_path,  f'{self.dataset_density}_{idx+self.start_idx}.npy'  ))[:self.clipped_res  , :self.clipped_res  ]
        np_velocity_x = np_velocity[...,0]
        np_velocity_y = np_velocity[...,1]

        normalized_np_velocity_x = 2 * ((np_velocity_x - self.min_value_velocity_norm) / (self.max_value_velocity_norm - self.min_value_velocity_norm)) - 1
        normalized_np_velocity_y = 2 * ((np_velocity_y - self.min_value_velocity_norm) / (self.max_value_velocity_norm - self.min_value_velocity_norm)) - 1
        normalized_np_vorticity  = 2 * ((np_vorticity  - self.min_value_vorticity)     / (self.max_value_vorticity     - self.min_value_vorticity))     - 1

        normalized_np_velocity   = np.stack([normalized_np_velocity_x, normalized_np_velocity_y], axis=0)
        hist_np_vorticity, _     = np.histogram(normalized_np_vorticity, bins=128, range=(-1, 1))

        inpput = torch.tensor(normalized_np_velocity, dtype=torch.float32)
        target = torch.tensor(hist_np_vorticity , dtype=torch.int32)
        aux    = torch.tensor(np_density, dtype=torch.float32)

        return inpput, target, aux 

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
    
    def min_norm_value(self, attr_name, attr_file_path):
        min_value = None
        for i in range(self.start_idx, self.start_idx + self.len):
            data = np.load(os.path.join(attr_file_path,f'{attr_name}_{i}.npy'))
            norm = np.linalg.norm(data, axis=-1)
            if i == self.start_idx:
                min_value = np.min(norm)
            else:
                min_value = min(min_value, np.min(norm))
        print(f"min_norm_value(): min_value for {attr_name} is {min_value}")
        return min_value

    def min_comp_value(self, attr_name, attr_file_path, comp):
        min_value = None
        for i in range(self.start_idx, self.start_idx + self.len):
            data = np.load(os.path.join(attr_file_path,f'{attr_name}_{i}.npy'))[...,comp]
            if i == self.start_idx:
                min_value = np.min(data)
            else:
                min_value = min(min_value, np.min(data))
        print(f"min_comp_value(): min_value for comp {comp} of {attr_name} is {min_value}")
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
    
    def max_norm_value(self, attr_name, attr_file_path):
        max_value = None
        for i in range(self.start_idx, self.start_idx + self.len):
            data = np.load(os.path.join(attr_file_path,f'{attr_name}_{i}.npy'))
            norm = np.linalg.norm(data, axis=-1)
            if i == self.start_idx:
                max_value = np.max(norm)
            else:
                max_value = max(max_value, np.max(norm))
        print(f"max_norm_value(): max_value for {attr_name} is {max_value}")
        return max_value

    def max_comp_value(self, attr_name, attr_file_path, comp):
        max_value = None
        for i in range(self.start_idx, self.start_idx + self.len):
            data = np.load(os.path.join(attr_file_path,f'{attr_name}_{i}.npy'))[..., comp]
            if i == self.start_idx:
                max_value = np.max(data)
            else:
                max_value = max(max_value, np.max(data))
        print(f"max_comp_value(): max_value for comp {comp} of {attr_name} is {max_value}")
        return max_value

