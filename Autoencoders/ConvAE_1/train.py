import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from .network import ConvAutoencoder_1
from .dataset import DatasetConvAutoencoder_1

class TrainConvAutoencoder_1:
    def __init__(self, feature_vector_size, res, attr_name_1, dataset_file_path_1, attr_name_2, dataset_file_path_2, platform='cuda'):
        self.platform = platform
        self.network = ConvAutoencoder_1(feature_vector_size)
        self.network.to(platform)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.network.parameters(), lr=0.001)
        self.dataset = DatasetConvAutoencoder_1(res, attr_name_1, dataset_file_path_1, attr_name_2, dataset_file_path_2)
        self.data_loader = DataLoader(self.dataset, batch_size=32, shuffle=True)

    
    def train(self, num_epochs):
        for epoch in range(num_epochs):

            for batch_inputs, batch_targets in self.data_loader:
                loss = 0.0
                self.optimizer.zero_grad()
                
                inputs = batch_inputs.to(self.platform) 
                outputs = self.network(inputs)
                print(outputs.shape)
    
