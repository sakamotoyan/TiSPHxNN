import torch
import torch.nn as nn
import torch.optim as optim
from torchview import draw_graph
from torch.utils.data import DataLoader

import os
import taichi as ti
import numpy as np

from .network import ConvAutoencoder_1
from .dataset import DatasetConvAutoencoder_1

@ti.data_oriented
class TrainConvAutoencoder_1:

    TI_Soble_X = ti.Matrix([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])

    TI_Soble_Y = ti.Matrix([[-1, -2, -1],
                            [ 0,  0,  0],
                            [ 1,  2,  1]])
    
    def __init__(self, feature_vector_size, res, attr_name_1, dataset_file_path_1, 
                                                 attr_name_2, dataset_file_path_2, 
                                                 attr_name_3, dataset_file_path_3,
                                                 platform='cuda'):
        

        self.batch_size = 64
        self.platform = platform
        self.network = ConvAutoencoder_1(feature_vector_size)
        self.network.to(platform)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.network.parameters(), lr=0.001)
        self.dataset = DatasetConvAutoencoder_1(res, attr_name_1, dataset_file_path_1, attr_name_2, dataset_file_path_2, attr_name_3, dataset_file_path_3, self.platform)
        self.data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    
    def train_histBased(self, num_epochs, network_model_path, former_model_file_path=None):
        current_epochs_num = 0
        if former_model_file_path is not None:
            self.network.load_state_dict(torch.load(former_model_file_path))
            current_epochs_num = int(former_model_file_path.rsplit('_', 1)[1].split('.')[0]) + 1
            print(f"Loaded former model from {former_model_file_path}")
        for epoch in range(num_epochs):
            running_loss = 0.0
            for batch_inputs, batch_targets, batch_aux in self.data_loader:
                loss = 0.0
                self.optimizer.zero_grad()
                
                inputs  = (batch_inputs + 1) / 2
                batch_outputs = (self.network(inputs)) * 2 - 1

                input_dv_dx = torch.diff(batch_inputs[:, 1, :, :], dim=1, prepend=batch_inputs[:, 1, :, -1].unsqueeze(1))
                input_du_dy = torch.diff(batch_inputs[:, 0, :, :], dim=2, prepend=batch_inputs[:, 0, :, -1].unsqueeze(2))
                input_vorticity = (input_dv_dx - input_du_dy)
                input_vorticity_hist = self.differentiable_histogram(input_vorticity, bins=128, min_value=-1, max_value=1)

                output_dv_dx = torch.diff(batch_outputs[:, 1, :, :], dim=1, prepend=batch_outputs[:, 1, :, -1].unsqueeze(1))
                output_du_dy = torch.diff(batch_outputs[:, 0, :, :], dim=2, prepend=batch_outputs[:, 0, :, -1].unsqueeze(2))
                output_vorticity = (output_dv_dx - output_du_dy)
                output_vorticity_hist = self.differentiable_histogram(output_vorticity, bins=128, min_value=-1, max_value=1)

                loss = self.criterion(output_vorticity_hist, input_vorticity_hist)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            average_loss = running_loss / len(self.data_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.8f}")
            torch.save(self.network.state_dict(), os.path.join(network_model_path,f'epochs_{epoch+current_epochs_num}.pth'))

    def train_vorticityBased(self, num_epochs, network_model_path, former_model_file_path=None):
        current_epochs_num = 0
        if former_model_file_path is not None:
            self.network.load_state_dict(torch.load(former_model_file_path))
            current_epochs_num = int(former_model_file_path.rsplit('_', 1)[1].split('.')[0]) + 1
            print(f"Loaded former model from {former_model_file_path}")
        for epoch in range(num_epochs):
            running_loss = 0.0
            for batch_inputs, batch_targets, batch_aux in self.data_loader:
                loss = 0.0
                self.optimizer.zero_grad()
                
                inputs  = (batch_inputs + 1) / 2
                batch_outputs = (self.network(inputs)) * 2 - 1

                input_dv_dx = torch.diff(batch_inputs[:, 1, :, :], dim=1, prepend=batch_inputs[:, 1, :, -1].unsqueeze(1))
                input_du_dy = torch.diff(batch_inputs[:, 0, :, :], dim=2, prepend=batch_inputs[:, 0, :, -1].unsqueeze(2))
                input_vorticity = (input_dv_dx - input_du_dy)

                output_dv_dx = torch.diff(batch_outputs[:, 1, :, :], dim=1, prepend=batch_outputs[:, 1, :, -1].unsqueeze(1))
                output_du_dy = torch.diff(batch_outputs[:, 0, :, :], dim=2, prepend=batch_outputs[:, 0, :, -1].unsqueeze(2))
                output_vorticity = (output_dv_dx - output_du_dy)

                loss = self.criterion(output_vorticity, input_vorticity)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            average_loss = running_loss / len(self.data_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.8f}")
            if (epoch+1) % 10 == 0:
                torch.save(self.network.state_dict(), os.path.join(network_model_path,f'epochs_{epoch+current_epochs_num}.pth'))

    def train_velocityBased(self, num_epochs, network_model_path, former_model_file_path=None):
        current_epochs_num = 0
        
        if former_model_file_path is not None:
            self.network.load_state_dict(torch.load(former_model_file_path))
            current_epochs_num = int(former_model_file_path.rsplit('_', 1)[1].split('.')[0]) + 1
            print(f"Loaded former model from {former_model_file_path}")

        for epoch in range(num_epochs):
            running_loss = 0.0
            for batch_inputs, _, _ in self.data_loader:
                loss = 0.0
                self.optimizer.zero_grad()
                
                inputs  = (batch_inputs + 1) / 2
                batch_outputs = (self.network(inputs)) * 2 - 1

                loss = self.criterion(batch_outputs, batch_inputs)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            average_loss = running_loss / len(self.data_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.8f}")
            if (epoch+1) % 10 == 0:
                torch.save(self.network.state_dict(), os.path.join(network_model_path,f'epochs_{epoch+current_epochs_num}.pth'))

    def change_dataset(self, attr_name_1, dataset_file_path_1, attr_name_2, dataset_file_path_2, attr_name_3, dataset_file_path_3):
        self.dataset = DatasetConvAutoencoder_1(attr_name_1, dataset_file_path_1, attr_name_2, dataset_file_path_2, attr_name_3, dataset_file_path_3)
        self.data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def draw_graph(self, model_file_path):
        model_graph = draw_graph(self.network, input_size=(self.batch_size, 2, 256, 256), device=self.platform, directory=model_file_path)
        model_graph.visual_graph.render()

    def test(self, model_file_path, output_path):
            
        self.network.load_state_dict(torch.load(model_file_path))
        print(f"Loaded former model from {model_file_path}")
        test_data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        counter = 0

        with torch.no_grad():
            for batch_inputs, batch_targets, batch_aux in test_data_loader:
                inputs = (batch_inputs + 1) / 2
                batch_outputs = (self.network(inputs)) * 2 - 1
                input_dv_dx  = torch.diff(batch_inputs[:, 1, :, :],  dim=1, prepend=batch_inputs[:, 1, :, -1].unsqueeze(1))
                input_du_dy  = torch.diff(batch_inputs[:, 0, :, :],  dim=2, prepend=batch_inputs[:, 0, :, -1].unsqueeze(2))
                output_dv_dx = torch.diff(batch_outputs[:, 1, :, :], dim=1, prepend=batch_outputs[:, 1, :, -1].unsqueeze(1))
                output_du_dy = torch.diff(batch_outputs[:, 0, :, :], dim=2, prepend=batch_outputs[:, 0, :, -1].unsqueeze(2))

                input_vorticity  = (input_dv_dx - input_du_dy)
                output_vorticity = (output_dv_dx - output_du_dy)
                input_vorticity_hist = self.differentiable_histogram(input_vorticity, bins=128, min_value=-1, max_value=1)
                output_vorticity_hist = self.differentiable_histogram(output_vorticity, bins=128, min_value=-1, max_value=1)
                # to numpy
                input_vorticity  = input_vorticity.cpu().numpy()
                output_vorticity = output_vorticity.cpu().numpy()
                input_vorticity_hist  = input_vorticity_hist.cpu().numpy()
                output_vorticity_hist = output_vorticity_hist.cpu().numpy()
                for i in range(input_vorticity.shape[0]):
                    np.save(os.path.join(output_path,f'input_vorticity_{counter}.npy'),  input_vorticity[i,...])
                    np.save(os.path.join(output_path,f'output_vorticity_{counter}.npy'), output_vorticity[i,...])

                    np.save(os.path.join(output_path,f'input_vorticity_hist_{counter}.npy'),  input_vorticity_hist[i,...])
                    np.save(os.path.join(output_path,f'output_vorticity_hist_{counter}.npy'), output_vorticity_hist[i,...])

                    counter += 1
            

    def differentiable_histogram(self, data, bins, min_value, max_value):
        """
        Compute a differentiable histogram.
        
        :param data: Input tensor of shape [batch_size, ...]
        :param bins: Number of histogram bins
        :param min_value: Minimum value for the histogram
        :param max_value: Maximum value for the histogram
        :return: Differentiable histogram of shape [batch_size, bins]
        """
        # Create bin centers
        bin_centers = torch.linspace(min_value, max_value, steps=bins, device=data.device)

        # Reshape data for broadcasting
        data_reshape = data.flatten(1).unsqueeze(-1)  # Shape: [batch_size, N, 1]
        

        # Compute contributions using a Gaussian kernel (with a small std deviation)
        sigma = (max_value - min_value) / bins
        contributions = torch.exp(-0.5 * torch.pow((data_reshape - bin_centers) / sigma, 2))

        # Normalize contributions (so that they sum up to 1)
        normalization_factor = contributions.sum(dim=-1, keepdim=True).clamp(min=1e-7)
        contributions_norm = contributions / normalization_factor

        # Sum contributions for each bin
        histogram = contributions_norm.sum(dim=1)

        return histogram
