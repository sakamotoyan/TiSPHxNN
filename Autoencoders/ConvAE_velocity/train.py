import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# from torchview import draw_graph
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

import os
import taichi as ti
import numpy as np

from .network import ConvAutoencoder
from .dataset import DatasetConvAutoencoder

@ti.data_oriented
class TrainConvAutoencoder:

    TI_Soble_X = ti.Matrix([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])

    TI_Soble_Y = ti.Matrix([[-1, -2, -1],
                            [ 0,  0,  0],
                            [ 1,  2,  1]])
    
    def __init__(self, res, attr_name_1, dataset_file_path_1, 
                            attr_name_2, dataset_file_path_2, 
                            attr_name_3, dataset_file_path_3,
                            network: nn.Module,
                            platform='cuda', multi_gpu=False, lr = 0.00005,
                            ):
        

        self.batch_size = 32
        self.lr = lr
        self.platform = platform
        self.network = network
        if multi_gpu:
            self.network = nn.DataParallel(self.network)
        self.network.to(platform)
        self.criterion = nn.MSELoss()

        self.dataset = DatasetConvAutoencoder(res, attr_name_1, dataset_file_path_1, attr_name_2, dataset_file_path_2, attr_name_3, dataset_file_path_3, self.platform)
        self.data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        self.loss_list = []

    def train(self, num_epochs, network_model_path, former_model_file_path=None, save_step=20, crop_boundary=False, crop=False, exclude_threshold=None):
        current_epochs_num = 0
        
        if former_model_file_path is not None:
            self.network.load_state_dict(torch.load(former_model_file_path))
            current_epochs_num = int(former_model_file_path.rsplit('_', 1)[1].split('.')[0]) + 1
            print(f"Loaded former model from {former_model_file_path}")
        else:
            self.network.apply(self.init_weights)
        
        self.network.train()
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)

        for epoch in range(num_epochs):
            running_loss = 0.0
            for batch_inputs in self.data_loader:
                batch_inputs = batch_inputs.to(self.platform)
                loss = 0.0
                self.optimizer.zero_grad()
                if crop is not False: 
                    crop_size = torch.randint(64, 256, (1,)).item()
                    # crop_size = 128
                    batch_inputs = self.func_random_crop_and_upsample(batch_inputs, crop_size=(crop_size,crop_size), exclude_threshold=exclude_threshold)
                
                inputs  = (batch_inputs + 1) / 2
                batch_outputs = (self.network(inputs)) * 2 - 1

                input_dv_dx = torch.diff(batch_inputs[:, 1, :, :], dim=1, prepend=batch_inputs[:, 1, :, -1].unsqueeze(1))
                input_du_dy = torch.diff(batch_inputs[:, 0, :, :], dim=2, prepend=batch_inputs[:, 0, :, -1].unsqueeze(2))
                input_vorticity = (input_dv_dx - input_du_dy)
                input_vorticity = input_vorticity.unsqueeze(1)
                

                output_dv_dx = torch.diff(batch_outputs[:, 1, :, :], dim=1, prepend=batch_outputs[:, 1, :, -1].unsqueeze(1))
                output_du_dy = torch.diff(batch_outputs[:, 0, :, :], dim=2, prepend=batch_outputs[:, 0, :, -1].unsqueeze(2))
                output_vorticity = (output_dv_dx - output_du_dy)
                output_vorticity = output_vorticity.unsqueeze(1)

                if crop_boundary:
                    input_vorticity  = input_vorticity [:, :, 3:-3, 3:-3]
                    output_vorticity = output_vorticity[:, :, 3:-3, 3:-3]

                loss = self.criterion(output_vorticity, input_vorticity)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            average_loss = running_loss / len(self.data_loader)
            self.loss_list.append(average_loss)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.8f}")

            if (epoch+1) % save_step == 0:    
                torch.save(self.network.state_dict(), os.path.join(network_model_path,f'epochs_{epoch+current_epochs_num}.pth'))
                self.save_loss_fig(epoch+current_epochs_num, network_model_path)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def save_loss_fig(self, epoch, path='./dataset_train', name=None, attr=None):
        if attr is not None:
            plt.plot(attr)
        else:
            plt.plot(self.loss_list)
        plt.title('Loss')
        plt.xlabel('epoch (batch_size=32)')
        if name is not None:
            save_path = os.path.join(path, f'loss_{name}_{epoch}.png')
        else:
            save_path = os.path.join(path, f'loss_{epoch}.png')
        plt.savefig(save_path)
        plt.close()

    def change_dataset(self, attr_name_1, dataset_file_path_1, attr_name_2, dataset_file_path_2, attr_name_3, dataset_file_path_3):
        self.dataset = DatasetConvAutoencoder(attr_name_1, dataset_file_path_1, attr_name_2, dataset_file_path_2, attr_name_3, dataset_file_path_3)
        self.data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def test(self, model_file_path, output_path, crop=False, exclude_threshold=None, export_bottleneck_layer=None, shuffle=False):
            
        self.network.load_state_dict(torch.load(model_file_path, torch.device(self.platform)))
        self.network.eval()
        print(f"Loaded former model from {model_file_path}")
        test_data_loader = DataLoader(self.dataset, batch_size=1, shuffle=shuffle)
        counter = 0
        loss_vel = []
        loss_vort = []
        loss_hist = []

        if export_bottleneck_layer is not None:
            self.network.bottleneck.bottleneck[export_bottleneck_layer].register_forward_hook(self.network.bottleneck.get_activation('bottleneck'))

        with torch.no_grad():
            for batch_inputs in test_data_loader:
                batch_inputs = batch_inputs.to(self.platform)
                if crop is not False: 
                    batch_inputs = self.func_random_crop_and_upsample(batch_inputs,crop_size=(64,64))

                inputs = (batch_inputs + 1) / 2
                batch_outputs = (self.network(inputs)) * 2 - 1
                input_dv_dx  = torch.diff(batch_inputs[:, 1, :, :],  dim=1, prepend=batch_inputs[:, 1, :, -1].unsqueeze(1))
                input_du_dy  = torch.diff(batch_inputs[:, 0, :, :],  dim=2, prepend=batch_inputs[:, 0, :, -1].unsqueeze(2))
                output_dv_dx = torch.diff(batch_outputs[:, 1, :, :], dim=1, prepend=batch_outputs[:, 1, :, -1].unsqueeze(1))
                output_du_dy = torch.diff(batch_outputs[:, 0, :, :], dim=2, prepend=batch_outputs[:, 0, :, -1].unsqueeze(2))

                input_vorticity  = (input_dv_dx - input_du_dy)
                output_vorticity = (output_dv_dx - output_du_dy)
                input_vorticity_hist = self.func_differentiable_histogram(input_vorticity, bins=128, min_value=-1, max_value=1)
                output_vorticity_hist = self.func_differentiable_histogram(output_vorticity, bins=128, min_value=-1, max_value=1)

                loss_vel.append(self.criterion(batch_outputs, batch_inputs).item())
                loss_vort.append(self.criterion(output_vorticity, input_vorticity).item())
                loss_hist.append(self.criterion(output_vorticity_hist, input_vorticity_hist).item())
                # to numpy
                batch_inputs  = batch_inputs.cpu().numpy()
                batch_outputs = batch_outputs.cpu().numpy()
                input_vorticity  = input_vorticity.cpu().numpy()
                output_vorticity = output_vorticity.cpu().numpy()
                input_vorticity_hist  = input_vorticity_hist.cpu().numpy()
                output_vorticity_hist = output_vorticity_hist.cpu().numpy()
                for i in range(input_vorticity.shape[0]):
                    np.save(os.path.join(output_path,f'input_velocity_{counter}.npy'),  batch_inputs[i,...])
                    np.save(os.path.join(output_path,f'output_velocity_{counter}.npy'), batch_outputs[i,...])

                    np.save(os.path.join(output_path,f'input_vorticity_{counter}.npy'),  input_vorticity[i,...])
                    np.save(os.path.join(output_path,f'output_vorticity_{counter}.npy'), output_vorticity[i,...])

                    np.save(os.path.join(output_path,f'input_vorticity_hist_{counter}.npy'),  input_vorticity_hist[i,...])
                    np.save(os.path.join(output_path,f'output_vorticity_hist_{counter}.npy'), output_vorticity_hist[i,...])

                    if export_bottleneck_layer is not None:
                        batch_bottleneck = self.network.bottleneck.feature_maps['bottleneck'].cpu().numpy()
                        np.save(os.path.join(output_path,f'bottleneck_{counter}.npy'), batch_bottleneck)

                    counter += 1
            self.save_loss_fig(0, output_path, 'vel', loss_vel)
            self.save_loss_fig(0, output_path, 'vort', loss_vort)
            self.save_loss_fig(0, output_path, 'hist', loss_hist)
    
    # TODO: implement this
    def test_conv(self, model_file_path, output_path, res):
        self.network.load_state_dict(torch.load(model_file_path))
        self.network.eval()
        print(f"Loaded former model from {model_file_path}")
        test_data_loader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        counter = 0
        loss_vel = []
        loss_vort = []
        loss_hist = []

        with torch.no_grad():
            for batch_inputs in test_data_loader:
                batch_inputs = batch_inputs.to(self.platform)
                # allocate memory for batch_outputs
                batch_outputs = torch.zeros((batch_inputs.shape[0], batch_inputs.shape[1], batch_inputs.shape[2], batch_inputs.shape[3])).to(self.platform)
                
                for i in range(int(batch_inputs.shape[2]/res)):
                    for j in range(int(batch_inputs.shape[3]/res)):
                        cropped_inputs = batch_inputs[:, :, i*res:(i+1)*res, j*res:(j+1)*res]

                        # max_velocity = torch.max(torch.sqrt(torch.pow(cropped_inputs[:, 0, :, :], 2) + torch.pow(cropped_inputs[:, 1, :, :], 2)))
                        # cropped_inputs[:, 0, :, :] = cropped_inputs[:, 0, :, :] / max_velocity if max_velocity > 0 else 0
                        # cropped_inputs[:, 1, :, :] = cropped_inputs[:, 1, :, :] / max_velocity if max_velocity > 0 else 0

                        upsampled_inputs = F.interpolate(cropped_inputs, size=(256, 256), mode='bilinear', align_corners=False)
                        cropped_batch_outputs = (self.network((upsampled_inputs+1)/2)) * 2 - 1
                        cropped_batch_outputs = F.interpolate(cropped_batch_outputs, size=(res, res), mode='bilinear', align_corners=False)

                        # cropped_batch_outputs *= max_velocity
                        # cropped_inputs *= max_velocity

                        batch_outputs[:, :, i*res:(i+1)*res, j*res:(j+1)*res] = cropped_batch_outputs
                input_dv_dx  = torch.diff(batch_inputs[:, 1, :, :],  dim=1, prepend=batch_inputs[:, 1, :, -1].unsqueeze(1))
                input_du_dy  = torch.diff(batch_inputs[:, 0, :, :],  dim=2, prepend=batch_inputs[:, 0, :, -1].unsqueeze(2))
                output_dv_dx = torch.diff(batch_outputs[:, 1, :, :], dim=1, prepend=batch_outputs[:, 1, :, -1].unsqueeze(1))
                output_du_dy = torch.diff(batch_outputs[:, 0, :, :], dim=2, prepend=batch_outputs[:, 0, :, -1].unsqueeze(2))

                input_vorticity  = (input_dv_dx - input_du_dy)
                output_vorticity = (output_dv_dx - output_du_dy)
                input_vorticity_hist = self.func_differentiable_histogram(input_vorticity, bins=128, min_value=-1, max_value=1)
                output_vorticity_hist = self.func_differentiable_histogram(output_vorticity, bins=128, min_value=-1, max_value=1)

                loss_vel.append(self.criterion(batch_outputs, batch_inputs).item())
                loss_vort.append(self.criterion(output_vorticity, input_vorticity).item())
                loss_hist.append(self.criterion(output_vorticity_hist, input_vorticity_hist).item())
                # to numpy
                batch_inputs  = batch_inputs.cpu().numpy()
                batch_outputs = batch_outputs.cpu().numpy()
                input_vorticity  = input_vorticity.cpu().numpy()
                output_vorticity = output_vorticity.cpu().numpy()
                input_vorticity_hist  = input_vorticity_hist.cpu().numpy()
                output_vorticity_hist = output_vorticity_hist.cpu().numpy()
                
                for i in range(input_vorticity.shape[0]):
                    np.save(os.path.join(output_path,f'input_velocity_{counter}.npy'),  batch_inputs[i,...])
                    np.save(os.path.join(output_path,f'output_velocity_{counter}.npy'), batch_outputs[i,...])

                    np.save(os.path.join(output_path,f'input_vorticity_{counter}.npy'),  input_vorticity[i,...])
                    np.save(os.path.join(output_path,f'output_vorticity_{counter}.npy'), output_vorticity[i,...])

                    np.save(os.path.join(output_path,f'input_vorticity_hist_{counter}.npy'),  input_vorticity_hist[i,...])
                    np.save(os.path.join(output_path,f'output_vorticity_hist_{counter}.npy'), output_vorticity_hist[i,...])

                    counter += 1
            self.save_loss_fig(0, output_path, 'vel', loss_vel)
            self.save_loss_fig(0, output_path, 'vort', loss_vort)
            self.save_loss_fig(0, output_path, 'hist', loss_hist)

    def output_bottleneck(self, model_file_path, output_path, layer):
        self.network.load_state_dict(torch.load(model_file_path))
        self.network.eval()
        print(f"Loaded former model from {model_file_path}")
        test_data_loader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        counter = 0
        
        with torch.no_grad():
            for batch_inputs in test_data_loader:
                inputs = (batch_inputs + 1) / 2
                batch_outputs = (self.network(inputs)) * 2 - 1
                batch_bottleneck = self.network.bottleneck.feature_maps['bottleneck'].cpu().numpy()
                np.save(os.path.join(output_path,f'bottleneck_{counter}.npy'), batch_bottleneck)
                counter += 1

    def Conv2D(self, input_data, kernel_size, stride):
        if len(input_data.shape) is not 2:
            raise ValueError("Input data must be 2D")
        padded_input = F.pad(input_data, (kernel_size, kernel_size, kernel_size, kernel_size), mode='constant', value=0)
        conv_counter = torch.zeros(input_data.shape, dtype=torch.uint8)
        output = torch.zeros(input_data.shape, dtype=torch.float32)
        

    def func_random_crop_and_upsample(self, data, crop_size=(128, 128), upsample_size=(256, 256), exclude_threshold=None):
        batch_size, num_channel, height, width = data.shape
        cropped_data = []

        for i in range(batch_size):
            # Randomly choose the top-left pixel of the cropping area
            top = torch.randint(0, height - crop_size[0] + 1, (1,)).item()
            left = torch.randint(0, width - crop_size[1] + 1, (1,)).item()

            # Perform the crop
            crop = data[i, :, top:top + crop_size[0], left:left + crop_size[1]]

            # Upsample the cropped image
            upsampled_crop = F.interpolate(crop.unsqueeze(0), size=upsample_size, mode='bilinear', align_corners=False)

            max_velocity = torch.max(torch.sqrt(torch.pow(upsampled_crop[0, 0, :, :], 2) + torch.pow(upsampled_crop[0, 1, :, :], 2)))
            upsampled_crop[0, 0, :, :] = upsampled_crop[0, 0, :, :] / max_velocity if max_velocity > 0 else 0
            upsampled_crop[0, 1, :, :] = upsampled_crop[0, 1, :, :] / max_velocity if max_velocity > 0 else 0

            if exclude_threshold is not None:
                # Compute sum of square root for each element
                sum_of_square_root = torch.sqrt(torch.sum(torch.pow(upsampled_crop, 2)))
                if sum_of_square_root < exclude_threshold:
                    continue
            
            cropped_data.append(upsampled_crop)

        # Concatenate all the upsampled images back into a single tensor
        new_data = torch.cat(cropped_data, dim=0)
        return new_data

    def func_differentiable_histogram(self, data, bins, min_value, max_value):
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
