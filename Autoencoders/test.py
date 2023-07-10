from ConvAE import ConvAutoencoder
from settings import *
from dataset import get_dataset

import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
from PIL import Image
from torchsummary import summary
import time
import matplotlib.pyplot as plt
import matplotlib.colors as colors

autoencoder = ConvAutoencoder()
autoencoder.to(PLATFORM)
autoencoder.load_state_dict(torch.load(os.path.join(network_model_path,'7_'+state_dict_name)))
autoencoder.eval()

test_results = []

dataset = get_dataset()
# test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
initial_input = dataset.data[start_frame].to(PLATFORM).unsqueeze(0)
current_input = initial_input
# generated_sequence = [current_input.cpu().numpy()]
generated_sequence = np.tile(current_input.cpu().numpy(), (num_test_steps,1,1,1))
generated_sequence[0] = current_input.cpu().numpy()

# start timing
start_time = time.time()

with torch.no_grad():
    for i in range(num_test_steps):
        # Generate prediction
        output = autoencoder(current_input)

        # Use the output as the next input
        current_input = output
        
        # Store the output
        # generated_sequence.append(output.detach().cpu().numpy())
        generated_sequence[i] = output.detach().cpu().numpy()

# end timing
end_time = time.time()
print(f"Time elapsed: {end_time - start_time:.2f} seconds")
        
for i in range(num_test_steps):
    singe_density_frame = generated_sequence[i,0]

    frame_data = np.flipud(np.transpose(singe_density_frame))
    normalized_array = ((frame_data - frame_data.min()) * (255 - 0) / (frame_data.max() - frame_data.min())).astype(np.uint8)
    img = Image.fromarray(normalized_array, 'L')
    img.save(os.path.join(output_test_path,f'frame_density_{i}.jpg'))


generated_sequence[:,1:3] -= 0.5
g_speed = np.sqrt(generated_sequence[:,1]**2 + generated_sequence[:,2]**2)
g_speed_min = g_speed.min()
g_speed_max = g_speed.max()
for i in range(num_test_steps):
    # Step 0
    # Get the velocity vector at each frame
    v = generated_sequence[i,1:3]
    v[0]=np.flipud(np.transpose(v[0]))
    v[1]=np.flipud(np.transpose(v[1]))
    # Step 1
    # Calculate speed and direction (angle) from x and y components of the velocity
    speed = np.sqrt(v[0]**2 + v[1]**2)
    angle = (np.arctan2(v[1], v[0]) + np.pi) / (2. * np.pi)
    # Step 2
    # Create HSV representation
    hsv = np.zeros((v.shape[1],v.shape[2],3))
    hsv[..., 0] = angle
    hsv[..., 1] = 1.0  # Set saturation to maximum
    # hsv[..., 2] = (speed - speed.min()) / (speed.max() - speed.min())  # SINGLE_FRAME Normalize speed to range [0,1]
    hsv[..., 2] = (speed - g_speed_min) / (g_speed_max - g_speed_min)  # GLOBAL Normalize speed to range [0,1]
    # Step 3
    # Convert HSV to RGB
    rgb = colors.hsv_to_rgb(hsv)
    plt.imsave(os.path.join(output_test_path,f'vel_density_{i}.jpg'), rgb)
# for i in range(len(generated_sequence)):
#     singe_density_frame_vel_x = generated_sequence[i][0,1]
#     singe_density_frame_vel_y = generated_sequence[i][0,2]

#     frame_data = np.flipud(np.transpose(singe_density_frame))
#     normalized_array = ((frame_data - frame_data.min()) * (255 - 0) / (frame_data.max() - frame_data.min())).astype(np.uint8)
#     img = Image.fromarray(normalized_array, 'L')
#     img.save(os.path.join(output_test_path,f'frame_density_{i}.jpg'))

