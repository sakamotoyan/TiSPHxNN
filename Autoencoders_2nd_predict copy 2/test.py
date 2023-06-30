from ConvNN import ConvAutoencoder
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

output_test_path = os.path.join(os.getcwd(),'test_result')

autoencoder = ConvAutoencoder()
autoencoder.to('cuda')
autoencoder.load_state_dict(torch.load(os.path.join(model_path,'autoencoder_2.pth')))
autoencoder.eval()

test_results = []

dataset = get_dataset()
# test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
initial_input = dataset.data[0].to('cuda').unsqueeze(0)
current_input = initial_input
generated_sequence = [current_input.cpu().numpy()]
# print(current_input.cpu().numpy().shape)

num_test_steps = 500
with torch.no_grad():
    for _ in range(num_test_steps):
        # Generate prediction
        output = autoencoder(current_input)

        # Use the output as the next input
        current_input = output
        
        # Store the output
        generated_sequence.append(output.detach().cpu().numpy())

for i in range(len(generated_sequence)):
    singe_density_frame = generated_sequence[i][0,0]

    frame_data = np.flipud(np.transpose(singe_density_frame))
    normalized_array = ((frame_data - frame_data.min()) * (255 - 0) / (frame_data.max() - frame_data.min())).astype(np.uint8)
    img = Image.fromarray(normalized_array, 'L')
    img.save(os.path.join(output_test_path,f'frame_density_{i}.jpg'))
        # print(singe_density_frame.shape)
# Save the results list to disk as a numpy object
# np.save('test_results.npy', np.array(test_results))

