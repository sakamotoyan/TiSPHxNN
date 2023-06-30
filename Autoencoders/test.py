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
autoencoder.load_state_dict(torch.load(os.path.join(model_path,'autoencoder.pth')))
autoencoder.eval()

test_results = []

dataset = get_dataset()
test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

with torch.no_grad():
    for batch in test_loader:
        # Move the batch to the device (e.g., GPU)
        inputs = batch.to('cuda')

        # Generate predictions
        outputs = autoencoder(inputs)
        
        # Append the predictions to our results list
        # Detach the tensor from the computation graph and convert to numpy
        test_results.append(outputs.detach().cpu().numpy())

for i in range(len(test_results)):
    for j in range(test_results[i].shape[0]):
        singe_density_frame = test_results[i][j,0]

        frame_data = np.flipud(np.transpose(singe_density_frame))
        normalized_array = ((frame_data - frame_data.min()) * (255 - 0) / (frame_data.max() - frame_data.min())).astype(np.uint8)
        img = Image.fromarray(normalized_array, 'L')
        img.save(os.path.join(output_test_path,f'frame_density_{i*batch_size+j}.jpg'))
        # print(singe_density_frame.shape)
# Save the results list to disk as a numpy object
# np.save('test_results.npy', np.array(test_results))

