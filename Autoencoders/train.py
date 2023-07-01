from ConvAE import ConvAutoencoder
from settings import *
from dataset import get_dataset

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os


dataset=get_dataset()


### TRAINING
autoencoder = ConvAutoencoder()
autoencoder.to(PLATFORM)
if os.path.exists(os.path.join(network_model_path,state_dict_name)):
    autoencoder.load_state_dict(torch.load(os.path.join(network_model_path,state_dict_name)))
criterion = nn.MSELoss()
optimizer = optim.Adagrad(autoencoder.parameters(), lr=0.001)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    running_loss = 0.0

    for batch_inputs, batch_targets in data_loader:
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        inputs = batch_inputs.to(PLATFORM) 
        targets = []
        for i in range(num_forecast_steps): 
            targets.append(batch_targets[i].to(PLATFORM))

        output_list = []
        current_input = inputs
        loss = 0.0
        for i in range(num_forecast_steps):
            outputs = autoencoder(current_input)
            output_list.append(outputs)
            current_input = outputs
            # Compute the loss
            loss += criterion(outputs, targets[i])
        

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Update the running loss
        running_loss += loss.item()

    # Compute the average loss for the epoch
    average_loss = running_loss / len(data_loader)

    # Print the average loss for the epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.8f}")

torch.save(autoencoder.state_dict(), os.path.join(network_model_path,state_dict_name))