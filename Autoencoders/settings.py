import os

state_dict_name='autoencoder_4.pth'
PLATFORM = 'cuda'

# Training
begin_index = 0
end_index = 1200
channel_num = 3
res = 64
batch_size = 32
num_epochs = 1000
num_forecast_steps = 5

# Testing
num_test_steps = 500

# Path
training_data_path = os.path.join(os.getcwd(),'training_data')
network_model_path = os.path.join(os.getcwd(),'NN_models')
output_test_path = os.path.join(os.getcwd(),'test_result')