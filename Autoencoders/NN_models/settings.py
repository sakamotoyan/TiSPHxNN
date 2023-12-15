import os

state_dict_name='autoencoder_4.pth'
PLATFORM = 'cuda'

# Training
channel_num = 3
res = 128
batch_size = 32
num_epochs = 100
num_rounds = 200
num_read_steps = 2
num_forecast_steps = 5

# Testing
start_frame = 0
num_test_steps = 3000

# Training & Testing
begin_index = 0
end_index = 10

# Path
training_data_path = os.path.join(os.getcwd(),'training_data')
network_model_path = os.path.join(os.getcwd(),'NN_models')
output_test_path = os.path.join(os.getcwd(),'test_result')