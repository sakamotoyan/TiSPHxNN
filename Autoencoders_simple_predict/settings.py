import os

begin_index = 0
end_index = 1200
channel_num = 3
res = 64
batch_size = 32
num_epochs = 1000
num_forecast_steps = 5

data_path = os.path.join(os.getcwd(),'training_data')
model_path = os.path.join(os.getcwd(),'NN_models')