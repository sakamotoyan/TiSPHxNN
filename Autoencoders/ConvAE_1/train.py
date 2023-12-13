import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

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
        

        self.batch_size = 32
        self.platform = platform
        self.network = ConvAutoencoder_1(feature_vector_size)
        self.network.to(platform)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.network.parameters(), lr=0.001)
        self.dataset = DatasetConvAutoencoder_1(res, attr_name_1, dataset_file_path_1, attr_name_2, dataset_file_path_2, attr_name_3, dataset_file_path_3)
        self.data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        self.ti_aux_density       = ti.field(dtype=ti.f32, shape=(res,   res,   self.batch_size))
        self.ti_output_velocity_x = ti.field(dtype=ti.f32, shape=(res,   res,   self.batch_size))
        self.ti_output_velocity_y = ti.field(dtype=ti.f32, shape=(res,   res,   self.batch_size))
        self.ti_output_vorticity  = ti.field(dtype=ti.f32, shape=(res-2, res-2, self.batch_size))

    
    def train(self, num_epochs):
        for epoch in range(num_epochs):
            for batch_inputs, batch_targets, batch_aux in self.data_loader:
                loss = 0.0
                self.optimizer.zero_grad()
                
                inputs = batch_inputs.to(self.platform) 
                outputs = self.network(inputs)

                self.ti_aux_density.from_torch(batch_aux.permute(1,2,0))
                self.ti_output_velocity_x.from_torch(outputs[:,0].permute(1,2,0))
                self.ti_output_velocity_y.from_torch(outputs[:,1].permute(1,2,0))
                self.compute_vorticity(self.ti_output_velocity_x, self.ti_output_velocity_y, self.ti_output_vorticity)
                self.output_hist = np.array([np.histogram(self.ti_output_vorticity.to_numpy()[:, :, i], bins=128, range=(-1, 1))[0] for i in range(self.batch_size)])


                print(self.output_hist.sum()/self.batch_size, batch_targets.sum()/self.batch_size)
    
    @ti.kernel
    def compute_vorticity(self, ti_velocity_x:ti.template(), ti_velocity_y:ti.template(), ti_vorticity:ti.template()):
        for index_x in range(1, ti_velocity_x.shape[0]-1):
            for index_y in range(1, ti_velocity_x.shape[1]-1):
                for batch in range(self.batch_size):
                    data3x3_x=ti.Matrix([[ti_velocity_x[index_x-1,index_y-1,batch],ti_velocity_x[index_x,index_y-1,batch],ti_velocity_x[index_x+1,index_y-1,batch]],
                                         [ti_velocity_x[index_x-1,index_y  ,batch],ti_velocity_x[index_x,index_y  ,batch],ti_velocity_x[index_x+1,index_y  ,batch]],
                                         [ti_velocity_x[index_x-1,index_y+1,batch],ti_velocity_x[index_x,index_y+1,batch],ti_velocity_x[index_x+1,index_y+1,batch]]])
                    data3x3_y=ti.Matrix([[ti_velocity_y[index_x-1,index_y-1,batch],ti_velocity_y[index_x,index_y-1,batch],ti_velocity_y[index_x+1,index_y-1,batch]],
                                         [ti_velocity_y[index_x-1,index_y  ,batch],ti_velocity_y[index_x,index_y  ,batch],ti_velocity_y[index_x+1,index_y  ,batch]],
                                         [ti_velocity_y[index_x-1,index_y+1,batch],ti_velocity_y[index_x,index_y+1,batch],ti_velocity_y[index_x+1,index_y+1,batch]]])
                    self.ti_output_vorticity[index_x-1, index_y-1, batch] = (data3x3_x * self.TI_Soble_X).sum() / 8 - (data3x3_y * self.TI_Soble_Y).sum() / 8

