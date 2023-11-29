import os
import numpy as np
from typing import List
from multiprocessing import Process

class Grid_Data_manager:
    def __init__(self, input_folder_path: str, output_folder_path: str) -> None:
        self.input_folder_path = os.path.abspath(input_folder_path)
        self.output_folder_path = os.path.abspath(output_folder_path)
        self.start_index = None
        self.end_index = None
        self.raw_data = []
        self.processed_data = []
    
    def read_data(self, attr:str, start_index:int, end_index: int):
        self.start_index = start_index
        self.end_index = end_index
        for i in range(start_index, end_index):
            file_name = attr+'_'+str(i)+'.npy'
            file_path = os.path.join(self.input_folder_path, file_name)
            data_arr = np.load(file_path)
            self.raw_data.append(data_arr)
    
    def export_data(self, name:str='data'):
        exported_data = self.processed_data
        np.save(os.path.join(self.output_folder_path, name+'.npy'), exported_data)
        return exported_data
    
    def export_single_frame_data(self, name:str='data', from_zero:bool=False):
        exported_data = self.processed_data
        if from_zero:
            for i in range(self.start_index, self.end_index):
                np.save(os.path.join(self.output_folder_path, name+'_'+str(i-self.start_index)+'.npy'), exported_data[i-self.start_index])
        else:
            for i in range(self.start_index, self.end_index):
                np.save(os.path.join(self.output_folder_path, name+'_'+str(i)+'.npy'), exported_data[i-self.start_index])
        return exported_data
    
    def reshape_data_to_2d(self, index_attr:str):
        for i in range(self.start_index, self.end_index):
            index_arr = self.get_index_arr_2d(index_attr, i)
            self.processed_data.append(self.reshape_data_with_index_2d(self.raw_data[i-self.start_index], index_arr))
    
    def get_index_arr_2d(self, index_attr:str, i:int):
        index_name = index_attr+'_'+str(i)+'.npy'
        index_path = os.path.join(self.input_folder_path, index_name)
        index_arr = np.load(index_path)
        return index_arr
    
    def reshape_data_with_index_2d(self, data_arr, index_arr):
        rows = index_arr[:,0].max() + 1
        cols = index_arr[:,1].max() + 1
        if len(data_arr.shape)==1:
            reshaped_data = np.empty((rows, cols), dtype=float)
            reshaped_data[index_arr[:, 0], index_arr[:, 1]] = data_arr
        elif len(data_arr.shape)==2:
            reshaped_data = np.empty((rows, cols, data_arr.shape[1]), dtype=float)
            for i in range(data_arr.shape[1]):
                reshaped_data[index_arr[:, 0], index_arr[:, 1], i] = data_arr[:, i]
        elif len(data_arr.shape)==3:
            reshaped_data = np.empty((rows, cols, data_arr.shape[1], data_arr.shape[2]), dtype=float)
            for i in range(data_arr.shape[1]):
                for j in range(data_arr.shape[2]):
                    reshaped_data[index_arr[:, 0], index_arr[:, 1], i, j] = data_arr[:, i, j]
        else:
            raise Exception("data_arr.shape is not supported")

        return reshaped_data

class Grid_Data:
    FRAME = 0
    SHAPE_X = 0
    SHAPE_Y = 1
    X_COMP = 0
    Y_COMP = 1

    # changing to the Cartesian coordinate system
    Soble_X = np.transpose(np.flip(np.array([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]]), axis=0))
    
    Soble_Y = np.transpose(np.flip(np.array([[ 1,  2,  1],
                                [ 0,  0,  0],
                                [-1, -2, -1]]), axis=0))

    def __init__(self, path, attr_name, start_index: int, end_index: int) -> None:
        self.path = path
        self.attr_name = attr_name
        self.start_index = start_index
        self.end_index = end_index
        self.shape_x = None
        self.shape_y = None
        
        if not os.path.exists(path):
            raise Exception("path does not exist")
        if not os.path.isdir(path):
            raise Exception("path is not a folder")
        if start_index < 0 or end_index < 0:
            raise Exception("start_index or end_index is negative")
        if start_index > end_index:
            raise Exception("start_index is larger than end_index")

        self.shape_x = self.read_single_frame_data(start_index).shape[self.SHAPE_X]
        self.shape_y = self.read_single_frame_data(start_index).shape[self.SHAPE_Y]
    
    def read_single_frame_data(self, frame):
        if frame < self.start_index or frame > self.end_index:
            raise Exception("frame is out of range")
        file_name = self.attr_name+'_'+str(frame)+'.npy'
        file_path = os.path.join(self.path, file_name)
        if not os.path.exists(file_path):
            raise Exception("file does not exist")
        np_single_frame_data = np.load(file_path)
        return np_single_frame_data
    
    def read_all_frame_data(self):
        np_all_frame_data = []
        for i in range(self.start_index, self.end_index):
            np_all_frame_data.append(self.read_single_frame_data(i))
        return np_all_frame_data

    def get_3x3_data(self, np_single_frame_data, grid_x, grid_y):
        # if (not 0 < grid_x < self.shape_x-1) or (not 0 < grid_y < self.shape_y-1):
        #     raise Exception("grid_x or grid_y is out of range")
        return np_single_frame_data[grid_x-1:grid_x+2, grid_y-1:grid_y+2]
    
    def get_single_data(self, np_single_frame_data, grid_x, grid_y):
        # if (not -1 < grid_x < self.shape_x) or (not -1 < grid_y < self.shape_y):
        #     raise Exception("grid_x or grid_y is out of range")
        return np_single_frame_data[grid_x, grid_y]
    
    def Sobel_conv(self, np_single_frame_data):
        channel_num = np_single_frame_data.shape[2]
        np_conv_val = np.zeros((self.shape_x-2, self.shape_y-2, channel_num, 2))

        for index_x in range(1, self.shape_x-1):
            for index_y in range(1, self.shape_y-1):
                for channel in range(channel_num):
                    np_conv_val[index_x-1, index_y-1, channel, 0] = \
                        np.sum(self.get_3x3_data(np_single_frame_data, index_x, index_y)[...,channel] * self.Soble_X)
                    np_conv_val[index_x-1, index_y-1, channel, 1] = \
                        np.sum(self.get_3x3_data(np_single_frame_data, index_x, index_y)[...,channel] * self.Soble_Y)
        return np_conv_val
        

