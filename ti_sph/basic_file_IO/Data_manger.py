import os
import numpy as np
from typing import List

class Grid_Data_manager:
    def __init__(self, input_folder_path: str, output_folder_path: str) -> None:
        self.input_folder_path = os.path.abspath(input_folder_path)
        self.output_folder_path = os.path.abspath(output_folder_path)
        self.start_index = None
        self.end_index = None
        self.channel_num = None
        self.raw_data = []
        self.processed_data = []
        
    # def read_data(self, attr:str, start_index:int, end_index: int, channel_num:int = 1):
    #     self.start_index = start_index
    #     self.end_index = end_index
    #     self.channel_num = channel_num
    #     for i in range(start_index, end_index):
    #         single_frame_data = []
    #         for c in range(channel_num):
    #             file_name = attr+'_i'+str(i)+'_c'+str(c)+'.npy'
    #             file_path = os.path.join(self.input_folder_path, file_name)
    #             data_arr = np.load(file_path)
    #             single_frame_data.append(data_arr)
    #         self.raw_data.append(single_frame_data)
    
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
    

    # def reshape_data_to_2d(self, index_attr:str):
    #     # self.processed_data = np.empty((self.end_index-self.start_index, self.channel_num), dtype=np.dtype(object))
    #     for i in range(self.start_index, self.end_index):
    #         index_arr = self.get_index_arr_2d(index_attr, i)
    #         single_frame_processed_data = []
    #         for c in range(self.channel_num):
    #             single_frame_processed_data.append(self.reshape_data_with_index_2d(self.raw_data[i-self.start_index][c], index_arr))
    #         self.processed_data.append(single_frame_processed_data)
    #     self.processed_data = np.array(self.processed_data)
    #     return self.processed_data
    
    def reshape_data_to_2d(self, index_attr:str):
        for i in range(self.start_index, self.end_index):
            index_arr = self.get_index_arr_2d(index_attr, i)
            self.processed_data.append(self.reshape_data_with_index_2d(self.raw_data[i-self.start_index], index_arr))

    # def get_index_arr_2d(self, index_attr:str, i:int):
    #     index_path = []
    #     for d in range(2):
    #         index_name = index_attr+'_i'+str(i)+'_c'+str(d)+'.npy'
    #         index_path.append(os.path.join(self.input_folder_path, index_name))
    #     index_arr = np.column_stack((np.load(index_path[0]), np.load(index_path[1])))
    #     return index_arr
    
    def get_index_arr_2d(self, index_attr:str, i:int):
        index_name = index_attr+'_'+str(i)+'.npy'
        index_path = os.path.join(self.input_folder_path, index_name)
        index_arr = np.load(index_path)
        return index_arr
    
    # def reshape_data_with_index_2d(self, data_arr, index_arr):
    #     rows = index_arr[:,0].max() + 1
    #     cols = index_arr[:,1].max() + 1
    #     reshaped_data = np.empty((rows, cols), dtype=float)
    #     reshaped_data[index_arr[:, 0], index_arr[:, 1]] = data_arr
    #     return reshaped_data
    
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

