import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
import taichi as ti
import ti_sph as tsph
from templates import part_template
from templates import grid_template
import time
import numpy as np
import csv
np.set_printoptions(threshold=sys.maxsize)

class Vis2D:
    def __init__(self, path, name, start_index, end_index=None, flag_name=None):
        self.path = path
        self.name = name
        self.flag_name = flag_name
        self.start_index = start_index
        if end_index is None:
            self.end_index = start_index + 1
        else:
            self.end_index = end_index + 1
        self.data = []
        self.flag_data = []
        self.load_data_npy()
        if self.flag_name is not None:
            self.load_flag_npy()
    
    def load_data_npy(self):
        for i in range(self.start_index, self.end_index):
            filename = os.path.join(self.path, self.name + str(i) + '.npy')
            self.data.append(np.load(filename))
    
    def load_flag_npy(self):
        for i in range(self.start_index, self.end_index):
            filename = os.path.join(self.path, self.flag_name + str(i) + '.npy')
            self.flag_data.append(np.load(filename))
    
    def output_visualized_data_single(self, index, output_path, output_name, output_flag_name=None):
        pass
    
    def output_visualized_data(self, output_path, output_name, output_flag_name=None):
        for i in range(self.start_index, self.end_index):
            if output_flag_name is not None:
                self.output_visualized_data_single(i, output_path, output_name, output_flag_name)
            else:
                self.output_visualized_data_single(i, output_path, output_name)
