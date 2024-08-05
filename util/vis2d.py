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
import matplotlib.pyplot as plt
np.set_printoptions(threshold=sys.maxsize)

class Vis2D:
    def __init__(self, path, output_path, name, start_index, radius, lb, rt, dpi=200, end_index=None, flag_name=None, offset=0):
        self.path = path
        self.output_path = output_path
        self.name = name
        self.radius = radius
        self.lb = lb
        self.rt = rt
        self.dpi = dpi
        self.flag_name = flag_name
        self.offset = offset
        self.start_index = start_index
        if end_index is None:
            self.end_index = start_index + 1
        else:
            self.end_index = end_index + 1
        self.pos_data = []
        self.flag_data = []
        self.load_data_npy()
        if self.flag_name is not None:
            self.load_flag_npy()
        
        self.output_name = 'vis2d'
        self.output_visualized_data(self.output_path, self.output_name)
    
    def load_data_npy(self):
        for i in range(self.start_index, self.end_index):
            filename = os.path.join(self.path, self.name + '_' + str(i) + '.npy')
            self.pos_data.append(np.load(filename))
    
    def load_flag_npy(self):
        for i in range(self.start_index, self.end_index):
            filename = os.path.join(self.path, self.flag_name + '_' + str(i) + '.npy')
            self.flag_data.append(np.load(filename))
    
    def output_visualized_data(self, output_path, output_name):
        for i in range(self.start_index, self.end_index):
            self.output_visualized_data_single(i-self.offset, output_path, output_name)
    
    def output_visualized_data_single(self, index, output_path, output_name):
        plt.figure()

        plt.xlim(self.lb[0], self.rt[0])
        plt.ylim(self.lb[1], self.rt[1])

        plt.xlabel('X-axis (m)')
        plt.ylabel('Y-axis (m)')

        plt.axis('square')

        plt.title('2D Position Plot with RGB Colors')

        if self.flag_name is not None:
            x = self.pos_data[index][:, 0]
            y = self.pos_data[index][:, 1]
            flag = self.flag_data[index]
            number_of_colors = len(np.unique(flag))
            # colors = plt.cm.viridis(np.linspace(0, 1, number_of_colors))
            colors = np.zeros((number_of_colors, 4))
            colors[0, :] = np.array([1.0, 1.0, 1.0, 1.0])
            colors[1, :] = np.array([0.0, 0.0, 0.0, 1.0])
            print(colors)
            for i in range(number_of_colors):
                x_i = x[flag == i]
                y_i = y[flag == i]
                plt.scatter(x_i, y_i, s=self.radius*10, c=colors[i], edgecolors='none')
        else:
            x = self.pos_data[index][:, 0]
            y = self.pos_data[index][:, 1]
            plt.scatter(x, y, s=self.radius*10, c='blue', edgecolors='none')
        
        ax = plt.gca()
        ax.set_facecolor('white')
        fig = plt.gcf()
        fig.set_facecolor('white')

        plt.savefig(os.path.join(output_path, output_name + '_' + str(index+self.offset) + '.png'), dpi=self.dpi)
        # plt.savefig(os.path.join(output_path, output_name + '_' + str(index+self.offset) + '.eps'))
        plt.close()

Vis2D(path=os.path.join(parent_dir, 'output2'), 
      output_path=os.path.join(parent_dir, 'output_vis'),
      name          ='part', 
      start_index   =155, 
      offset        =155,
    #   end_index     =155,
      radius        =0.002, 
      lb            =np.array([-2.5, -4.0]), 
      rt            =np.array([ 2.5,  1.0]), 
      dpi           =3500, 
      flag_name     ='flag',
      )