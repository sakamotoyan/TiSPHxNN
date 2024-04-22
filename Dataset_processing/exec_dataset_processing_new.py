import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import taichi as ti
import ti_sph as tsph
from Dataset_processing import *

ti.init(arch=ti.gpu)

main_path = '../output'
output_path = '../output/organized'
input_compressed = True
output_compressed = False
attr_list = ['vel', 'pos', 'sensed_density', 'strainRate']
output_attr_list = ['velocity', 'position', 'density', 'strainRate']

for attr, output_attr in zip(attr_list, output_attr_list):
    tsph.SeqData(path=main_path, attr_name=attr, compressed=input_compressed)\
        .reshape_to_3d(index_attr_name='node_index', output_path=output_path, output_name=output_attr, compressed=output_compressed)