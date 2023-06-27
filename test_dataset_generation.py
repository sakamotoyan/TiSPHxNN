import taichi as ti
from ti_sph import *
import numpy as np

dm = Grid_Data_manager('./output', './output')
dm.read_data(attr='pos',start_index=0,end_index=6,channel_num=2)
dm.reshape_data_to_2d(index_attr='node_index')
data = dm.export_data('pos')

print(data[0,1,0,:])
print(data[6,1,0,:])
