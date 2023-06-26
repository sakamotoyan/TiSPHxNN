import taichi as ti
from ti_sph import *
import numpy as np

dm = Grid_Data_manager('./output')
dm.batch_op(attr='pos',start_index=0,end_index=0,channel_num=2, operation=dm.op_reshape_data_with_index, index_attr='node_index', dim=2)
print(dm.data[0])