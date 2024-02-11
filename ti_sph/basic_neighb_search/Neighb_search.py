import taichi as ti
from ..basic_op.type import *
from .Neighb_pool import Neighb_pool
from .Neighb_cell import NeighbGrid
from ..basic_obj.Obj_Particle import Particle

class Neighb_search:
    def __init__(self, obj:Particle, max_neighb_num:ti.template()=0):
        # self.attach_to_obj(obj)
        self.neighbGrid = NeighbGrid(obj)
        self.neighbPool = Neighb_pool(obj, max_neighb_num)
        self.update_self()

    # def attach_to_obj(self, obj):
    #     self.obj = obj
    #     if not hasattr(obj, "neighb_search"):
    #         obj.neighb_search = self
    #     else:
    #         raise Exception("obj already has neighb_search")

    def add_neighb_obj(self, neighb_obj: Particle, search_range: ti.template()):
        self.neighbPool.add_neighb_obj(neighb_obj, search_range)

    def update_self(self):
        self.neighbGrid.update_part_in_cell()

    def search_neighbors(self):
        self.neighbPool.register_neighbours()