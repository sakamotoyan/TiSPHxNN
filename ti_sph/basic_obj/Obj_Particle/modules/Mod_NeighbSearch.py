import taichi as ti
from ....basic_op.type import *
from ....basic_neighb_search import Neighb_search

DEFAULT_VALUE = 0

@ti.data_oriented
class Mod_NeighbSearch:

    def add_module_neighb_search(self, 
                                 neighb_obj_list: list = [],  # list of Particle class
                                 max_neighb_num:ti.template()=0,):
        self.m_neighb_search = Neighb_search(self, neighb_obj_list, max_neighb_num)

    def add_unit_neighb_search(
            self,
            neighb_obj_list: list = [],
            max_neighb_num:ti.template()=0,
    ):
        unit_neighb_search = Neighb_search(self, neighb_obj_list, max_neighb_num)
        return unit_neighb_search
    
    # def check_neighb_search(self):
    #     if self.m_neighb_search is None:
    #         raise Exception("neighb_search not added")

    # def add_module_neighb_search(self, max_neighb_num:ti.template()=0):
    #     self.m_neighb_search = Neighb_search(self, max_neighb_num)

    # def add_neighb_obj(self, neighb_obj, search_range: ti.template()=DEFAULT_VALUE):
    #     self.check_neighb_search()
    #     if search_range == DEFAULT_VALUE:
    #         search_range = self.m_world.support_radius
    #     self.m_neighb_search.neighb_pool.add_neighb_obj(neighb_obj, search_range)

    # def add_neighb_objs(self, neighb_objs, search_range: ti.template()=DEFAULT_VALUE):
    #     self.check_neighb_search()
    #     if search_range == DEFAULT_VALUE:
    #         search_range = self.m_world.support_radius
    #     for neighb_obj in neighb_objs:
    #         self.m_neighb_search.neighb_pool.add_neighb_obj(neighb_obj, search_range)
    
    @ti.func
    def tiGet_module_neighbSearch(self,)->Neighb_search:
        return self.m_neighb_search
    def get_module_neighbSearch(self,)->Neighb_search:
        return self.m_neighb_search