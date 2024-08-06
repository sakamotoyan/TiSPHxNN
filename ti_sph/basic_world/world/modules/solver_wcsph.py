import taichi as ti
import numpy as np
from typing import List
from ....basic_obj.Obj_Particle import Particle

def init_solver_wcsph(self):
    self.wcsph_solver_list = []
    for part_obj in self.part_obj_list:
        part_obj: Particle
        if part_obj.getSolverWCSPH() is not None:
            self.wcsph_solver_list.append(part_obj)
    
def step_wcsph_add_acc_pressure(self):
    for part_obj in self.wcsph_solver_list:
        part_obj: Particle

        part_obj.getSolverWCSPH().compute_B()
        part_obj.getSolverWCSPH().ReLU_density()
        part_obj.getSolverWCSPH().compute_pressure()
    
    for part_obj in self.wcsph_solver_list:
        if part_obj.m_is_dynamic is True:
            for neighb_obj in part_obj.get_module_neighbSearch().neighb_obj_list:
                part_obj.getSolverWCSPH().loop_neighb(part_obj.m_neighb_search.neighb_pool, neighb_obj, part_obj.getSolverWCSPH().inloop_add_acc_pressure)

def step_wcsph_add_acc_number_density_pressure(self):
    for part_obj in self.wcsph_solver_list:
        part_obj: Particle
        
        part_obj.getSolverWCSPH().compute_B()
        part_obj.getSolverWCSPH().ReLU_density()
        part_obj.getSolverWCSPH().compute_pressure()
    
    for part_obj in self.wcsph_solver_list:
        if part_obj.m_is_dynamic is True:
            for neighb_obj in part_obj.get_module_neighbSearch().neighb_obj_list:
                part_obj.getSolverWCSPH().loop_neighb(part_obj.m_neighb_search.neighb_pool, neighb_obj, part_obj.getSolverWCSPH().inloop_add_acc_number_density_pressure)