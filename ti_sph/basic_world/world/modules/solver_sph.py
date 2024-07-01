import taichi as ti
from ....basic_obj.Obj_Particle import Particle

def init_solver_sph(self):
    self.sph_solver_list = []
    for part_obj in self.part_obj_list:
        part_obj: Particle
        if part_obj.getSolverSPH() is not None:
            self.sph_solver_list.append(part_obj)

def step_sph_compute_density(self):
    for part_obj in self.sph_solver_list:
        part_obj: Particle
        part_obj.getSolverSPH().sph_compute_density(part_obj.m_neighb_search.neighb_pool)

def step_sph_compute_number_density(self):
    for part_obj in self.sph_solver_list:
        part_obj: Particle
        part_obj.getSolverSPH().sph_compute_number_density(part_obj.m_neighb_search.neighb_pool)

def step_sph_compute_compression_ratio(self):
    for part_obj in self.sph_solver_list:
        part_obj: Particle
        part_obj.getSolverSPH().sph_compute_compression_ratio(part_obj.m_neighb_search.neighb_pool)