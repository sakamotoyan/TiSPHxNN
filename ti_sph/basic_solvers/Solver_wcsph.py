import taichi as ti
import math
from .sph_funcs import *
from .Solver_sph import SPH_solver
from ..basic_op.type import *
from ..basic_obj.Obj_Particle import Particle
from typing import List

@ti.data_oriented
class WCSPH_solver(SPH_solver):
    def __init__(self, obj: Particle, gamma, stiffness):
        
        super().__init__(obj)
        self.B = None

        self.gamma = gamma
        self.stiffness = stiffness

        self.B_by_rest_density = stiffness / self.gamma
    
    @ti.kernel
    def compute_B(self):
        for part_id in range(self.obj.tiGet_stack_top()[None]):
            self.obj.sph_wc[part_id].B = self.B_by_rest_density * self.obj.rest_density[part_id]

    @ti.kernel
    def ReLU_density(self):
        for part_id in range(self.obj.tiGet_stack_top()[None]):
            if self.obj.sph[part_id].density < self.obj.rest_density[part_id]:
                self.obj.sph[part_id].density = self.obj.rest_density[part_id]
    
    @ti.kernel
    def compute_pressure(self):
        for part_id in range(self.obj.tiGet_stack_top()[None]):
            self.obj.pressure[part_id] = self.obj.sph_wc[part_id].B * ((self.obj.sph[part_id].density / self.obj.rest_density[part_id]) ** self.gamma - 1)

    @ti.func
    def inloop_add_acc_pressure(self, part_id: ti.i32, neighb_part_id: ti.i32, neighb_part_shift: ti.i32, neighb_pool:ti.template(), neighb_obj:ti.template()):
        cached_dist = neighb_pool.tiGet_cachedDist(neighb_part_shift)
        cached_grad_W = neighb_pool.tiGet_cachedGradW(neighb_part_shift)
        if bigger_than_zero(cached_dist):
            acc_pressure = -neighb_obj.mass[neighb_part_id] * ((self.obj.pressure[part_id] / self.obj.sph[part_id].density**2) + (neighb_obj.pressure[neighb_part_id] / neighb_obj.sph[neighb_part_id].density**2)) * cached_grad_W
            self.obj.acc[part_id] += acc_pressure
    
    @ti.func
    def inloop_add_acc_number_density_pressure(self, part_id: ti.i32, neighb_part_id: ti.i32, neighb_part_shift: ti.i32, neighb_pool:ti.template(), neighb_obj:ti.template()):
        cached_dist = neighb_pool.tiGet_cachedDist(neighb_part_shift)
        cached_grad_W = neighb_pool.tiGet_cachedGradW(neighb_part_shift)
        if bigger_than_zero(cached_dist):
            part_volume2 = (self.obj.mass[part_id] / self.obj.sph[part_id].density)**2
            neighb_part_volume2 = (neighb_obj.mass[neighb_part_id] / neighb_obj.sph[neighb_part_id].density)**2
            acc_pressure = - (self.obj.pressure[part_id]*part_volume2 + neighb_obj.pressure[neighb_part_id]*neighb_part_volume2) / self.obj.mass[part_id] * cached_grad_W 
            self.obj.acc[part_id] += acc_pressure
                