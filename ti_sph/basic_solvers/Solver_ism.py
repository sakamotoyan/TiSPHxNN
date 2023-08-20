import taichi as ti
import math
from .sph_funcs import *
from .Solver_sph import SPH_solver
from .Neighb_looper import Neighb_looper
from ..basic_op.type import *
from ..basic_obj.Obj_Particle import Particle
from typing import List

@ti.data_oriented
class Implicit_mixture_solver(SPH_solver):
    def __init__(self, obj: Particle, Cd: ti.f32, world):
        
        super().__init__(obj)

        self.obj = obj
        self.Cd = Cd
        self.phase_num = world.g_phase_num
        self.world = world
        
    @ti.kernel
    def update_vel_from_phase_vel(self):
        for part_id in range(self.obj.ti_get_stack_top()[None]):
            self.obj.vel[part_id] *= 0
            for phase_id in range(self.phase_num[None]):
                if not bigger_than_zero(self.obj.phase.val_frac[part_id, phase_id]):
                    self.obj.phase.val_frac[part_id, phase_id] = 0
                self.obj.vel[part_id] += self.obj.phase.vel[part_id, phase_id] * self.obj.phase.val_frac[part_id, phase_id]
            for phase_id in range(self.phase_num[None]):
                if not bigger_than_zero(self.obj.phase.val_frac[part_id, phase_id]):
                    self.obj.phase.vel[part_id, phase_id] = self.obj.vel[part_id]
                self.obj.phase.drift_vel[part_id, phase_id] = self.obj.phase.vel[part_id, phase_id] - self.obj.vel[part_id]

    # @ti.kernel
    # def update_drift_vel(self):
    #     for part_id in range(self.obj.ti_get_stack_top()[None]):
    #         for phase_id in range(self.phase_num[None]):
    #             self.obj.phase.drift_vel[part_id, phase_id] = self.obj.phase.vel[part_id, phase_id] - self.obj.vel[part_id]

    @ti.kernel
    def zero_drift(self):
        for part_id in range(self.obj.ti_get_stack_top()[None]):
            for phase_id in range(self.phase_num[None]):
                self.obj.phase.drift_vel[part_id, phase_id] *= 0
                self.obj.phase.vel[part_id, phase_id] = self.obj.vel[part_id]

    @ti.kernel
    def update_rest_density_and_mass(self):
        for part_id in range(self.obj.ti_get_stack_top()[None]):
            rest_density = 0
            for phase_id in range(self.phase_num[None]):
                rest_density += self.obj.phase.val_frac[part_id, phase_id] * self.world.g_phase_rest_density[None][phase_id]
            self.obj.rest_density[part_id] = rest_density
            self.obj.mass[part_id] = self.obj.rest_density[part_id] * self.obj.volume[part_id]
    
    @ti.kernel
    def clear_phase_acc(self):
        for part_id in range(self.obj.ti_get_stack_top()[None]):
            for phase_id in range(self.phase_num[None]):
                self.obj.phase.acc[part_id, phase_id] *= 0
                # self.obj.phase.val_frac_tmp[part_id, phase_id] = self.obj.phase.val_frac[part_id, phase_id]
    
    @ti.kernel
    def add_phase_acc_gravity(self):
        for part_id in range(self.obj.ti_get_stack_top()[None]):
            for phase_id in range(self.phase_num[None]):
                self.obj.phase.acc[part_id, phase_id] += self.world.g_gravity[None]

    @ti.kernel
    def ditribute_acc_pressure_2_phase(self):
        for part_id in range(self.obj.ti_get_stack_top()[None]):
            for phase_id in range(self.phase_num[None]):
                self.obj.phase.acc[part_id, phase_id] += self.obj.mixture.acc_pressure[part_id] * \
                    (self.Cd + ((1 - self.Cd) * (self.obj.rest_density[part_id]/self.world.g_phase_rest_density[None][phase_id])))
                
    @ti.kernel
    def phase_acc_2_phase_vel(self):
        for part_id in range(self.obj.ti_get_stack_top()[None]):
            for phase_id in range(self.phase_num[None]):
                self.obj.phase.vel[part_id, phase_id] += self.obj.phase.acc[part_id, phase_id] * self.world.g_dt[None]

    @ti.kernel
    def phase_vel_2_phase_phase_vel_adv(self):
        for part_id in range(self.obj.ti_get_stack_top()[None]):
            for phase_id in range(self.phase_num[None]):
                self.obj.phase.vel_adv[part_id, phase_id] = self.obj.phase.vel[part_id, phase_id]
    
    @ti.kernel
    def clear_val_frac_tmp(self):
        for part_id in range(self.obj.ti_get_stack_top()[None]):
            for phase_id in range(self.phase_num[None]):
                self.obj.phase.val_frac_tmp[part_id, phase_id] = 0

    @ti.func
    def inloop_update_phase_change(self, part_id: ti.i32, neighb_part_id: ti.i32, neighb_part_shift: ti.i32, neighb_pool:ti.template(), neighb_obj:ti.template()):
        cached_dist = neighb_pool.cached_neighb_attributes[neighb_part_shift].dist
        cached_grad_W = neighb_pool.cached_neighb_attributes[neighb_part_shift].grad_W
        if bigger_than_zero(cached_dist) and (self.obj.mixture[part_id].flag_negative_val_frac == 0 and neighb_obj.mixture[neighb_part_id].flag_negative_val_frac == 0):
            for phase_id in range(self.phase_num[None]):
                self.obj.phase.val_frac_tmp[part_id, phase_id] -= self.dt[None] * neighb_obj.volume[neighb_part_id] * \
                (self.obj.phase.val_frac[part_id, phase_id] * self.obj.phase.drift_vel[part_id, phase_id] + \
                 neighb_obj.phase.val_frac[neighb_part_id, phase_id] * neighb_obj.phase.drift_vel[neighb_part_id, phase_id]).dot(cached_grad_W)

    @ti.kernel
    def check_negative(self) -> ti.i32:
        all_positive = 1
        for part_id in range(self.obj.ti_get_stack_top()[None]):
            if self.obj.mixture[part_id].flag_negative_val_frac == 0:
                for phase_id in range(self.phase_num[None]):
                    if self.obj.phase.val_frac_tmp[part_id, phase_id] + self.obj.phase.val_frac[part_id, phase_id] < 0:
                        self.obj.mixture[part_id].flag_negative_val_frac = 1
                        all_positive = 0
        return all_positive
    
    @ti.kernel
    def update_phase_change(self):
        for part_id in range(self.obj.ti_get_stack_top()[None]):
            if self.obj.mixture[part_id].flag_negative_val_frac == 0:
                for phase_id in range(self.phase_num[None]):
                    self.obj.phase.val_frac[part_id, phase_id] += self.obj.phase.val_frac_tmp[part_id, phase_id]
            else:
                self.obj.mixture[part_id].flag_negative_val_frac = 0
                for phase_id in range(self.phase_num[None]):
                    self.obj.phase.drift_vel[part_id, phase_id] *= 0
                    self.obj.phase.vel[part_id, phase_id] = self.obj.vel[part_id]

    @ti.kernel
    def update_color(self):
        for part_id in range(self.obj.ti_get_stack_top()[None]):
            color = ti.Vector([0.0, 0.0, 0.0])
            for phase_id in range(self.phase_num[None]):
                for rgb_id in ti.static(range(3)):
                    color[rgb_id] += self.obj.phase.val_frac[part_id, phase_id] * self.world.g_phase_color[phase_id][rgb_id]
            for rgb_id in range(self.phase_num[None]):
                color[rgb_id] = ti.min(color[rgb_id], 1.0)
            self.obj.rgb[part_id] = color