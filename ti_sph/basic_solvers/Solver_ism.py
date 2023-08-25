import taichi as ti
import math
from .sph_funcs import *
from .Solver_sph import SPH_solver
from .Neighb_looper import Neighb_looper
from .Solver_multiphase import Multiphase_solver
from ..basic_op.type import *
from ..basic_obj.Obj_Particle import Particle

from typing import List

GREEN = ti.Vector([0.0, 1.0, 0.0])
WHITE = ti.Vector([1.0, 1.0, 1.0])
DARK = ti.Vector([0.0, 0.0, 0.0])

@ti.data_oriented
class Implicit_mixture_solver(Multiphase_solver):
    def __init__(self, obj: Particle, Cd: ti.f32, Cf: ti.f32, k_vis_inter: ti.f32, k_vis_inner: ti.f32, world):
        
        super().__init__(obj, Cf, world)

        self.Cd = Cd
        self.k_vis_inter = k_vis_inter
        self.k_vis_inner = k_vis_inner

        
    @ti.kernel
    def update_vel_from_phase_vel(self):
        for part_id in range(self.obj.ti_get_stack_top()[None]):
            self.obj.vel[part_id] *= 0
            for phase_id in range(self.phase_num[None]):
                # if bigger_than_zero(self.obj.phase.val_frac[part_id, phase_id]):
                self.obj.vel[part_id] += self.obj.phase.vel[part_id, phase_id] * self.obj.phase.val_frac[part_id, phase_id]
            for phase_id in range(self.phase_num[None]):
                # if not bigger_than_zero(self.obj.phase.val_frac[part_id, phase_id]):
                #     self.obj.phase.vel[part_id, phase_id] = self.obj.vel[part_id]
                self.obj.phase.drift_vel[part_id, phase_id] = self.obj.phase.vel[part_id, phase_id] - self.obj.vel[part_id]

    @ti.kernel
    def regularize_val_frac(self):
        for part_id in range(self.obj.ti_get_stack_top()[None]):
            frac_sum = 0.0
            for phase_id in range(self.phase_num[None]):
                frac_sum += self.obj.phase.val_frac[part_id, phase_id]
            for phase_id in range(self.phase_num[None]):
                self.obj.phase.val_frac[part_id, phase_id] /= frac_sum

    @ti.kernel
    def zero_out_drift_vel(self):
        for part_id in range(self.obj.ti_get_stack_top()[None]):
            for phase_id in range(self.phase_num[None]):
                self.obj.phase.drift_vel[part_id, phase_id] *= 0
                # self.obj.phase.vel[part_id, phase_id] = self.obj.vel[part_id]

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
    
    @ti.kernel
    def copy_val_frac(self):
        for part_id in range(self.obj.ti_get_stack_top()[None]):
            for phase_id in range(self.phase_num[None]):
                self.obj.phase.val_frac_tmp[part_id, phase_id] = self.obj.phase.val_frac[part_id, phase_id]
                self.obj.phase.val_frac_tmp2[part_id, phase_id] = self.obj.phase.val_frac[part_id, phase_id]

    @ti.func
    def inloop_update_phase_change_from_drift(self, part_id: ti.i32, neighb_part_id: ti.i32, neighb_part_shift: ti.i32, neighb_pool:ti.template(), neighb_obj:ti.template()):
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
    def release_negative(self):
        for part_id in range(self.obj.ti_get_stack_top()[None]):
            self.obj.mixture[part_id].flag_negative_val_frac = 0

    @ti.kernel
    def release_unused_drift_vel(self):
        for part_id in range(self.obj.ti_get_stack_top()[None]):
            if not self.obj.mixture[part_id].flag_negative_val_frac == 0:
                for phase_id in range(self.phase_num[None]):
                    self.obj.phase.vel[part_id, phase_id] = self.obj.vel[part_id]

    def update_phase_change(self):
        self.update_phase_change_ker()
        # self.clear_phase_acc()
        # self.copy_val_frac()
        # self.loop_neighb(self.obj.m_neighb_search.neighb_pool, self.obj, self.inloop_update_phase_change_from_all)
        # self.re_arrange_phase_vel()

    @ti.kernel
    def update_phase_change_ker(self):
        for part_id in range(self.obj.ti_get_stack_top()[None]):
            for phase_id in range(self.phase_num[None]):
                self.obj.phase.val_frac[part_id, phase_id] += self.obj.phase.val_frac_tmp[part_id, phase_id]
    
    @ti.func
    def inloop_update_phase_change_from_all(self, part_id: ti.i32, neighb_part_id: ti.i32, neighb_part_shift: ti.i32, neighb_pool:ti.template(), neighb_obj:ti.template()):
        cached_dist = neighb_pool.cached_neighb_attributes[neighb_part_shift].dist
        cached_grad_W = neighb_pool.cached_neighb_attributes[neighb_part_shift].grad_W
        if bigger_than_zero(cached_dist) and (self.obj.mixture[part_id].flag_negative_val_frac == 0 and neighb_obj.mixture[neighb_part_id].flag_negative_val_frac == 0):
            for phase_id in range(self.phase_num[None]):
                val_frac_ij = self.obj.phase.val_frac[part_id, phase_id] - neighb_obj.phase.val_frac[neighb_part_id, phase_id]
                x_ij = self.obj.pos[part_id] - neighb_obj.pos[neighb_part_id]
                diffuse_val_change = self.dt[None] * self.Cf * val_frac_ij * neighb_obj.volume[neighb_part_id] * cached_grad_W.dot(x_ij) / (cached_dist**2)
                drift_val_change = -self.dt[None] * neighb_obj.volume[neighb_part_id] * \
                    (self.obj.phase.val_frac[part_id, phase_id] * self.obj.phase.drift_vel[part_id, phase_id] + \
                    neighb_obj.phase.val_frac[neighb_part_id, phase_id] * neighb_obj.phase.drift_vel[neighb_part_id, phase_id]).dot(cached_grad_W)
                val_frac_change = diffuse_val_change + drift_val_change
                self.obj.phase.val_frac_tmp[part_id, phase_id] += val_frac_change
                if val_frac_change > 0:
                    self.obj.phase.acc[part_id, phase_id] += val_frac_change * neighb_obj.phase.vel[neighb_part_id, phase_id]
                else:
                    self.obj.phase.val_frac_tmp2[part_id, phase_id] += val_frac_change

    @ti.kernel
    def re_arrange_phase_vel(self):
        for part_id in range(self.obj.ti_get_stack_top()[None]):
            if self.obj.mixture[part_id].flag_negative_val_frac == 0:
                for phase_id in range(self.phase_num[None]):
                    self.obj.phase.val_frac[part_id, phase_id] = self.obj.phase.val_frac_tmp[part_id, phase_id]
                    if bigger_than_zero(self.obj.phase.val_frac_tmp2[part_id, phase_id]):
                        new_phase_vel = self.obj.phase.acc[part_id, phase_id] + (self.obj.phase.val_frac_tmp2[part_id, phase_id]*self.obj.phase.vel[part_id, phase_id])
                        self.obj.phase.vel[part_id, phase_id] = new_phase_vel / self.obj.phase.val_frac[part_id, phase_id]
                    else:
                        self.obj.phase.vel[part_id, phase_id] = self.obj.phase.acc[part_id, phase_id]/self.obj.phase.val_frac[part_id, phase_id]

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

    @ti.kernel
    def cfl_dt(self, cfl_factor: ti.f32, max_dt: ti.f32) -> ti.f32:
        max_vel = 0.0
        for part_id in range(self.obj.ti_get_stack_top()[None]):
            for phase_id in range(self.phase_num[None]):
                ti.atomic_max(max_vel, ti.math.length(self.obj.phase.vel[part_id, phase_id]))
        new_dt = ti.min(max_dt, self.world.g_part_size[None] / max_vel * cfl_factor)
        return new_dt
    
    @ti.kernel
    def draw_drift_vel(self, phase:ti.i32):
        max_vel = 0.0
        phase_id = phase
        for part_id in range(self.obj.ti_get_stack_top()[None]):
            length = ti.math.length(self.obj.phase.drift_vel[part_id, phase_id])/20
            self.obj.rgb[part_id] = ti.Vector([length, length, length])
            # ti.atomic_max(max_vel, ti.math.length(self.obj.phase.drift_vel[part_id, phase_id]))
    
    @ti.kernel
    def max_phase_vel(self) -> ti.f32:
        max_vel = 0.0
        for part_id in range(self.obj.ti_get_stack_top()[None]):
            for phase_id in range(self.phase_num[None]):
                ti.atomic_max(max_vel, ti.math.length(self.obj.phase.vel[part_id, phase_id]))
        return max_vel
    
    @ti.kernel
    def check_empty_phase(self):
        fact = 0.99999
        for part_id in range(self.obj.ti_get_stack_top()[None]):
            sum = 0.0
            for phase_id in range(self.phase_num[None]):
                sum += self.obj.phase.val_frac[part_id, phase_id]
            if sum < fact:
                # print('empty phase', part_id, sum)
                self.obj.rgb[part_id] = WHITE
            if sum > 2-fact:
                # print('empty phase', part_id, sum)
                self.obj.rgb[part_id] = DARK
    
    @ti.kernel
    def check_negative_phase(self):
        for part_id in range(self.obj.ti_get_stack_top()[None]):
            for phase_id in range(self.phase_num[None]):
                if self.obj.phase.val_frac[part_id, phase_id] < 0:
                    self.obj.rgb[part_id] = GREEN
                    print('negative phase', part_id, phase_id, self.obj.phase.val_frac[part_id, phase_id])
    
    @ti.kernel
    def check_val_frac(self):
        sum_phase_1 = 0.0
        sum_phase_2 = 0.0
        sum_phase_3 = 0.0
        for part_id in range(self.obj.ti_get_stack_top()[None]):
            sum_phase_1 += self.obj.phase.val_frac[part_id, 0]
            sum_phase_2 += self.obj.phase.val_frac[part_id, 1]
            sum_phase_3 += self.obj.phase.val_frac[part_id, 2]
        print('phase 1 total', sum_phase_1)
        # print('phase 2 total', sum_phase_2)
        print('phase 3 total', sum_phase_3)

    @ti.func
    def inloop_add_phase_acc_vis(self, part_id: ti.i32, neighb_part_id: ti.i32, neighb_part_shift: ti.i32, neighb_pool:ti.template(), neighb_obj:ti.template()):
        cached_dist = neighb_pool.cached_neighb_attributes[neighb_part_shift].dist
        cached_grad_W = neighb_pool.cached_neighb_attributes[neighb_part_shift].grad_W
        x_ij = self.obj.pos[part_id] - neighb_obj.pos[neighb_part_id]
        if bigger_than_zero(cached_dist):
            v_ij = self.obj.vel[part_id] - neighb_obj.vel[neighb_part_id]
            for phase_id in range(self.phase_num[None]):
                v_ki_mj = self.obj.phase.vel[part_id, phase_id] - neighb_obj.vel[neighb_part_id]
                self.obj.phase.acc[part_id, phase_id] += 2*(2+self.obj.m_world.g_dim[None]) * neighb_obj.volume[neighb_part_id] * \
                    ((self.k_vis_inner * (1-self.Cd) *  v_ki_mj) + (self.k_vis_inter * self.Cd * v_ij)).dot(x_ij) * cached_grad_W \
                    / (cached_dist**2) 