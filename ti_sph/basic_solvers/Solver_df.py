import taichi as ti
import math
from .sph_funcs import *
from .Solver_sph import SPH_solver
from .Neighb_looper import Neighb_looper
from ..basic_op.type import *
from ..basic_obj.Obj_Particle import Particle
from typing import List

@ti.data_oriented
class DF_solver(SPH_solver):
    def __init__(self, obj: Particle, incompressible_threshold: ti.f32 = 1e-4, div_free_threshold: ti.f32 = 1e-3, incompressible_iter_max: ti.i32 = 100, div_free_iter_max: ti.i32 = 50):
        
        super().__init__(obj)
        
        self.incompressible_threshold = val_f(incompressible_threshold)
        self.div_free_threshold = val_f(div_free_threshold)
        self.incompressible_iter_max = val_i(incompressible_iter_max)
        self.div_free_iter_max = val_i(div_free_iter_max)

        self.compressible_ratio = val_f(1)

        self.incompressible_iter = val_i(0)
        self.div_free_iter = val_i(0)
    
    @ti.func
    def inloop_compute_u_alpha_1_2(self, part_id: ti.i32, neighb_part_id: ti.i32, neighb_part_shift: ti.i32, neighb_pool:ti.template(), neighb_obj:ti.template()):
        cached_dist = neighb_pool.cached_neighb_attributes[neighb_part_shift].dist
        cached_grad_W = neighb_pool.cached_neighb_attributes[neighb_part_shift].grad_W
        if bigger_than_zero(cached_dist):
            self.obj.sph_df[part_id].alpha_1 += neighb_obj.mass[neighb_part_id] * cached_grad_W
            self.obj.sph_df[part_id].alpha_2 += cached_grad_W.dot(cached_grad_W)

    @ti.func
    def inloop_accumulate_alpha_1(self, part_id: ti.i32, neighb_part_id: ti.i32, neighb_part_shift: ti.i32, neighb_pool:ti.template(), neighb_obj:ti.template()):
        cached_dist = neighb_pool.cached_neighb_attributes[neighb_part_shift].dist
        cached_grad_W = neighb_pool.cached_neighb_attributes[neighb_part_shift].grad_W
        if bigger_than_zero(cached_dist):
            self.obj.sph_df[part_id].alpha_1 += neighb_obj.mass[neighb_part_id] * cached_grad_W

    @ti.func
    def inloop_accumulate_alpha_2(self, part_id: ti.i32, neighb_part_id: ti.i32, neighb_part_shift: ti.i32, neighb_pool:ti.template(), neighb_obj:ti.template()):
        cached_dist = neighb_pool.cached_neighb_attributes[neighb_part_shift].dist
        cached_grad_W = neighb_pool.cached_neighb_attributes[neighb_part_shift].grad_W
        if bigger_than_zero(cached_dist):
            self.obj.sph_df[part_id].alpha_2 += cached_grad_W.dot(cached_grad_W) * neighb_obj.mass[neighb_part_id]

    @ti.kernel
    def ker_compute_alpha(self):
        for part_id in range(self.obj.ti_get_stack_top()[None]):
            self.obj.sph_df[part_id].alpha = self.obj.sph_df[part_id].alpha_1.dot(self.obj.sph_df[part_id].alpha_1) / self.obj.mass[part_id] + self.obj.sph_df[part_id].alpha_2
            if not bigger_than_zero(self.obj.sph_df[part_id].alpha):
                self.obj.sph_df[part_id].alpha = make_bigger_than_zero()

    @ti.kernel
    def compute_delta_density(self):
        for part_id in range(self.obj.ti_get_stack_top()[None]):
            self.obj.sph_df[part_id].delta_density = self.obj.sph[part_id].density - self.obj.rest_density[part_id]
    
    @ti.kernel
    def ReLU_delta_density(self):
        for part_id in range(self.obj.ti_get_stack_top()[None]):
            if self.obj.sph_df[part_id].delta_density < 0:
                self.obj.sph_df[part_id].delta_density = 0

    @ti.func
    def inloop_update_delta_density_from_vel_adv(self, part_id: ti.i32, neighb_part_id: ti.i32, neighb_part_shift: ti.i32, neighb_pool:ti.template(), neighb_obj:ti.template()):
        cached_dist = neighb_pool.cached_neighb_attributes[neighb_part_shift].dist
        cached_grad_W = neighb_pool.cached_neighb_attributes[neighb_part_shift].grad_W
        if bigger_than_zero(cached_dist):
            self.obj.sph_df[part_id].delta_density += cached_grad_W.dot(self.obj.sph_df[part_id].vel_adv-neighb_obj.sph_df[neighb_part_id].vel_adv) * neighb_obj.mass[neighb_part_id] * self.dt[None]

    @ti.func
    def inloop_update_vel_adv_from_alpha(self, part_id: ti.i32, neighb_part_id: ti.i32, neighb_part_shift: ti.i32, neighb_pool:ti.template(), neighb_obj:ti.template()):
        cached_dist = neighb_pool.cached_neighb_attributes[neighb_part_shift].dist
        cached_grad_W = neighb_pool.cached_neighb_attributes[neighb_part_shift].grad_W
        if bigger_than_zero(cached_dist):
            self.obj.sph_df[part_id].vel_adv += self.neg_inv_dt[None] * cached_grad_W / self.obj.mass[part_id] \
                * ((self.obj.sph_df[part_id].delta_density * neighb_obj.mass[neighb_part_id] / self.obj.sph_df[part_id].alpha) \
                   + (neighb_obj.sph_df[neighb_part_id].delta_density * self.obj.mass[part_id] / neighb_obj.sph_df[neighb_part_id].alpha))

    @ti.kernel 
    def update_compressible_ratio(self):
        self.compressible_ratio[None] = 0
        for part_id in range(self.obj.ti_get_stack_top()[None]):
            self.compressible_ratio[None] += self.obj.sph_df[part_id].delta_density / self.obj.rest_density[part_id]
        self.compressible_ratio[None] /= self.obj.ti_get_stack_top()[None]

    @ti.kernel
    def update_vel(self, out_vel: ti.template()):
        for part_id in range(self.obj.ti_get_stack_top()[None]):
            out_vel[part_id] = self.obj.sph_df[part_id].vel_adv

    @ti.kernel
    def get_vel_adv(self, in_vel_adv: ti.template()):
        for part_id in range(self.obj.ti_get_stack_top()[None]):
            self.obj.sph_df[part_id].vel_adv = in_vel_adv[part_id]

    def compute_alpha(self, neighb_pool:ti.template()):
    
        self.obj.clear(self.obj.sph_df.alpha_1)
        self.obj.clear(self.obj.sph_df.alpha_2)

        for neighb_obj in neighb_pool.neighb_obj_list:
            ''' Compute Alpha_1, Alpha_2 ''' 
            if self.obj.m_is_dynamic:
                self.loop_neighb(neighb_pool, neighb_obj, self.inloop_accumulate_alpha_1)
                if neighb_obj.m_is_dynamic:
                    self.loop_neighb(neighb_pool, neighb_obj, self.inloop_accumulate_alpha_2)
            else: 
                if neighb_obj.m_is_dynamic:
                    self.loop_neighb(neighb_pool, neighb_obj, self.inloop_accumulate_alpha_2)

        ''' Compute Alpha '''
        self.ker_compute_alpha()            

# @ti.data_oriented
# class DF_layer:
#     def __init__(self, DF_solvers: List[DF_solver] = []):

#         self.df_solvers = DF_solvers
#         self.incompressible_states: List[bool] = [False for _ in range(len(DF_solvers))]
#         self.divergence_free_states: List[bool] = [False for _ in range(len(DF_solvers))]

#         for solver in DF_solvers:
#             '''check if solver is DF_solver'''
#             if not isinstance(solver, DF_solver):
#                 raise TypeError('DF_layer only accepts DF_solver')
    
#     def add_solver(self, solver: DF_solver):
#         if not isinstance(solver, DF_solver):
#             raise TypeError('DF_layer only accepts DF_solver')
#         self.df_solvers.append(solver)
    
#     def step(self):
#         for solver in self.df_solvers:
#             if solver.obj.m_is_dynamic:
#                 solver.get_vel_adv(solver.obj.vel_adv)
#                 self.incompressible_states[self.df_solvers.index(solver)] = False
#                 self.divergence_free_states[self.df_solvers.index(solver)] = False
#             else:
#                 self.incompressible_states[self.df_solvers.index(solver)] = True
#                 self.divergence_free_states[self.df_solvers.index(solver)] = True

#             solver.df_step_static_phase(solver.obj.m_neighb_search.neighb_pool)
            
#         while True:
#             for solver in self.df_solvers:
#                 solver.incompressible_iter[None] += 1

#                 solver.compute_delta_density()

#                 for neighb_obj in solver.obj.m_neighb_search.neighb_pool.neighb_obj_list:
#                     solver.loop_neighb(solver.obj.m_neighb_search.neighb_pool, neighb_obj, solver.inloop_update_delta_density_from_vel_adv)
#                 solver.ReLU_delta_density()
#                 solver.update_compressible_ratio()

#                 if solver.compressible_ratio[None] < solver.incompressible_threshold[None] \
#                     or solver.incompressible_iter[None] > solver.incompressible_iter_max[None]:
#                     self.incompressible_states[self.df_solvers.index(solver)] = True
#             if all(self.incompressible_states):
#                 break
        
#             for solver in self.df_solvers:
#                 if solver.obj.m_is_dynamic:
#                     for neighb_obj in solver.obj.m_neighb_search.neighb_pool.neighb_obj_list:
#                         solver.loop_neighb(solver.obj.m_neighb_search.neighb_pool, neighb_obj, solver.inloop_update_vel_adv_from_alpha)

#         for solver in self.df_solvers:
#             if solver.obj.m_is_dynamic:
#                 solver.update_vel(solver.obj.vel)
            


                