import taichi as ti
import math
from .sph_funcs import *
from .Solver import Solver
from ..basic_op.type import *
from ..basic_obj.Obj_Particle import Particle
from typing import List

@ti.data_oriented
class DF_solver(Solver):
    def __init__(self, obj: Particle, incompressible_threshold: ti.f32 = 1e-4, div_free_threshold: ti.f32 = 1e-3, incompressible_iter_max: ti.i32 = 100, div_free_iter_max: ti.i32 = 50, incomp_warm_start: bool = False, div_warm_start: bool = False):
        
        super().__init__(obj)
        
        self.incompressible_threshold = val_f(incompressible_threshold)
        self.div_free_threshold = val_f(div_free_threshold)
        self.incompressible_iter_max = val_i(incompressible_iter_max)
        self.div_free_iter_max = val_i(div_free_iter_max)

        self.incomp_warm_start = incomp_warm_start
        self.div_warm_start = div_warm_start

        self.compressible_ratio = val_f(1)

        self.incompressible_iter = val_i(0)
        self.div_free_iter = val_i(0)
        self.reduction_buffer = ti.field(dtype=ti.f32, shape=self.getObj().getPartNum())
    
    @ti.func
    def inloop_compute_u_alpha_1_2(self, part_id: ti.i32, neighb_part_id: ti.i32, neighb_part_shift: ti.i32, neighb_pool:ti.template(), neighb_obj:ti.template()):
        cached_dist = neighb_pool.tiGet_cachedDist(neighb_part_shift)
        cached_grad_W = neighb_pool.tiGet_cachedGradW(neighb_part_shift)
        if bigger_than_zero(cached_dist):
            self.tiGetObj().sph_df[part_id].alpha_1 += neighb_obj.mass[neighb_part_id] * cached_grad_W
            self.tiGetObj().sph_df[part_id].alpha_2 += cached_grad_W.dot(cached_grad_W)

    @ti.func
    def inloop_accumulate_alpha_1(self, part_id: ti.i32, neighb_part_id: ti.i32, neighb_part_shift: ti.i32, neighb_pool:ti.template(), neighb_obj:ti.template()):
        cached_dist = neighb_pool.tiGet_cachedDist(neighb_part_shift)
        cached_grad_W = neighb_pool.tiGet_cachedGradW(neighb_part_shift)
        if bigger_than_zero(cached_dist):
            self.tiGetObj().sph_df[part_id].alpha_1 += neighb_obj.mass[neighb_part_id] * cached_grad_W

    @ti.func
    def inloop_accumulate_beta_1(self, part_id: ti.i32, neighb_part_id: ti.i32, neighb_part_shift: ti.i32, neighb_pool:ti.template(), neighb_obj:ti.template()):
        cached_dist = neighb_pool.tiGet_cachedDist(neighb_part_shift)
        cached_grad_W = neighb_pool.tiGet_cachedGradW(neighb_part_shift)
        if bigger_than_zero(cached_dist):
            self.tiGetObj().sph_df[part_id].alpha_1 += neighb_obj.volume[neighb_part_id] * cached_grad_W

    @ti.func
    def inloop_accumulate_alpha_2(self, part_id: ti.i32, neighb_part_id: ti.i32, neighb_part_shift: ti.i32, neighb_pool:ti.template(), neighb_obj:ti.template()):
        cached_dist = neighb_pool.tiGet_cachedDist(neighb_part_shift)
        cached_grad_W = neighb_pool.tiGet_cachedGradW(neighb_part_shift)
        if bigger_than_zero(cached_dist):
            self.tiGetObj().sph_df[part_id].alpha_2 += cached_grad_W.dot(cached_grad_W) * neighb_obj.mass[neighb_part_id]

    @ti.func
    def inloop_accumulate_beta_2(self, part_id: ti.i32, neighb_part_id: ti.i32, neighb_part_shift: ti.i32, neighb_pool:ti.template(), neighb_obj:ti.template()):
        cached_dist = neighb_pool.tiGet_cachedDist(neighb_part_shift)
        cached_grad_W = neighb_pool.tiGet_cachedGradW(neighb_part_shift)
        if bigger_than_zero(cached_dist):
            self.tiGetObj().sph_df[part_id].alpha_2 += cached_grad_W.dot(cached_grad_W) * neighb_obj.volume[neighb_part_id] * neighb_obj.volume[neighb_part_id] / neighb_obj.mass[neighb_part_id]

    @ti.kernel
    def ker_compute_alpha(self):
        for part_id in range(self.tiGetObj().tiGetStackTop()):
            self.tiGetObj().sph_df[part_id].alpha = self.tiGetObj().sph_df[part_id].alpha_1.dot(self.tiGetObj().sph_df[part_id].alpha_1) / self.tiGetObj().mass[part_id] + self.tiGetObj().sph_df[part_id].alpha_2
            if not bigger_than_zero(self.tiGetObj().sph_df[part_id].alpha):
                self.tiGetObj().sph_df[part_id].alpha = make_bigger_than_zero()

    @ti.kernel
    def compute_delta_density(self):
        for part_id in range(self.tiGetObj().tiGetStackTop()):
            self.tiGetObj().sph_df[part_id].delta_density = self.tiGetObj().sph[part_id].density - self.tiGetObj().rest_density[part_id]
    
    @ti.kernel
    def compute_delta_compression_ratio(self):
        for part_id in range(self.tiGetObj().tiGetStackTop()):
            self.tiGetObj().sph_df[part_id].delta_compression_ratio = self.tiGetObj().sph[part_id].compression_ratio - 1
    
    @ti.kernel
    def ReLU_delta_density(self):
        for part_id in range(self.tiGetObj().tiGetStackTop()):
            if self.tiGetObj().sph_df[part_id].delta_density < 0:
                self.tiGetObj().sph_df[part_id].delta_density = 0

    @ti.kernel
    def ReLU_delta_compression_ratio(self):
        for part_id in range(self.tiGetObj().tiGetStackTop()):
            if self.tiGetObj().sph_df[part_id].delta_compression_ratio < 0:
                self.tiGetObj().sph_df[part_id].delta_compression_ratio = 0

    @ti.func
    def inloop_update_delta_density_from_vel_adv(self, part_id: ti.i32, neighb_part_id: ti.i32, neighb_part_shift: ti.i32, neighb_pool:ti.template(), neighb_obj:ti.template()):
        cached_dist = neighb_pool.tiGet_cachedDist(neighb_part_shift)
        cached_grad_W = neighb_pool.tiGet_cachedGradW(neighb_part_shift)
        if bigger_than_zero(cached_dist):
            self.tiGetObj().sph_df[part_id].delta_density += cached_grad_W.dot(self.tiGetObj().sph_df[part_id].vel_adv-neighb_obj.sph_df[neighb_part_id].vel_adv) * neighb_obj.mass[neighb_part_id] * self.tiGetObj().tiGetWorld().tiGetDt()

    @ti.func
    def inloop_update_delta_compression_ratio_from_vel_adv(self, part_id: ti.i32, neighb_part_id: ti.i32, neighb_part_shift: ti.i32, neighb_pool:ti.template(), neighb_obj:ti.template()):
        cached_dist = neighb_pool.tiGet_cachedDist(neighb_part_shift)
        cached_grad_W = neighb_pool.tiGet_cachedGradW(neighb_part_shift)
        if bigger_than_zero(cached_dist):
            self.tiGetObj().sph_df[part_id].delta_compression_ratio += cached_grad_W.dot(self.tiGetObj().sph_df[part_id].vel_adv-neighb_obj.sph_df[neighb_part_id].vel_adv) * neighb_obj.volume[neighb_part_id] * self.tiGetObj().tiGetWorld().tiGetDt()

    @ti.func
    def inloop_update_vel_adv_from_alpha(self, part_id: ti.i32, neighb_part_id: ti.i32, neighb_part_shift: ti.i32, neighb_pool:ti.template(), neighb_obj:ti.template()):
        cached_dist = neighb_pool.tiGet_cachedDist(neighb_part_shift)
        cached_grad_W = neighb_pool.tiGet_cachedGradW(neighb_part_shift)
        if bigger_than_zero(cached_dist):
            self.tiGetObj().sph_df[part_id].vel_adv += self.neg_inv_dt[None] * cached_grad_W / self.tiGetObj().mass[part_id] \
                * ((self.tiGetObj().sph_df[part_id].delta_density * neighb_obj.mass[neighb_part_id] / self.tiGetObj().sph_df[part_id].alpha) \
                   + (neighb_obj.sph_df[neighb_part_id].delta_density * self.tiGetObj().mass[part_id] / neighb_obj.sph_df[neighb_part_id].alpha))

    @ti.kernel
    def compute_kappa_incomp_from_delta_density(self):
        for part_id in range(self.tiGetObj().tiGetStackTop()):
            self.tiGetObj().sph_df[part_id].kappa_incomp = self.tiGetObj().sph_df[part_id].delta_density / self.tiGetObj().sph_df[part_id].alpha * self.tiGetObj().tiGetSolverSPH().inv_dt2[None] / self.tiGetObj().volume[part_id]
            self.tiGetObj().sph_df[part_id].alpha_2 += self.tiGetObj().sph_df[part_id].kappa_incomp
    
    @ti.kernel
    def compute_kappa_incomp_from_delta_compression_ratio(self):
        for part_id in range(self.tiGetObj().tiGetStackTop()):
            self.tiGetObj().sph_df[part_id].kappa_incomp = self.tiGetObj().sph_df[part_id].delta_compression_ratio / self.tiGetObj().sph_df[part_id].alpha * self.tiGetObj().tiGetSolverSPH().inv_dt2[None] / self.tiGetObj().volume[part_id]
            self.tiGetObj().sph_df[part_id].alpha_2 += self.tiGetObj().sph_df[part_id].kappa_incomp

    @ti.kernel
    def log_kappa_incomp(self):
        for part_id in range(self.tiGetObj().tiGetStackTop()):
            self.tiGetObj().sph_df[part_id].kappa_incomp = self.tiGetObj().sph_df[part_id].alpha_2

    @ti.kernel
    def compute_kappa_div_from_delta_density(self):
        for part_id in range(self.tiGetObj().tiGetStackTop()):
            self.tiGetObj().sph_df[part_id].kappa_div = self.tiGetObj().sph_df[part_id].delta_density / self.tiGetObj().sph_df[part_id].alpha * self.tiGetObj().tiGetSolverSPH().inv_dt2[None] / self.tiGetObj().volume[part_id]
            self.tiGetObj().sph_df[part_id].alpha_2 += self.tiGetObj().sph_df[part_id].kappa_div

    @ti.kernel
    def compute_kappa_div_from_delta_compression_ratio(self):
        for part_id in range(self.tiGetObj().tiGetStackTop()):
            self.tiGetObj().sph_df[part_id].kappa_div = self.tiGetObj().sph_df[part_id].delta_compression_ratio / self.tiGetObj().sph_df[part_id].alpha * self.tiGetObj().tiGetSolverSPH().inv_dt2[None] / self.tiGetObj().volume[part_id]
            self.tiGetObj().sph_df[part_id].alpha_2 += self.tiGetObj().sph_df[part_id].kappa_div

    @ti.kernel
    def log_kappa_div(self):
        for part_id in range(self.tiGetObj().tiGetStackTop()):
            self.tiGetObj().sph_df[part_id].kappa_div = self.tiGetObj().sph_df[part_id].alpha_2

    @ti.func
    def inloop_df_update_vel_adv_from_kappa_incomp(self, part_id: ti.i32, neighb_part_id: ti.i32, neighb_part_shift: ti.i32, neighb_pool:ti.template(), neighb_obj:ti.template()):
        cached_dist = neighb_pool.tiGet_cachedDist(neighb_part_shift)
        cached_grad_W = neighb_pool.tiGet_cachedGradW(neighb_part_shift)
        if bigger_than_zero(cached_dist):
            self.tiGetObj().sph_df[part_id].vel_adv -= self.tiGetObj().tiGetWorld().tiGetDt() * \
                (neighb_obj.mass[neighb_part_id] / self.tiGetObj().mass[part_id] * self.tiGetObj().volume[part_id] * self.tiGetObj().sph_df[part_id].kappa_incomp + 
                 neighb_obj.volume[neighb_part_id] * neighb_obj.sph_df[neighb_part_id].kappa_incomp) \
            * cached_grad_W
    
    @ti.func
    def inloop_df_update_vel_adv_from_kappa_div(self, part_id: ti.i32, neighb_part_id: ti.i32, neighb_part_shift: ti.i32, neighb_pool:ti.template(), neighb_obj:ti.template()):
        cached_dist = neighb_pool.tiGet_cachedDist(neighb_part_shift)
        cached_grad_W = neighb_pool.tiGet_cachedGradW(neighb_part_shift)
        if bigger_than_zero(cached_dist):
            self.tiGetObj().sph_df[part_id].vel_adv -= self.tiGetObj().tiGetWorld().tiGetDt() * \
                (neighb_obj.mass[neighb_part_id] / self.tiGetObj().mass[part_id] * self.tiGetObj().volume[part_id] * self.tiGetObj().sph_df[part_id].kappa_div + 
                 neighb_obj.volume[neighb_part_id] * neighb_obj.sph_df[neighb_part_id].kappa_div) \
            * cached_grad_W

    @ti.func
    def inloop_vf_update_vel_adv_from_kappa_incomp(self, part_id: ti.i32, neighb_part_id: ti.i32, neighb_part_shift: ti.i32, neighb_pool:ti.template(), neighb_obj:ti.template()):
        cached_dist = neighb_pool.tiGet_cachedDist(neighb_part_shift)
        cached_grad_W = neighb_pool.tiGet_cachedGradW(neighb_part_shift)
        if bigger_than_zero(cached_dist):
            self.tiGetObj().sph_df[part_id].vel_adv -= self.tiGetObj().tiGetWorld().tiGetDt() * self.tiGetObj().volume[part_id] * neighb_obj.volume[neighb_part_id] / self.tiGetObj().mass[part_id] * \
            (self.tiGetObj().sph_df[part_id].kappa_incomp + neighb_obj.sph_df[neighb_part_id].kappa_incomp) * \
            cached_grad_W

    @ti.func
    def inloop_vf_update_vel_adv_from_kappa_div(self, part_id: ti.i32, neighb_part_id: ti.i32, neighb_part_shift: ti.i32, neighb_pool:ti.template(), neighb_obj:ti.template()):
        cached_dist = neighb_pool.tiGet_cachedDist(neighb_part_shift)
        cached_grad_W = neighb_pool.tiGet_cachedGradW(neighb_part_shift)
        if bigger_than_zero(cached_dist):
            self.tiGetObj().sph_df[part_id].vel_adv -= self.tiGetObj().tiGetWorld().tiGetDt() * self.tiGetObj().volume[part_id] * neighb_obj.volume[neighb_part_id] / self.tiGetObj().mass[part_id] * \
            (self.tiGetObj().sph_df[part_id].kappa_div + neighb_obj.sph_df[neighb_part_id].kappa_div) * \
            cached_grad_W

    @ti.kernel 
    def update_df_compressible_ratio(self):
        self.compressible_ratio[None] = 0
        for part_id in range(self.tiGetObj().tiGetStackTop()):
            self.compressible_ratio[None] += self.tiGetObj().sph_df[part_id].delta_density / self.tiGetObj().rest_density[part_id]
        self.compressible_ratio[None] /= self.tiGetObj().tiGetStackTop()

    @ti.kernel 
    def update_vf_compressible_ratio(self):

        # for part_id in range(self.tiGetObj().tiGetStackTop()):
        #     self.reduction_buffer[part_id] = self.tiGetObj().sph_df[part_id].delta_compression_ratio
        # stride = 1
        # while stride < self.tiGetObj().tiGetStackTop():
        #     for i in range(0, self.tiGetObj().tiGetStackTop()//(stride*2)):
        #         self.reduction_buffer[i * 2 * stride] += self.reduction_buffer[(i * 2 * stride) + stride]
        #     stride *= 2
        # self.compressible_ratio[None] = self.reduction_buffer[0] / self.tiGetObj().tiGetStackTop()
        
        for part_id in range(self.tiGetObj().tiGetStackTop()):
            self.compressible_ratio[None] += self.tiGetObj().sph_df[part_id].delta_compression_ratio
        self.compressible_ratio[None] /= self.tiGetObj().tiGetStackTop()

        # for part_id in range(self.tiGetObj().tiGetStackTop()):
        #     if self.tiGetObj().sph_df[part_id].delta_compression_ratio > self.incompressible_threshold[None]:
        #         self.compressible_ratio[None] = 1.0

    @ti.kernel
    def update_vel(self, out_vel: ti.template()):
        for part_id in range(self.tiGetObj().tiGetStackTop()):
            out_vel[part_id] = self.tiGetObj().sph_df[part_id].vel_adv

    @ti.kernel
    def set_vel_adv(self, in_vel_adv: ti.template()):
        for part_id in range(self.tiGetObj().tiGetStackTop()):
            self.tiGetObj().sph_df[part_id].vel_adv = in_vel_adv[part_id]

    # @ti.kernel
    # def get_acc_pressure_1of2(self):
    #     for part_id in range(self.tiGetObj().tiGetStackTop()):
    #         self.tiGetObj().mixture[part_id].acc_pressure = self.tiGetObj().vel[part_id]
    
    @ti.kernel
    def get_acc_pressure(self):
        for part_id in range(self.tiGetObj().tiGetStackTop()):
            self.tiGetObj().mixture[part_id].acc_pressure = (self.tiGetObj().sph_df[part_id].vel_adv - self.tiGetObj().vel[part_id]) / self.tiGetObj().tiGetWorld().tiGetDt()

    def compute_alpha(self, neighb_pool:ti.template()):
    
        self.getObj().clear(self.getObj().sph_df.alpha_1)
        self.getObj().clear(self.getObj().sph_df.alpha_2)

        for neighb_obj in self.getObj().get_module_neighbSearch().neighb_obj_list:
            ''' Compute Alpha_1, Alpha_2 ''' 
            if self.getObj().m_is_dynamic:
                self.getObj().get_module_neighbSearch().loop_neighb(neighb_obj, self.inloop_accumulate_alpha_1)
                if neighb_obj.m_is_dynamic:
                    self.getObj().get_module_neighbSearch().loop_neighb(neighb_obj, self.inloop_accumulate_alpha_2)
            else: 
                if neighb_obj.m_is_dynamic:
                    self.getObj().get_module_neighbSearch().loop_neighb(neighb_obj, self.inloop_accumulate_alpha_2)

        ''' Compute Alpha '''
        self.ker_compute_alpha()          

    def compute_beta(self, neighb_pool:ti.template()):
    
        self.getObj().clear(self.getObj().sph_df.alpha_1)
        self.getObj().clear(self.getObj().sph_df.alpha_2)

        for neighb_obj in self.getObj().get_module_neighbSearch().neighb_obj_list:
            ''' Compute Alpha_1, Alpha_2 ''' 
            # self.getObj().get_module_neighbSearch().loop_neighb(neighb_obj, self.inloop_accumulate_beta_1)
            # self.getObj().get_module_neighbSearch().loop_neighb(neighb_obj, self.inloop_accumulate_beta_2)
            if self.getObj().m_is_dynamic:
                self.getObj().get_module_neighbSearch().loop_neighb(neighb_obj, self.inloop_accumulate_beta_1)
                if neighb_obj.m_is_dynamic:
                    self.getObj().get_module_neighbSearch().loop_neighb(neighb_obj, self.inloop_accumulate_beta_2)
            else: 
                if neighb_obj.m_is_dynamic:
                    self.getObj().get_module_neighbSearch().loop_neighb(neighb_obj, self.inloop_accumulate_beta_2)

        ''' Compute Alpha '''
        self.ker_compute_alpha()     




                