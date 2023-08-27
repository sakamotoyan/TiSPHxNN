import taichi as ti

from typing import List

from .modules import neighb_search
from .modules import solver_sph
from .modules import solver_adv
from .modules import solver_df
from .modules import cfl
from .modules import solver_wcsph

from ...basic_op.type import *
from ...basic_obj.Obj_Particle import Particle

@ti.data_oriented
class World:
    def __init__(self, dim=3, lb = -16, rt = 16):
        ''' GLOBAL CONFIGURATION '''
        self.g_dim = val_i(dim)
        self.g_space_lb = vecx_f(self.g_dim[None])
        self.g_space_rt = vecx_f(self.g_dim[None])
        self.g_gravity = vecx_f(self.g_dim[None])
        self.g_space_lb.fill(lb)
        self.g_space_rt.fill(rt)
        self.g_gravity[None][1] = -9.8
        self.g_dt = val_f(0.001)
        self.g_inv_dt = val_f(1/self.g_dt[None])
        self.g_neg_inv_dt = val_f(-1/self.g_dt[None])
        self.g_inv_dt2 = val_f(self.g_inv_dt[None] ** 2)
        self.g_part_size = val_f(0.1)
        self.g_avg_neighb_part_num = val_i(32)
        self.g_obj_num = val_i(3)
        self.g_sound_speed = val_f(100)

        self.dependent_init()
        self.part_obj_list = []

    # Functions: init related
    def dependent_init(self):
        self.space_size = vecx_f(self.g_dim[None])
        self.space_center = vecx_f(self.g_dim[None])
        self.space_size[None] = self.g_space_rt[None] - self.g_space_lb[None]
        self.space_center[None] = (self.g_space_rt[None] + self.g_space_lb[None]) / 2

        self.part_volume = val_f(self.g_part_size[None] ** self.g_dim[None])
        self.support_radius = val_f(self.g_part_size[None] * 2)

    def refresh(self):
        self.dependent_init()
    
    def set_part_size(self, size):
        self.g_part_size = val_f(size)
        self.refresh()
    
    def set_dt(self, dt):
        self.g_dt[None] = dt
        self.g_inv_dt[None] = 1/dt
        self.g_neg_inv_dt[None] = -1/dt
        self.g_inv_dt2[None] = self.g_inv_dt[None] ** 2

    def set_multiphase(self, phase_num, phase_color:List[vec3f], phase_rest_density:List[float]):
        self.g_phase_num = val_i(phase_num)
        self.g_phase_color = ti.Vector.field(3, dtype=ti.f32, shape=phase_num)
        self.g_phase_rest_density = vecx_f(phase_num)
        for i in range(phase_num):
            self.g_phase_color[i] = phase_color[i]
            self.g_phase_rest_density[None][i] = phase_rest_density[i]
        print('world.g_phase_num\n', self.g_phase_num[None])
        print('world.g_phase_color\n', self.g_phase_color.to_numpy())
        print('world.g_phase_rest_density\n', self.g_phase_rest_density.to_numpy())
    def add_part_obj(self, part_num, is_dynamic, size: ti.template()):
        obj = Particle(part_num, size, is_dynamic)
        self.part_obj_list.append(obj)
        obj.set_id(self.part_obj_list.index(obj))
        obj.set_world(self)
        return obj
    
    def init_modules(self):
        neighb_search.init_neighb_search(self)
        solver_sph.init_solver_sph(self)
        solver_adv.init_solver_adv(self)
        solver_df.init_solver_df(self)
        solver_wcsph.init_solver_wcsph(self)
        cfl.init_cfl(self)

    # Functions: neighbour search
    update_pos_in_neighb_search = neighb_search.update_pos_in_neighb_search
    update_neighbs = neighb_search.update_neighbs
    neighb_search = neighb_search.search_neighb

    # Functions: advection utils
    clear_acc = solver_adv.clear_acc
    add_acc_gravity = solver_adv.add_acc_gravity
    acc2vel_adv = solver_adv.acc2vel_adv
    acc2vel = solver_adv.acc2vel
    vel_adv2vel = solver_adv.vel_adv2vel
    update_pos_from_vel = solver_adv.update_pos_from_vel

    # Functions: SPH
    step_sph_compute_density = solver_sph.step_sph_compute_density
    step_sph_compute_number_density = solver_sph.step_sph_compute_number_density
    step_sph_compute_compression_ratio = solver_sph.step_sph_compute_compression_ratio

    # Functions: DFSPH
    step_df_compute_alpha = solver_df.step_df_compute_alpha
    step_df_compute_beta = solver_df.step_df_compute_beta
    step_df_incomp = solver_df.step_df_incomp
    step_df_div = solver_df.step_df_div
    step_dfsph_incomp = solver_df.step_dfsph_incomp
    step_dfsph_div = solver_df.step_dfsph_div
    step_vfsph_incomp = solver_df.step_vfsph_incomp
    step_vfsph_div = solver_df.step_vfsph_div
    
    # Functions: WCSPH
    step_wcsph_add_acc_pressure = solver_wcsph.step_wcsph_add_acc_pressure
    step_wcsph_add_acc_number_density_pressure = solver_wcsph.step_wcsph_add_acc_number_density_pressure

    # Functions: CFL time step
    find_max_vec = cfl.find_max_vec
    cfl_dt = cfl.cfl_dt
    get_cfl_dt_obj = cfl.get_cfl_dt_obj