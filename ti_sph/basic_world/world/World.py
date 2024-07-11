import taichi as ti

from typing import List
from .modules import neighb_search
from .modules import solver_sph
from .modules import solver_adv
from .modules import solver_df
from .modules import cfl
from .modules import solver_wcsph

from ...basic_op.type import *
from ...basic_op import *

@ti.data_oriented
class World:
    def __init__(self, dim=3, lb = -8, rt = 8):
        ''' GLOBAL CONFIGURATION '''
        self.g_dim = val_i(dim)
        self.g_space_lb = vecx_f(self.g_dim[None])
        self.g_space_rt = vecx_f(self.g_dim[None])
        self.g_gravity = vecx_f(self.g_dim[None])
        self.g_space_lb.fill(lb)
        self.g_space_rt.fill(rt)
        self.g_gravity[None][1] = -9.8
        self.g_dt = val_f(0.001)
        self.g_time = val_f(0)
        self.g_inv_dt = val_f(1/self.g_dt[None])
        self.g_neg_inv_dt = val_f(-1/self.g_dt[None])
        self.g_inv_dt2 = val_f(self.g_inv_dt[None] ** 2) 
        self.g_part_size = val_f(0.1)
        self.g_avg_neighb_part_num = val_i(5**self.g_dim[None]) if dim==2 else val_i(4**self.g_dim[None])
        self.g_obj_num = val_i(3)
        self.g_sound_speed = val_f(100)

        self.dependent_init()
        self.part_obj_list = []

    def dependent_init(self):
        self.space_size = vecx_f(self.g_dim[None])
        self.space_center = vecx_f(self.g_dim[None])
        self.space_size[None] = self.g_space_rt[None] - self.g_space_lb[None]
        self.space_center[None] = (self.g_space_rt[None] + self.g_space_lb[None]) / 2

        self.part_volume = val_f(self.g_part_size[None] ** self.g_dim[None])
        self.support_radius = val_f(self.g_part_size[None] * 2)

    # Functions: init related
    def refresh(self):
        self.space_size[None] = self.g_space_rt[None] - self.g_space_lb[None]
        self.space_center[None] = (self.g_space_rt[None] + self.g_space_lb[None]) / 2

        self.part_volume = val_f(self.g_part_size[None] ** self.g_dim[None])
        self.support_radius = val_f(self.g_part_size[None] * 2)

    @ti.func
    def tiGetPartSize(self) -> ti.f32:
        return self.g_part_size[None]
    def getPartSize(self) -> ti.f32:
        return self.g_part_size[None]
    
    @ti.func
    def tiGetTime(self) -> ti.f32:
        return self.g_time[None]
    def getTime(self) -> ti.f32:
        return self.g_time[None]
    @ti.func
    def tiSetTime(self, time):
        self.g_time[None] = time
    def setTime(self, time):
        self.g_time[None] = time

    @ti.func
    def tiGetDim(self)->ti.i32:
        return self.g_dim[None]
    def getDim(self)->int:
        return self.g_dim[None]

    @ti.func
    def tiGetLb(self):
        return self.g_space_lb[None]
    def getLb(self):
        return self.g_space_lb[None]

    @ti.func
    def tiGetRt(self):
        return self.g_space_rt[None]
    def getRt(self):
        return self.g_space_rt[None]
    
    @ti.func
    def tiGetGravity(self):
        return self.g_gravity[None]
    def getGravity(self):  
        return self.g_gravity[None]
    
    def setGravityMagnitude(self, gravity_val):
        self.g_gravity[None][1] = gravity_val
    
    @ti.func
    def tiGetDt(self)->ti.f32:
        return self.g_dt[None]
    def getdDt(self)->ti.f32:
        return self.g_dt[None]
    @ti.func
    def tiSetDt(self, dt):
        self.g_dt[None] = dt
        self.g_inv_dt[None] = 1/dt
        self.g_neg_inv_dt[None] = -1/dt
        self.g_inv_dt2[None] = self.g_inv_dt[None] ** 2
    def setDt(self, dt):
        self.g_dt[None] = dt
        self.g_inv_dt[None] = 1/dt
        self.g_neg_inv_dt[None] = -1/dt
        self.g_inv_dt2[None] = self.g_inv_dt[None] ** 2
    @ti.func
    def tiGetInvDt(self):
        return self.g_inv_dt
    def getInvDt(self):
        return self.g_inv_dt
    @ti.func
    def tiGetNegInvDt(self):
        return self.g_neg_inv_dt
    def getNegInvDt(self):
        return self.g_neg_inv_dt
    @ti.func
    def tiGetSqInvDt(self):
        return self.g_inv_dt2
    def getSqInvDt(self):
        return self.g_inv_dt2
    
    @ti.func
    def tiGetPartSize(self)->ti.f32:
        return self.g_part_size[None]
    def getPartSize(self)->float:
        return self.g_part_size[None]
    def setPartSize(self, size):
        self.g_part_size = val_f(size)
        self.refresh()
    
    @ti.func
    def tiGetAvgNeighbPartNum(self):
        return self.g_avg_neighb_part_num
    def getAvgNeighbPartNum(self):
        return self.g_avg_neighb_part_num
    
    @ti.func
    def tiGetObjNum(self):
        return self.g_obj_num
    def getObjNum(self):
        return self.g_obj_num
    def setObjNum(self, num):
        self.g_obj_num = val_i(num)
    
    @ti.func
    def tiGetSoundSpeed(self):
        return self.g_sound_speed
    def getSoundSpeed(self):
        return self.g_sound_speed
    
    @ti.func
    def tiGetPartVolume(self):
        return self.part_volume
    def getPartVolume(self):
        return self.part_volume
    
    @ti.func
    def tiGetSupportRadius(self):
        return self.support_radius[None]
    def getSupportRadius(self):
        return self.support_radius[None]
    
    @ti.func
    def tiGetSpace(self):
        return self.space_size
    def getSpace(self):
        return self.space_size
    
    @ti.func
    def tiGetCenterCord(self):
        return self.space_center
    def getCenterCord(self):
        return self.space_center

    @ti.func
    def tiGetPhaseNum(self):
        return self.g_phase_num
    def getPhaseNum(self):
        return self.g_phase_num
    
    @ti.func
    def tiGetPhaseColor(self, phase_id: ti.i32):
        return self.g_phase_color[phase_id]
    def getPhaseColor(self, phase_id: ti.i32):
        return self.g_phase_color[phase_id]
    
    @ti.func
    def tiGetPhaseRestDensity(self, phase_id: ti.i32):
        return self.g_phase_rest_density[None][phase_id]
    def getPhaseRestDensity(self, phase_id: ti.i32):
        return self.g_phase_rest_density[None][phase_id]
    


    def set_multiphase(self, phase_num, phase_color:List[vec3f], phase_rest_density:List[float]):
        self.g_phase_num = val_i(phase_num)
        self.g_phase_color = ti.Vector.field(3, dtype=ti.f32, shape=phase_num)
        self.g_phase_rest_density = vecx_f(phase_num)
        for i in range(phase_num):
            self.g_phase_color[i] = phase_color[i]
            self.g_phase_rest_density[None][i] = phase_rest_density[i]
        DEBUG('world.g_phase_num\n', self.g_phase_num[None])
        DEBUG('world.g_phase_color\n', self.g_phase_color.to_numpy())
        DEBUG('world.g_phase_rest_density\n', self.g_phase_rest_density.to_numpy())
        
    # def add_part_obj(self, part_num, is_dynamic, size: ti.template()):
    #     obj = Particle(part_num, size, is_dynamic)
    #     self.part_obj_list.append(obj)
    #     obj.setObjId(val_i(self.part_obj_list.index(obj)))
    #     obj.setObjWorld(self)
    #     return obj
    
    def attachPartObj(self, partObj):
        DEBUG("attaching Particle object to world ...")
        self.part_obj_list.append(partObj)
        partObj.setId(self.part_obj_list.index(partObj))
        partObj.setWorld(self)
        DEBUG('Done! ' + 'object id: ' + str(partObj.getId()))
    
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