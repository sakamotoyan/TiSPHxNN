import taichi as ti
import math
from ..basic_obj.Obj_Particle import Particle

@ti.data_oriented
class SPH_solver:
    def __init__(self, obj: Particle):
        
        self.obj = obj

        self.dim = obj.m_world.g_dim
        sig_dim = self.sig_dim(self.dim[None])
        self.compute_sig(sig_dim)

        self.dt=obj.m_world.g_dt
        self.inv_dt = obj.m_world.g_inv_dt
        self.neg_inv_dt = obj.m_world.g_neg_inv_dt
        self.inv_dt2 = obj.m_world.g_inv_dt2

    @ti.kernel
    def loop_neighb(self, neighb_pool:ti.template(), neighb_obj:ti.template(), func:ti.template()):
        for part_id in range(self.obj.ti_get_stack_top()[None]):
            neighb_part_num = neighb_pool.neighb_obj_pointer[part_id, neighb_obj.ti_get_id()[None]].size
            neighb_part_shift = neighb_pool.neighb_obj_pointer[part_id, neighb_obj.ti_get_id()[None]].begin
            for neighb_part_iter in range(neighb_part_num):
                neighb_part_id = neighb_pool.neighb_pool_container[neighb_part_shift].neighb_part_id
                ''' Code for Computation'''
                func(part_id, neighb_part_id, neighb_part_shift, neighb_pool, neighb_obj)
                ''' End of Code for Computation'''
                ''' DO NOT FORGET TO COPY/PASE THE FOLLOWING CODE WHEN REUSING THIS FUNCTION '''
                neighb_part_shift = neighb_pool.neighb_pool_container[neighb_part_shift].next
    
    ''' [NOTICE] If sig_dim is decorated with @ti.func, and called in a kernel, 
    it will cause a computation error due to the use of math.pi. This bug is tested. '''
    def sig_dim(self, dim):
        sig = 0
        if dim == 3:
            sig = 8 / math.pi 
        elif dim == 2:
            sig = 40 / 7 / math.pi
        elif dim == 1:
            sig = 4 / 3
        return sig
    
    @ti.kernel
    def compute_sig(self, sig_dim: ti.f32):
        for part_id in range(self.obj.ti_get_stack_top()[None]):
            self.obj.sph[part_id].h = self.obj.size[part_id] * 2
            self.obj.sph[part_id].sig = sig_dim / ti.pow(self.obj.sph[part_id].h, self.dim[None])
            self.obj.sph[part_id].sig_inv_h = self.obj.sph[part_id].sig / self.obj.sph[part_id].h

    @ti.func
    def inloop_accumulate_density(self, part_id: ti.i32, neighb_part_id: ti.i32, neighb_part_shift: ti.i32, neighb_pool:ti.template(), neighb_obj:ti.template()):
        cached_W = neighb_pool.cached_neighb_attributes[neighb_part_shift].W
        self.obj.sph[part_id].density += neighb_obj.mass[neighb_part_id] * cached_W
    
    @ti.func
    def inloop_accumulate_number_density(self, part_id: ti.i32, neighb_part_id: ti.i32, neighb_part_shift: ti.i32, neighb_pool:ti.template(), neighb_obj:ti.template()):
        cached_W = neighb_pool.cached_neighb_attributes[neighb_part_shift].W
        self.obj.sph[part_id].density += self.obj.mass[part_id] * cached_W

    @ti.func
    def inloop_accumulate_compression_ratio(self, part_id: ti.i32, neighb_part_id: ti.i32, neighb_part_shift: ti.i32, neighb_pool:ti.template(), neighb_obj:ti.template()):
        cached_W = neighb_pool.cached_neighb_attributes[neighb_part_shift].W
        self.obj.sph[part_id].compression_ratio += neighb_obj.volume[neighb_part_id] * cached_W
    
    @ti.func
    def inloop_avg_velocity(self, part_id: ti.i32, neighb_part_id: ti.i32, neighb_part_shift: ti.i32, neighb_pool:ti.template(), neighb_obj:ti.template()):
        cached_W = neighb_pool.cached_neighb_attributes[neighb_part_shift].W
        cached_grad_W = neighb_pool.cached_neighb_attributes[neighb_part_shift].grad_W
        self.obj.vel[part_id] += neighb_obj.vel[neighb_part_id] * neighb_obj.volume[neighb_part_id] * cached_W

    def sph_compute_density(self, neighb_pool):
        self.obj.clear(self.obj.sph.density)
        for neighb_obj in neighb_pool.neighb_obj_list:
            ''' Compute Density '''
            self.loop_neighb(neighb_pool, neighb_obj, self.inloop_accumulate_density)
    
    def sph_compute_number_density(self, neighb_pool):
        self.obj.clear(self.obj.sph.density)
        for neighb_obj in neighb_pool.neighb_obj_list:
            ''' Compute Density '''
            self.loop_neighb(neighb_pool, neighb_obj, self.inloop_accumulate_number_density)
    
    def sph_compute_compression_ratio(self, neighb_pool):
        self.obj.clear(self.obj.sph.compression_ratio)
        for neighb_obj in neighb_pool.neighb_obj_list:
            ''' Compute Compression Ratio '''
            self.loop_neighb(neighb_pool, neighb_obj, self.inloop_accumulate_compression_ratio)
    
    def sph_avg_velocity(self, neighb_pool):
        self.obj.clear(self.obj.vel)
        for neighb_obj in neighb_pool.neighb_obj_list:
            ''' Compute Average Velocity '''
            self.loop_neighb(neighb_pool, neighb_obj, self.inloop_avg_velocity)