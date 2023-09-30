import taichi as ti
from.sph_funcs import *
from ..basic_op.type import *
from ..basic_obj.Obj_Particle import Particle

@ti.data_oriented
class Adv_slover:
    def __init__(self, obj: Particle):
        self.obj = obj
        self.dt = obj.m_world.g_dt
        self.gravity = obj.m_world.g_gravity
        self.dim = obj.m_world.g_dim

        self.clean_acc = vecxf(self.dim[None])(0)
    
    @ti.kernel
    def clear_acc(self):
        for i in range(self.obj.tiGet_stack_top()[None]):
            self.obj.acc[i] = self.clean_acc

    @ti.kernel
    def add_gravity_acc(self):
        for i in range(self.obj.tiGet_stack_top()[None]):
            self.obj.acc[i] += self.gravity[None]
    
    @ti.kernel
    def add_vis_acc(self, kinetic_vis_coeff: ti.template()):
        for i in range(self.obj.tiGet_stack_top()[None]):
            self.obj.acc[i] += 0
    
    @ti.func
    def inloop_accumulate_vis_acc(self, part_id: ti.i32, neighb_part_id: ti.i32, neighb_part_shift: ti.i32, neighb_pool:ti.template(), neighb_obj:ti.template()):
        cached_dist = neighb_pool.tiGet_
        cached_grad_W = neighb_pool.tiGet_cachedGradW(neighb_part_shift)
        if bigger_than_zero(cached_dist):
            k_vis = (self.obj.k_vis[part_id] + neighb_obj.k_vis[neighb_part_id]) / 2
            A_ij = self.obj.vel[part_id] - neighb_obj.vel[neighb_part_id]
            x_ij = self.obj.pos[part_id] - neighb_obj.pos[neighb_part_id]
            self.obj.acc[part_id] += k_vis*2*(2+self.obj.m_world.g_dim[None]) * neighb_obj.volume[neighb_part_id] * cached_grad_W * A_ij.dot(x_ij) / (cached_dist**2)
    
    @ti.kernel
    def add_acc_gravity(self):
        for i in range(self.obj.tiGet_stack_top()[None]):
            self.obj.acc[i] += self.gravity[None]

    @ti.kernel
    def acc2vel_adv(self):
        for i in range(self.obj.tiGet_stack_top()[None]):
            self.obj.vel_adv[i] = self.obj.acc[i] * self.dt[None] + self.obj.vel[i]
    
    @ti.kernel
    def acc2vel(self):
        for i in range(self.obj.tiGet_stack_top()[None]):
            self.obj.vel[i] = self.obj.acc[i] * self.dt[None] + self.obj.vel[i]

    @ti.kernel
    def vel_adv2vel(self):
        for i in range(self.obj.tiGet_stack_top()[None]):
            self.obj.vel[i] = self.obj.vel_adv[i]

    @ti.kernel
    def update_pos(self):
        for i in range(self.obj.tiGet_stack_top()[None]):
            self.obj.pos[i] += self.obj.vel[i] * self.dt[None]

    @ti.kernel
    def adv_step(self, in_vel: ti.template(), out_vel_adv: ti.template()):
        for i in range(self.obj.tiGet_stack_top()[None]):
            out_vel_adv[i] = in_vel[i]
            self.obj.acc[i] = self.clean_acc
            self.obj.acc[i] += self.gravity[None]
            out_vel_adv[i] += self.obj.acc[i] * self.dt[None]
