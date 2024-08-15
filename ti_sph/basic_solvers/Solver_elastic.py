import taichi as ti
import math

from ti_sph.basic_obj.Obj_Particle import Particle
from .sph_funcs import *
from .Solver import Solver
from ..basic_op.type import *
from ..basic_obj.Obj_Particle import Particle
from ..basic_neighb_search import Neighb_search
from typing import List

@ti.data_oriented
class Elastic_solver(Solver):
    def __init__(self, obj: Particle, lame_lambda, lame_mu):
        super().__init__(obj)

        self.lame_lambda        = ti.field(ti.f32, ())
        self.lame_mu            = ti.field(ti.f32, ())

        self.lame_lambda[None]  = lame_lambda
        self.lame_mu[None]      = lame_mu

        self.pos0               = ti.Vector.field(self.getObj().getWorld().getDim(), dtype=ti.f32, shape=self.getObj().getPartNum())
        self.pos0neighb         = self.getObj().add_unit_neighb_search([self.getObj()])
        
    def init(self):
        self.init_pos()
        self.init_neighb_search()

    @ti.kernel
    def init_pos(self):
        pos = ti.static(self.tiGetObj().tiGetPosArr())
        for i in range(self.tiGetObj().tiGetStackTop()):
            self.pos0[i] = pos[i]
    
    def init_neighb_search(self):
        self.pos0neighb.init_module()
        self.pos0neighb.update()
        self.pos0neighb.pool()

    @ti.kernel
    def clear_force(self):
        elastic_force = ti.static(self.tiGetObj().tiGetElasticForceArr())
        for i in range(self.tiGetObj().tiGetStackTop()):
            elastic_force[i] *= 0
    
    @ti.kernel
    def clear_defGrad(self):
        defGrad = ti.static(self.tiGetObj().tiGetElasticDefGradArr())
        for i in range(self.tiGetObj().tiGetStackTop()):
            defGrad[i] *= 0
    
    @ti.kernel
    def clear_corMat(self):
        corMat = ti.static(self.tiGetObj().tiGetElasticCorMatArr())
        for i in range(self.tiGetObj().tiGetStackTop()):
            corMat[i] *= 0
    
    @ti.func
    def inloop_compute_F(self, part_id: ti.i32, neighb_part_id: ti.i32, neighb_part_shift: ti.i32, neighb_search_module:ti.template(), neighb_obj:ti.template()):
        cached_dist_0 = neighb_search_module.tiGet_cachedDist(neighb_part_shift)
        cached_grad_W_0 = neighb_search_module.tiGet_cachedGradW(neighb_part_shift)
        if bigger_than_zero(cached_dist_0):            
            x_ij = self.obj.pos[part_id] - neighb_obj.pos[neighb_part_id]            
            grad_W_vec=self.obj.elastic_sph[part_id].L @ cached_grad_W_0
            self.obj.elastic_sph[part_id].F += neighb_obj.volume[neighb_part_id] * (-x_ij).outer_product(grad_W_vec)