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
        self.pos0neighb_module  = self.getObj().add_unit_neighb_search([self.getObj()])
        
        self.debug_flag         = val_i(0)

    def init(self):
        self.init_pos()
        self.init_neighb_search()
        self.compute_modulous()
        self.clear_corMatInv()
        self.pos0neighb_module.loop_self(self.inloop_compute_corMat_inv)
        self.compute_corMat()

    def update_rest(self):
        self.init_pos()
        self.pos0neighb_module.update()
        self.pos0neighb_module.pool()
        self.compute_modulous()
        self.clear_corMatInv()
        self.pos0neighb_module.loop_self(self.inloop_compute_corMat_inv)
        self.compute_corMat()

    def step(self):
        self.clear_force()
        self.clear_defGrad()
        
        self.pos0neighb_module.loop_self(self.inloop_compute_defGrad)
        # self.compute_svd_defGrad()
        self.compute_rotation()

        self.clear_defGrad()
        self.pos0neighb_module.loop_self(self.inloop_compute_corrected_defGrad)
        self.compute_corrected_defGrad()
        self.compute_strain_Green()
        self.compute_stress_StVK()
        self.pos0neighb_module.loop_self(self.inloop_compute_force)
        self.update_acc()

    @ti.kernel
    def init_pos(self):
        pos = ti.static(self.tiGetObj().tiGetPosArr())
        for i in range(self.tiGetObj().tiGetStackTop()):
            self.pos0[i] = pos[i]
    
    def init_neighb_search(self):
        self.pos0neighb_module.init_module()
        self.pos0neighb_module.update()
        self.pos0neighb_module.pool()

    @ti.kernel
    def compute_modulous(self):
        G   = ti.static(self.tiGetObj().tiGetElasticShearModulusArr())
        K   = ti.static(self.tiGetObj().tiGetElasticBulkModulusArr())
        for i in range(self.tiGetObj().tiGetStackTop()):
            G[i] = self.lame_mu[None]
            K[i] = self.lame_lambda[None] + (2/3 * self.lame_mu[None])

    @ti.kernel
    def compute_corMat(self):
        L      = ti.static(self.tiGetObj().tiGetElasticCorMatArr())
        L_inv  = ti.static(self.tiGetObj().tiGetElasticCorMatInvArr())
        for i in range(self.tiGetObj().tiGetStackTop()):
            # if ((L[i]*L[i]).sum() > 1e-8):
            L[i] = L_inv[i].inverse()
            self.tiGetObj().vis_1[i] = (L[i]*L[i]).sum()
            L[i] = ti.Matrix.identity(ti.f32, L[i].n)
            # if ((L_inv[i]*L_inv[i]).sum() < 1e-8):
            #     self.debug_flag[None] = 1
                # print("singular matrixsingular matrixsingular matrixsingular matrixsingular matrixsingular matrixsingular matrixsingular matrixsingular matrixsingular matrixsingular matrixsingular matrixsingular matrixsingular matrixsingular matrixsingular matrix")
            #     L[i] = ti.Matrix.identity(ti.f32, L[i].n)

    @ti.kernel
    def compute_svd_defGrad(self):
        F   = ti.static(self.tiGetObj().tiGetElasticDefGradArr())
        U   = ti.static(self.tiGetObj().tiGetElasticSvdUArr())
        V_T = ti.static(self.tiGetObj().tiGetElasticSvdVTArr())
        SIG = ti.static(self.tiGetObj().tiGetElasticSvdSigArr())
        for i in range(self.tiGetObj().tiGetStackTop()):
            U[i], SIG[i], V_T[i] = ti.svd(F[i])

    @ti.kernel
    def compute_rotation(self):
        R = ti.static(self.tiGetObj().tiGetElasticRotationArr())
        F = ti.static(self.tiGetObj().tiGetElasticDefGradArr())
        for i in range(self.tiGetObj().tiGetStackTop()):
            R[i] = ti.polar_decompose(F[i])[0]

    @ti.kernel
    def compute_corrected_defGrad(self):
        F = ti.static(self.tiGetObj().tiGetElasticDefGradArr())
        L = ti.static(self.tiGetObj().tiGetElasticCorMatArr())
        for i in range(self.tiGetObj().tiGetStackTop()):
            F[i] = L[i] @ F[i]

    @ti.func
    def inloop_compute_corMat_inv(self, part_id: ti.i32, neighb_part_id: ti.i32, neighb_part_shift: ti.i32, neighb_search_module:ti.template(), neighb_obj:ti.template()):
        cached_dist_0   = neighb_search_module.tiGet_cachedDist(neighb_part_shift)
        cached_grad_W_0 = neighb_search_module.tiGet_cachedGradW(neighb_part_shift)
        L_inv           = ti.static(self.tiGetObj().tiGetElasticCorMatInvArr())
        if bigger_than_zero(cached_dist_0):
            pos0_i              = self.tiGetPos0(part_id)
            pos0_j              = neighb_obj.tiGetSolverElastic().tiGetPos0(neighb_part_id)
            x_ji_0              = pos0_j  - pos0_i
            V_j                 = neighb_obj.tiGetVolume(neighb_part_id)
            L_inv[part_id] += V_j * cached_grad_W_0.outer_product(x_ji_0)

    @ti.func
    def inloop_compute_defGrad(self, part_id: ti.i32, neighb_part_id: ti.i32, neighb_part_shift: ti.i32, neighb_search_module:ti.template(), neighb_obj:ti.template()):
        cached_dist_0   = neighb_search_module.tiGet_cachedDist(neighb_part_shift)
        cached_grad_W_0 = neighb_search_module.tiGet_cachedGradW(neighb_part_shift)
        F               = ti.static(self.tiGetObj().tiGetElasticDefGradArr())
        L               = ti.static(self.tiGetObj().tiGetElasticCorMatArr())
        pos             = ti.static(self.tiGetObj().tiGetPosArr())
        neighb_pos      = ti.static(neighb_obj.tiGetPosArr())
        if bigger_than_zero(cached_dist_0):
            x_ji                = neighb_pos[neighb_part_id] - pos[part_id]
            corrected_grad_W    = L[part_id] @ cached_grad_W_0
            V_j                 = neighb_obj.tiGetVolume(neighb_part_id)
            F[part_id]         += V_j * x_ji.outer_product(corrected_grad_W)

    @ti.func
    def inloop_compute_corrected_defGrad(self, part_id: ti.i32, neighb_part_id: ti.i32, neighb_part_shift: ti.i32, neighb_search_module:ti.template(), neighb_obj:ti.template()):
        cached_dist_0   = neighb_search_module.tiGet_cachedDist(neighb_part_shift)
        cached_grad_W_0 = neighb_search_module.tiGet_cachedGradW(neighb_part_shift)
        F               = ti.static(self.tiGetObj().tiGetElasticDefGradArr())
        R               = ti.static(self.tiGetObj().tiGetElasticRotationArr())
        L               = ti.static(self.tiGetObj().tiGetElasticCorMatArr())
        if bigger_than_zero(cached_dist_0):
            x_ji_0              = neighb_obj.tiGetSolverElastic().tiGetPos0(neighb_part_id) - self.tiGetPos0(part_id)
            x_ji                = neighb_obj.tiGetPos(neighb_part_id) - self.tiGetObj().tiGetPos(part_id)
            corrected_grad_W    = R[part_id] @ L[part_id] @ cached_grad_W_0   
            F[part_id]         += neighb_obj.tiGetVolume(neighb_part_id) * (x_ji - (R[part_id] @ x_ji_0)).outer_product(corrected_grad_W)

    @ti.kernel
    def compute_corrected_defGrad(self):
        F = ti.static(self.tiGetObj().tiGetElasticDefGradArr())
        I = ti.math.eye(F.n)
        for i in range(self.tiGetObj().tiGetStackTop()):
            F[i] += I

    @ti.kernel
    def compute_strain_infinitesimal(self):
        F   = ti.static(self.tiGetObj().tiGetElasticDefGradArr())
        I   = ti.math.eye(F.n)
        eps = ti.static(self.tiGetObj().tiGetElasticStrainArr())
        for i in range(self.tiGetObj().tiGetStackTop()):
            eps[i] = 0.5 * (F[i].transpose() + F[i]) - I

    @ti.kernel
    def compute_stress_linear(self):
        eps = ti.static(self.tiGetObj().tiGetElasticStrainArr())
        P   = ti.static(self.tiGetObj().tiGetElasticStressArr())
        K   = ti.static(self.tiGetObj().tiGetElasticBulkModulusArr())
        G   = ti.static(self.tiGetObj().tiGetElasticShearModulusArr())
        I   = ti.math.eye(eps.n)
        for i in range(self.tiGetObj().tiGetStackTop()):
            P[i] = (2*G[i]*eps[i]) + ((K[i]-(2/3*G[i])) * eps[i].trace() * I)
    
    @ti.kernel
    def compute_strain_Green(self):
        F   = ti.static(self.tiGetObj().tiGetElasticDefGradArr())
        I   = ti.math.eye(F.n)
        eps = ti.static(self.tiGetObj().tiGetElasticStrainArr())
        for i in range(self.tiGetObj().tiGetStackTop()):
            eps[i] = 0.5 * (F[i].transpose() @ F[i] - I)
    
    @ti.kernel
    def compute_stress_StVK(self):
        eps = ti.static(self.tiGetObj().tiGetElasticStrainArr())
        P   = ti.static(self.tiGetObj().tiGetElasticStressArr())
        I   = ti.math.eye(eps.n)
        F   = ti.static(self.tiGetObj().tiGetElasticDefGradArr())
        for i in range(self.tiGetObj().tiGetStackTop()):
            P[i] = F[i] @ (2 * self.lame_mu[None] * eps[i] + self.lame_lambda[None] * eps[i].trace() * I)

    @ti.func
    def inloop_compute_force(self, part_id: ti.i32, neighb_part_id: ti.i32, neighb_part_shift: ti.i32, neighb_search_module:ti.template(), neighb_obj:ti.template()):
        cached_dist_0   = neighb_search_module.tiGet_cachedDist(neighb_part_shift)
        cached_grad_W_0 = neighb_search_module.tiGet_cachedGradW(neighb_part_shift)
        force           = ti.static(self.tiGetObj().tiGetElasticForceArr())
        Pi              = ti.static(self.tiGetObj().tiGetElasticStressArr())
        Pj              = ti.static(neighb_obj.tiGetElasticStressArr())
        Li              = ti.static(self.tiGetObj().tiGetElasticCorMatArr())
        Ri              = ti.static(self.tiGetObj().tiGetElasticRotationArr())
        Lj              = ti.static(neighb_obj.tiGetElasticCorMatArr())
        Rj              = ti.static(neighb_obj.tiGetElasticRotationArr())
        if bigger_than_zero(cached_dist_0):
            V_i             = self.tiGetObj().tiGetVolume(part_id)
            V_j             = neighb_obj.tiGetVolume(neighb_part_id)
            P_i             = Pi[part_id]
            P_j             = Pj[neighb_part_id]
            L_i             = Li[part_id]
            L_j             = Lj[neighb_part_id]
            R_i             = Ri[part_id]
            R_j             = Rj[neighb_part_id]
            W_star_i        = R_i @ L_i @ cached_grad_W_0
            W_star_j        = R_j @ L_j @ cached_grad_W_0
            force[part_id] += V_i*V_j*((P_i @ W_star_i) + (P_j @ W_star_j))

    @ti.kernel
    def update_acc(self):
        acc     = ti.static(self.tiGetObj().tiGetAccArr())
        force   = ti.static(self.tiGetObj().tiGetElasticForceArr())
        for i in range(self.tiGetObj().tiGetStackTop()):
            # print(force[i])
            acc[i] += force[i] / self.tiGetObj().tiGetMass(i)

    @ti.func
    def tiGetPos0(self, i):
        return self.pos0[i]
    
    @ti.kernel
    def clear_force(self):
        force = ti.static(self.tiGetObj().tiGetElasticForceArr())
        for i in range(self.tiGetObj().tiGetStackTop()):
            force[i] *= 0
    
    @ti.kernel
    def clear_defGrad(self):
        F = ti.static(self.tiGetObj().tiGetElasticDefGradArr())
        for i in range(self.tiGetObj().tiGetStackTop()):
            F[i] *= 0
    
    @ti.kernel
    def clear_corMatInv(self):
        L_inv = ti.static(self.tiGetObj().tiGetElasticCorMatInvArr())
        for i in range(self.tiGetObj().tiGetStackTop()):
            L_inv[i] *= 0