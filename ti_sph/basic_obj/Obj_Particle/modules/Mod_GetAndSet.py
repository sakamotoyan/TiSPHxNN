import taichi as ti

@ti.data_oriented
class Mod_GetAndSet:
# tiGet functions

    ''' particle class attrs get&set '''
    @ti.func
    def tiGetStackTop(self:ti.template()):
        return self.m_stack_top[None]
    @ti.func
    def tiGetPartNum(self)->ti.i32:
        return self.m_part_num[None]
    @ti.func
    def tiGetPartSize(self)->ti.f32:
        return self.m_part_size[None]
    @ti.func
    def tiCheckStackOpen(self):
        return self.m_if_stack_open
    @ti.func
    def tiSetStackOpen(self, if_open):
        self.m_if_stack_open = if_open
    @ti.func
    def tiGetStackOpenNum(self)->ti.i32:
        return self.m_stack_open_num[None]
    @ti.func
    def tiSetStackOpenNum(self, num: ti.i32):
        self.m_stack_open_num[None] = num
    @ti.func
    def tiGetWorld(self):
        return self.m_world
    @ti.func
    def tiSetWorld(self, world):
        self.m_world = world
    @ti.func
    def tiGetId(self):
        return self.m_id[None]
    @ti.func
    def tiSetId(self, id:int):
        self.m_id[None] = id

    def getStackTop(self)->int:
        return self.m_stack_top[None]
    def AddStackTop(self, num: int):
        self.m_stack_top[None] += num
    def getPartNum(self)->int:
        return self.m_part_num[None]
    def getPartSize(self)->float:
        return self.m_part_size[None]
    def checkStackOpen(self):
        return self.m_if_stack_open
    def setStackOpen(self, if_open: bool):
        self.m_if_stack_open = if_open
    def getStackOpenNum(self)->int:
        return self.m_stack_open_num[None]
    def setStackOpenNum(self, num: int):
        self.m_stack_open_num[None] = num
    def getWorld(self):
        return self.m_world
    def setWorld(self, world):
        self.m_world = world
    def getId(self):
        return self.m_id[None]
    def setId(self, id:int):
        self.m_id[None] = id

    ''' particle basic physical attrs get&set '''

    def getPos(self, i):
        return self.pos[i]
    def getPosArr(self):
        return self.pos
    def getVel(self, i):
        return self.vel[i]
    def getVelArr(self):
        return self.vel
    def getVelAdv(self, i):
        return self.vel_adv[i]
    def getAcc(self, i):
        return self.acc[i]
    def getMass(self, i):
        return self.mass[i]
    def getVolume(self, i):
        return self.volume[i]
    def getRestDensity(self, i):
        return self.rest_density[i]
    def getPressure(self, i):
        return self.pressure[i]
    def getKVis(self, i):
        return self.k_vis[i]
    def getRgb(self, i):
        return self.rgb[i]
    def getRgbArr(self):
        return self.rgb
    # def getPartSize(self, i):
    #     return self.size[i]
    
    # taichi version
    @ti.func
    def tiGetPos(self, i):
        return self.pos[i]
    @ti.func
    def tiSetPos(self, i, pos):
        self.pos[i] = pos
    @ti.func
    def tiAddPos(self, i, pos):
        self.pos[i] += pos
    @ti.func
    def tiGetPosArr(self):
        return self.pos
    @ti.func
    def tiGetVel(self, i):
        return self.vel[i]
    @ti.func
    def tiSetVel(self, i, vel):
        self.vel[i] = vel
    @ti.func
    def tiGetVelArr(self):
        return self.vel
    @ti.func
    def tiAddVel(self, i, vel):
        self.vel[i] += vel
    @ti.func
    def tiGetVelAdv(self, i):
        return self.vel_adv[i]
    @ti.func
    def tiSetVelAdv(self, i, vel_adv):
        self.vel_adv[i] = vel_adv
    @ti.func
    def tiGetAcc(self, i):
        return self.acc[i]
    @ti.func
    def tiSetAcc(self, i, acc):
        self.acc[i] = acc
    @ti.func
    def tiAddAcc(self, i, acc):
        self.acc[i] += acc
    @ti.func
    def tiGetAccArr(self):
        return self.acc
    def getAccArr(self):
        return self.acc

    @ti.func
    def tiGetMass(self, i):
        return self.mass[i]
    @ti.func
    def tiGetVolume(self, i):
        return self.volume[i]
    @ti.func
    def tiGetVolumeArr(self):
        return self.volume
    @ti.func
    def tiGetRestDensity(self, i):
        return self.rest_density[i]
    @ti.func
    def tiGetPressure(self, i):
        return self.pressure[i]
    @ti.func
    def tiGetKVis(self, i):
        return self.k_vis[i]
    @ti.func
    def tiGetRgb(self, i):
        return self.rgb[i]
    @ti.func
    def tiGetRgbArr(self):
        return self.rgb
    # @ti.func
    # def tiGetPartSize(self, i):
    #     return self.size[i]
    
    
    ''' particle sph attrs get&set '''

    def getSphH(self, i):
        return self.sph[i].h
    def getSphSig(self, i):
        return self.sph[i].sig
    def getSphSigInvH(self, i):
        return self.sph[i].sig_inv_h
    def getSphDensity(self, i):
        return self.sph[i].density
    def getSphDensityArr(self):
        return self.sph.density
    def getSphCompressionRatio(self, i):
        return self.sph[i].compression_ratio
    def getSphCompressionRatioArr(self):
        return self.sph.compression_ratio
    def getSphPressure(self, i):
        return self.sph[i].pressure
    def getSphPressureForce(self, i):
        return self.sph[i].pressure_force
    def getSphViscosityForce(self, i):
        return self.sph[i].viscosity_force
    def getSphGravityForce(self, i):
        return self.sph[i].gravity_force
    def getSphAlpha1(self, i):
        return self.sph_df[i].alpha_1
    def getSphAlpha2(self, i):
        return self.sph_df[i].alpha_2
    def getSphAlpha(self, i):
        return self.sph_df[i].alpha
    def getSphKappaIncomp(self, i):
        return self.sph_df[i].kappa_incomp
    def getSphKappaDiv(self, i):
        return self.sph_df[i].kappa_div
    def getSphDeltaDensity(self, i):
        return self.sph_df[i].delta_density
    def getSphDeltaCompressionRatio(self, i):
        return self.sph_df[i].delta_compression_ratio
    def getSphVelAdv(self, i):
        return self.sph_df[i].vel_adv
    def getSphB(self, i):
        return self.sph_wc[i].B
    
    # taichi version
    @ti.func
    def tiGetSphH(self, i):
        return self.sph[i].h
    @ti.func
    def tiSetSphH(self, i, h):
        self.sph[i].h = h
    @ti.func
    def tiGetSphSig(self, i):
        return self.sph[i].sig
    @ti.func
    def tiSetSphSig(self, i, sig):
        self.sph[i].sig = sig
    @ti.func
    def tiGetSphSigInvH(self, i):
        return self.sph[i].sig_inv_h
    @ti.func
    def tiSetSphSigInvH(self, i, sig_inv_h):
        self.sph[i].sig_inv_h = sig_inv_h
    @ti.func
    def tiGetSphDensity(self, i):
        return self.sph[i].density
    @ti.func
    def tiGetSphDensityArr(self):
        return self.sph.density
    @ti.func
    def tiSetSphDensity(self, i, density):
        self.sph[i].density = density
    @ti.func
    def tiAddSphDensity(self, i, density):
        self.sph[i].density += density
    @ti.func
    def tiGetSphCompressionRatio(self, i):
        return self.sph[i].compression_ratio
    @ti.func
    def tiGetSphCompressionRatioArr(self):
        return self.sph.compression_ratio
    @ti.func
    def tiSetSphCompressionRatio(self, i, compression_ratio):
        self.sph[i].compression_ratio = compression_ratio
    @ti.func
    def tiAddSphCompressionRatio(self, i, compression_ratio):
        self.sph[i].compression_ratio += compression_ratio
    @ti.func
    def tiGetSphPressure(self, i):
        return self.sph[i].pressure
    @ti.func
    def tiSetSphPressure(self, i, pressure):
        self.sph[i].pressure = pressure
    @ti.func
    def tiGetSphPressureForce(self, i):
        return self.sph[i].pressure_force
    @ti.func
    def tiSetSphPressureForce(self, i, pressure_force):
        self.sph[i].pressure_force = pressure_force
    @ti.func
    def tiGetSphViscosityForce(self, i):
        return self.sph[i].viscosity_force
    @ti.func
    def tiSetSphViscosityForce(self, i, viscosity_force):
        self.sph[i].viscosity_force = viscosity_force
    @ti.func
    def tiGetSphGravityForce(self, i):
        return self.sph[i].gravity_force
    @ti.func
    def tiSetSphGravityForce(self, i, gravity_force):
        self.sph[i].gravity_force = gravity_force
    @ti.func
    def tiGetSphAlpha1(self, i):
        return self.sph_df[i].alpha_1
    @ti.func
    def tiSetSphAlpha1(self, i, alpha_1):
        self.sph_df[i].alpha_1 = alpha_1
    @ti.func
    def tiGetSphAlpha2(self, i):
        return self.sph_df[i].alpha_2
    @ti.func
    def tiSetSphAlpha2(self, i, alpha_2):
        self.sph_df[i].alpha_2 = alpha_2
    @ti.func
    def tiGetSphAlpha(self, i):
        return self.sph_df[i].alpha
    @ti.func
    def tiSetSphAlpha(self, i, alpha):
        self.sph_df[i].alpha = alpha
    @ti.func
    def tiGetSphKappaIncomp(self, i):
        return self.sph_df[i].kappa_incomp
    @ti.func
    def tiSetSphKappaIncomp(self, i, kappa_incomp):
        self.sph_df[i].kappa_incomp = kappa_incomp
    @ti.func
    def tiGetSphKappaDiv(self, i):
        return self.sph_df[i].kappa_div
    @ti.func
    def tiSetSphKappaDiv(self, i, kappa_div):
        self.sph_df[i].kappa_div = kappa_div
    @ti.func
    def tiGetSphDeltaDensity(self, i):
        return self.sph_df[i].delta_density
    @ti.func
    def tiSetSphDeltaDensity(self, i, delta_density):
        self.sph_df[i].delta_density = delta_density
    @ti.func
    def tiGetSphDeltaCompressionRatio(self, i):
        return self.sph_df[i].delta_compression_ratio
    @ti.func
    def tiSetSphDeltaCompressionRatio(self, i, delta_compression_ratio):
        self.sph_df[i].delta_compression_ratio = delta_compression_ratio
    @ti.func
    def tiGetSphVelAdv(self, i):
        return self.sph_df[i].vel_adv
    @ti.func
    def tiSetSphVelAdv(self, i, vel_adv):
        self.sph_df[i].vel_adv = vel_adv
    @ti.func
    def tiGetSphB(self, i):
        return self.sph_wc[i].B
    @ti.func
    def tiSetSphB(self, i, B):
        self.sph_wc[i].B = B
    

    def getStrainRate(self, i):
        return self.strainRate[i]
    @ti.func
    def tiGetStrainRate(self, i: ti.i32):
        return self.strainRate[i]
    @ti.func
    def tiAddStrainRate(self, i: ti.i32, strainRate):
        self.strainRate[i] += strainRate
    def getStrainRateArr(self):
        return self.strainRate
    
    @ti.func
    def tiGetElasticForce(self, i):
        return self.sph_elastic[i].force
    @ti.func
    def tiSetElasticForce(self, i, force):
        self.sph_elastic[i].force = force
    @ti.func
    def tiGetElasticForceArr(self):
        return self.sph_elastic.force
    
    # @ti.func
    # def tiGetElasticPos0(self, i):
    #     return self.sph_elastic[i].pos_0
    # @ti.func
    # def tiSetElasticPos0(self, i, pos_0):
    #     self.sph_elastic[i].pos_0 = pos_0
    # @ti.func
    # def tiGetElasticPos0Arr(self):
    #     self.sph_elastic.pos_0

    @ti.func
    def tiGetElasticDefGrad(self, i):
        return self.sph_elastic[i].F
    @ti.func
    def tiSetElasticDefGrad(self, i, F):
        self.sph_elastic[i].F = F
    @ti.func
    def tiGetElasticDefGradArr(self):
        return self.sph_elastic.F
    
    @ti.func
    def tiGetElasticSvdUArr(self):
        return self.sph_elastic.svd_U
    @ti.func
    def tiGetElasticSvdSigArr(self):
        return self.sph_elastic.svd_SIG
    @ti.func
    def tiGetElasticSvdVTArr(self):
        return self.sph_elastic.svd_V_T
    @ti.func
    def tiGetElasticShearModulusArr(self):
        return self.sph_elastic.G
    @ti.func
    def tiGetElasticBulkModulusArr(self):
        return self.sph_elastic.K

    @ti.func
    def tiGetElasticCorMat(self, i):
        return self.sph_elastic[i].L
    @ti.func
    def tiSetElasticCorMat(self, i, L):
        self.sph_elastic[i].L = L
    @ti.func
    def tiGetElasticCorMatArr(self):
        return self.sph_elastic.L

    @ti.func
    def tiGetElasticCorMatInv(self, i):
        return self.sph_elastic[i].L_inv
    @ti.func
    def tiSetElasticCorMatInv(self, i, L_inv):
        self.sph_elastic[i].L_inv = L_inv
    @ti.func
    def tiGetElasticCorMatInvArr(self):
        return self.sph_elastic.L_inv

    @ti.func
    def tiGetElasticRotation(self, i):
        return self.sph_elastic[i].R
    @ti.func
    def tiSetElasticRotation(self, i, R):
        self.sph_elastic[i].R = R
    @ti.func
    def tiGetElasticRotationArr(self):
        return self.sph_elastic.R

    @ti.func
    def tiGetElasticStrain(self, i):
        return self.sph_elastic[i].eps
    @ti.func
    def tiSetElasticStrain(self, i, eps):
        self.sph_elastic[i].eps = eps
    @ti.func
    def tiGetElasticStrainArr(self):
        return self.sph_elastic.eps

    @ti.func
    def tiGetElasticStress(self, i):
        return self.sph_elastic[i].P
    @ti.func
    def tiSetElasticStress(self, i, P):
        self.sph_elastic[i].P = P
    @ti.func
    def tiGetElasticStressArr(self):
        return self.sph_elastic.P
