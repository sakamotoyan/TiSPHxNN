import taichi as ti

@ti.data_oriented
class Mod_GetAndSet:
# tiGet functions

    ''' particle class attrs get&set '''
    @ti.func
    def tiGetObjStackTop(self:ti.template()):
        return self.m_stack_top
    @ti.func
    def tiGetObjPartNum(self):
        return self.m_part_num
    @ti.func
    def tiGetObjPartSize(self):
        return self.m_part_size
    @ti.func
    def tiCheckObjStackOpen(self):
        return self.m_if_stack_open
    @ti.func
    def tiGetObjStackOpenNum(self):
        return self.m_stack_open_num
    @ti.func
    def tiSetObjStackOpenNum(self, num: ti.i32):
        self.m_stack_open_num = num
    @ti.func
    def tiGetObjWorld(self):
        return self.m_world
    @ti.func
    def tiSetObjWorld(self, world):
        self.m_world = world
    @ti.func
    def tiGetObjId(self):
        return self.m_id
    @ti.func
    def tiSetObjId(self, id):
        self.m_id = id

    def getObjStackTop(self):
        return self.m_stack_top
    def getObjPartNum(self):
        return self.m_part_num
    def getObjPartSize(self):
        return self.m_part_size
    def checkObjStackOpen(self):
        return self.m_if_stack_open
    def setObjStackOpen(self, if_open: bool):
        self.m_if_stack_open = if_open
    def getObjStackOpenNum(self):
        return self.m_stack_open_num
    def setObjStackOpenNum(self, num: int):
        self.m_stack_open_num = num
    def getObjWorld(self):
        return self.m_world
    def setObjWorld(self, world):
        self.m_world = world
    def getObjId(self):
        return self.m_id
    def setObjId(self, id):
        self.m_id = id

    ''' particle basic physical attrs get&set '''

    def getPos(self, i):
        return self.pos[i]
    def getPosArr(self):
        return self.pos
    def getVel(self, i):
        return self.vel[i]
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
    def getPartSize(self, i):
        return self.size[i]
    
    # taichi version
    @ti.func
    def tiGetPos(self, i):
        return self.pos[i]
    @ti.func
    def tiGetPosArr(self):
        return self.pos
    @ti.func
    def tiGetVel(self, i):
        return self.vel[i]
    @ti.func
    def tiGetVelAdv(self, i):
        return self.vel_adv[i]
    @ti.func
    def tiGetAcc(self, i):
        return self.acc[i]
    @ti.func
    def tiGetMass(self, i):
        return self.mass[i]
    @ti.func
    def tiGetVolume(self, i):
        return self.volume[i]
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
    @ti.func
    def tiGetPartSize(self, i):
        return self.size[i]
    
    
    ''' particle sph attrs get&set '''

    def getSphH(self, i):
        return self.sph[i].h
    def getSphSig(self, i):
        return self.sph[i].sig
    def getSphSigInvH(self, i):
        return self.sph[i].sig_inv_h
    def getSphDensity(self, i):
        return self.sph[i].density
    def getSphCompressionRatio(self, i):
        return self.sph[i].compression_ratio
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
    def tiGetSphSig(self, i):
        return self.sph[i].sig
    @ti.func
    def tiGetSphSigInvH(self, i):
        return self.sph[i].sig_inv_h
    @ti.func
    def tiGetSphDensity(self, i):
        return self.sph[i].density
    @ti.func
    def tiGetSphCompressionRatio(self, i):
        return self.sph[i].compression_ratio
    @ti.func
    def tiGetSphPressure(self, i):
        return self.sph[i].pressure
    @ti.func
    def tiGetSphPressureForce(self, i):
        return self.sph[i].pressure_force
    @ti.func
    def tiGetSphViscosityForce(self, i):
        return self.sph[i].viscosity_force
    @ti.func
    def tiGetSphGravityForce(self, i):
        return self.sph[i].gravity_force
    @ti.func
    def tiGetSphAlpha1(self, i):
        return self.sph_df[i].alpha_1
    @ti.func
    def tiGetSphAlpha2(self, i):
        return self.sph_df[i].alpha_2
    @ti.func
    def tiGetSphAlpha(self, i):
        return self.sph_df[i].alpha
    @ti.func
    def tiGetSphKappaIncomp(self, i):
        return self.sph_df[i].kappa_incomp
    @ti.func
    def tiGetSphKappaDiv(self, i):
        return self.sph_df[i].kappa_div
    @ti.func
    def tiGetSphDeltaDensity(self, i):
        return self.sph_df[i].delta_density
    @ti.func
    def tiGetSphDeltaCompressionRatio(self, i):
        return self.sph_df[i].delta_compression_ratio
    @ti.func
    def tiGetSphVelAdv(self, i):
        return self.sph_df[i].vel_adv
    @ti.func
    def tiGetSphB(self, i):
        return self.sph_wc[i].B
    
    