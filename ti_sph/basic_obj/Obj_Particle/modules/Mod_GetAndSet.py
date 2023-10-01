import taichi as ti

@ti.data_oriented
class Mod_GetAndSet:
# tiGet functions

    ''' particle class attrs get&set '''
    @ti.func
    def tiGet_stack_top(self:ti.template()):
        return self.m_stack_top
    @ti.func
    def tiGet_part_num(self):
        return self.m_part_num
    @ti.func
    def tiGet_part_size(self):
        return self.m_part_size
    @ti.func
    def tiCheck_stackOpen(self):
        return self.m_if_stack_open
    @ti.func
    def tiGet_stackOpenNum(self):
        return self.m_stack_open_num
    @ti.func
    def tiSet_stackOpenNum(self, num: ti.i32):
        self.m_stack_open_num = num
    @ti.func
    def tiGet_world(self):
        return self.m_world
    @ti.func
    def tiSet_world(self, world):
        self.m_world = world
    @ti.func
    def tiGet_id(self):
        return self.m_id
    @ti.func
    def tiSet_id(self, id):
        self.m_id = id

    def get_stack_top(self):
        return self.m_stack_top
    def get_part_num(self):
        return self.m_part_num
    def get_part_size(self):
        return self.m_part_size
    def check_stackOpen(self):
        return self.m_if_stack_open
    def set_stackOpen(self, if_open: bool):
        self.m_if_stack_open = if_open
    def get_stackOpenNum(self):
        return self.m_stack_open_num
    def set_stackOpenNum(self, num: int):
        self.m_stack_open_num = num
    def get_world(self):
        return self.m_world
    def set_world(self, world):
        self.m_world = world
    def get_id(self):
        return self.m_id
    def set_id(self, id):
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