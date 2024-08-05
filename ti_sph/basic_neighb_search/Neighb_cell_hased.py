import taichi as ti
import numpy as np
from ..basic_op import *
from ..basic_obj.Obj_Particle import Particle
from ..basic_solvers.Solver import Solver

@ti.data_oriented
class Neighb_cell_hased(Solver):
    def __init__(
            self,
            obj: Particle, # Particle class
    ):
        # get all parameters from obj
        self.obj       =    obj
        self.dim       =    obj.getPosArr().n
        self.cell_size =    obj.getWorld().getSupportRadius()
        self.lb        =    obj.getWorld().getLb()
        self.rt        =    obj.getWorld().getRt()

        self.attach_to_obj(obj)
        
        self.part_id             = ti.field(ti.u32, self.getObj().getPartNum())
        self.cell_total_part_num = ti.field(ti.u32, self.getObj().getPartNum())
        self.hashed_part_id      = ti.field(ti.u32, self.getObj().getPartNum())
        self.start_point         = ti.field(ti.u32, self.getObj().getPartNum())
        naturalSeq(self.part_id)

    @ti.kernel
    def hash(self):
        for i in range(self.getObj().getStackTop()):
            self.hashed_part_id[i] = self.get_hash(self.getObj().getPos(i))
        for i in range(self.getObj().getPartNum()-self.getObj().getStackTop()):
            self.hashed_part_id[i+self.getObj().getStackTop()] = 0xFFFFFFFF
    
    def sortPart(self):
        ti.algorithms.parallel_sort(self.hashed_part_id, self.part_id)

    @ti.kernel
    def computeStartPointAndPartNumInCell(self):
        self.hashed_part_id.fill(ti.u32(0xFFFFFFFF))
        self.cell_total_part_num.fill(0)
        self.start_point[self.hashed_part_id[0]] = 0
        for i in range(1, self.getObj().getPartNum()):
            if self.hashed_part_id[i] != self.hashed_part_id[i-1]:
                self.start_point[self.hashed_part_id[i]] = i
        for i in range(self.getObj().getPartNum()):
            self.cell_total_part_num[self.hashed_part_id[i]] += 1
    
    @ti.kernel
    def loop(self):
        for i in range(self.getObj().getPartNum()):
            hash             = self.get_hash(self.getObj().getPos(i))
            pointer          = self.hashed_part_id[hash]
            part_num_in_cell = self.cell_total_part_num[pointer]
            for j in range(part_num_in_cell):
                neighb_part_id = self.part_id[pointer+j]

    @ti.func
    def get_hash(self, pos):
        seed = ti.static(124232,43351221,4623421)
        cell_vec = self.locate_part_to_cell(pos)
        hash_val = 0
        for j in ti.static(range(self.dim)):
            hash_val += ti.u32(cell_vec[j]) * seed[j] & ti.u32(0xFFFFFFFF)
        return hash_val % self.getObj().getStackTop()
        

    @ti.func
    def locate_part_to_cell(
        self,
        part_pos: ti.template(),  # 粒子的位置
    ):
        return ti.floor((part_pos - self.lb) // self.cell_size).cast(ti.i32)
    
    def attach_to_obj(self, obj):
        self.obj = obj
        # if obj does not have neighb_cell, then create it and attach self to it
        if not hasattr(obj, "neighb_cell"):
            obj.neighb_cell = self
        else:
            raise Exception("obj already has neighb_cell")