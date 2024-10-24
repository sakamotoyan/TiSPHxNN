import taichi as ti
import numpy as np
import time
from ..basic_op import *
from ..basic_solvers.sph_funcs import *
from ..basic_obj.Obj_Particle import Particle
from ..basic_solvers.Solver import Solver

@ti.data_oriented
class Neighb_search_template(Solver):
    def __init__(
        self,
        dim,  # int32
        search_cell_range,  # int32
    ):
        self.search_cell_range = val_i(0)
        self.search_cell_range[None] = search_cell_range

        self.search_template = ti.Vector.field(
            dim, ti.i32, (self.search_cell_range[None] * 2 + 1) ** dim
        )

        self.neighb_dice = ti.field(
            ti.i32, self.search_cell_range[None] * 2 + 1)

        for i in range(self.neighb_dice.shape[0]):
            self.neighb_dice[i] = -self.search_cell_range[None] + i

        for i in range(self.search_template.shape[0]):
            tmp = i
            for j in ti.static(range(dim)):
                digit = tmp // (self.neighb_dice.shape[0] ** (dim - j - 1))
                tmp = tmp % (self.neighb_dice.shape[0] ** (dim - j - 1))
                self.search_template[i][dim - j - 1] = self.neighb_dice[digit]

    @ti.func
    def get_neighb_cell_num(self):
        return self.search_template.shape[0]
    
    @ti.func
    def get_neighb_cell_vec(self, i):
        return self.search_template[i]

@ti.data_oriented
class Neighb_search_hashed(Solver):
    def __init__(
            self,
            obj: Particle, # Particle class
            neighb_obj_list: list = [],  # list of Particle class
            max_neighb_part_num: ti.template() = 0,  # int32
    ):
        Solver.__init__(self, obj)

        if max_neighb_part_num == 0:
            self.max_neighb_part_num = val_i(obj.getPartNum() * self.getObj().getWorld().g_avg_neighb_part_num[None])
            # self.max_neighb_part_num = None
        else:
            self.max_neighb_part_num = val_i(max_neighb_part_num[None])
        print("part_num: ", self.getObj().getPartNum())
        print("max_neighb_part_num: ", self.max_neighb_part_num)
        self.max_neighb_obj_num = val_i(self.getObj().getWorld().g_obj_num[None])

        # get all parameters from obj
        self.obj          = obj
        self.dim          = obj.getPosArr().n
        self.cell_size    = obj.getWorld().getSupportRadius()
        self.search_range = obj.getWorld().getSupportRadius()
        self.lb           = obj.getWorld().getLb()
        self.rt           = obj.getWorld().getRt()
        
        self.cell_num            = ti.max(self.getObj().getPartNum() + 983, self.getObj().getPartNum()*2)
        self.part_id             = ti.field(ti.u32, self.getObj().getPartNum())
        self.hashed_part_cellId  = ti.field(ti.u32, self.getObj().getPartNum())
        self.cell_total_part_num = ti.field(ti.u32, self.cell_num)
        self.ptr_start_point     = ti.field(ti.u32, self.cell_num)
        self.ptr_inc             = ti.field(ti.u32, self.cell_num)

        self.neighb_obj_list = neighb_obj_list
        self.neighb_obj_num  = len(neighb_obj_list)
        self.neighb_search_template_list = []

        self.partNeighb_size            = ti.field(ti.i32) # [part_id]
        self.partNeighbObj_begin        = ti.field(ti.i32) # [part_id, neighb_obj_id]
        self.partNeighbObj_size         = ti.field(ti.i32) # [part_id, neighb_obj_id]
        self.neighb_pool_type           = ti.types.struct( # [begining + shift pointer]
            neighbPartId                = ti.i32,
            neighbObjId                 = ti.i32,
            dist                        = ti.f32,
            W                           = ti.f32,
            xijNorm                     = ti.types.vector(self.getObj().getWorld().getDim(), ti.f32),
            gradW                       = ti.types.vector(self.getObj().getWorld().getDim(), ti.f32),
        )

        ti.root.dense(ti.i,   self.obj.getPartNum()).place(self.partNeighb_size)
        ti.root.dense(ti.ij, (self.obj.getPartNum(), self.max_neighb_obj_num[None])).place(
            self.partNeighbObj_begin,
            self.partNeighbObj_size)

        self.neighb_pool            = self.neighb_pool_type.field(shape=self.max_neighb_part_num[None])
        self.neighb_pool_used_space = val_i(0)
        self.neighb_pool_size       = ti.static(self.max_neighb_part_num)

    def init_module(self):
        for neighb_obj in self.neighb_obj_list:
            neighb_obj: Particle
            search_cell_range = int(ti.ceil(self.search_range / neighb_obj.get_module_neighbSearch().get_cellSize()))
            self.neighb_search_template_list.append(Neighb_search_template(self.dim, search_cell_range))
    
    @ti.kernel
    def loop_neighb(self, neighb_obj:ti.template(), func:ti.template()):
        # neighb_search_module = self.tiGetObj().tiGet_module_neighbSearch()
        for part_id in range(self.tiGetObj().tiGetStackTop()):
            neighbPart_num              = self.tiGet_partNeighbObjSize           (part_id, neighb_obj.tiGetId())
            neighbPool_begining_pointer = self.tiGet_partNeighbObjBeginingPointer(part_id, neighb_obj.tiGetId())
            for shift in range(neighbPart_num):
                neighbPool_pointer = neighbPool_begining_pointer + shift
                neighbPart_id = self.tiGet_neighbPartId(neighbPool_pointer)
                ''' Code for Computation'''
                func(part_id, neighbPart_id, neighbPool_pointer, self, neighb_obj)
                ''' End of Code for Computation'''

    def check_neighb(self, pid:ti.i32):
        neighbPart_num              = self.get_partNeighbObjSize           (pid, self.getObj().getId())
        neighbPool_begining_pointer = self.get_partNeighbObjBeginingPointer(pid, self.getObj().getId())
        print("part ", pid, " neighbPart_num:", neighbPart_num)
        for shift in range(neighbPart_num):
                neighbPool_pointer = neighbPool_begining_pointer + shift
                neighbPart_id = self.get_neighbPartId(neighbPool_pointer)
                ''' Code for Computation'''
                dist = self.get_cachedDist(neighbPool_pointer)
                print("neighbPart_id: ", neighbPart_id, "dist: ", dist)

    def check_neighb_all(self):
        for pid in range(self.getObj().getPartNum()):
            self.check_neighb(pid)

    @ti.kernel
    def loop_neighb_ex(self, neighb_obj:ti.template(), neighb_search_module:ti.template(), func:ti.template()):
        for part_id in range(self.tiGetObj().tiGetStackTop()):
            neighbPart_num              = neighb_search_module.tiGet_partNeighbObjSize           (part_id, neighb_obj.tiGetId())
            neighbPool_begining_pointer = neighb_search_module.tiGet_partNeighbObjBeginingPointer(part_id, neighb_obj.tiGetId())
            for shift in range(neighbPart_num):
                neighbPool_pointer = neighbPool_begining_pointer + shift
                neighbPart_id = neighb_search_module.tiGet_neighbPartId(neighbPool_pointer)
                ''' Code for Computation'''
                func(part_id, neighbPart_id, neighbPool_pointer, neighb_search_module, neighb_obj)
                ''' End of Code for Computation'''

    @ti.kernel
    def loop_self(self, func:ti.template()):
        for part_id in range(self.tiGetObj().tiGetStackTop()):
            neighbPart_num              = self.tiGet_partNeighbObjSize           (part_id, self.tiGetObj().tiGetId())
            neighbPool_begining_pointer = self.tiGet_partNeighbObjBeginingPointer(part_id, self.tiGetObj().tiGetId())
            for shift in range(neighbPart_num):
                neighbPool_pointer = neighbPool_begining_pointer + shift
                neighbPart_id = self.tiGet_neighbPartId(neighbPool_pointer)
                ''' Code for Computation'''
                func(part_id, neighbPart_id, neighbPool_pointer, self, self.tiGetObj())
                ''' End of Code for Computation'''

    def update(self,):
        # start_time = time.time()  
        # # Record the start time
        # naturalSeq(self.part_id)
        # # Record the end time
        # end_time = time.time()  
        # execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
        # print(f"    Execution time naturalSeq(): {execution_time:.2f} ms")

        # start_time = time.time()
        # # Record the start time
        # self.hash()
        # # Record the end time
        # end_time = time.time()
        # execution_time = (end_time - start_time) * 1000
        # print(f"    Execution time hash(): {execution_time:.2f} ms")

        # start_time = time.time()
        # # Record the start time
        # self.sortPart()
        # # Record the end time
        # end_time = time.time()
        # execution_time = (end_time - start_time) * 1000
        # print(f"    Execution time sortPart(): {execution_time:.2f} ms")

        # start_time = time.time()
        # # Record the start time
        # self.computeStartPointAndPartNumInCell()
        # # Record the end time
        # end_time = time.time()
        # execution_time = (end_time - start_time) * 1000
        # print(f"    Execution time computeStartPointAndPartNumInCell(): {execution_time:.2f} ms")
        
        # NEW
        self.hash8log()


    def pool(self,):
        self.neighb_pool_used_space[None] = 0
        for i in range(self.neighb_obj_num):
            self.pool_a_neighbObj(self.neighb_obj_list[i], self.neighb_search_template_list[i])
        if not self.neighb_pool_size[None] > self.neighb_pool_used_space[None]:
            raise Exception(f"neighb_pool overflow, need {self.neighb_pool_used_space[None]} but only {self.neighb_pool_size[None]}")

    @ti.kernel
    def clear_pool(self):
        for part_id in range(self.tiGetObj().tiGetStackTop()):
            self.tiSet_partNeighbBeginingPointer(part_id, -1)
            self.tiSet_partNeighbCurrnetPointer(part_id, -1) 
            self.tiSet_partNeighbSize(part_id, 0)
            for obj_seq in range(self.max_neighb_obj_num[None]):
                self.tiSet_partNeighbObjBeginingPointer(part_id, obj_seq, -1)
                self.tiSet_partNeighbObjSize(part_id, obj_seq, 0)
        self.neighb_pool_used_space[None] = 0

    @ti.kernel
    def pool_a_neighbObj(
        self,
        neighbObj: ti.template(),  # Particle class
        neighb_search_template: ti.template(),  # Neighb_search_template class
    ):
        neighbObj_id = neighbObj.tiGetId()
        for part_id in range(self.tiGetObj().tiGetStackTop()):
            self.tiSet_partNeighbSize(part_id, 0)
        #
        ti.sync()
        for part_id in range(self.tiGetObj().tiGetStackTop()):
            part_pos         = self.tiGetObj().tiGetPos(part_id)
            located_cell_vec = neighbObj.tiGet_module_neighbSearch().locate_part_to_cell(part_pos)
            for neighb_cell_iter in range(neighb_search_template.get_neighb_cell_num()):
                cell_vec        = located_cell_vec + neighb_search_template.get_neighb_cell_vec(neighb_cell_iter)
                hash            = ti.u32(neighbObj.tiGet_module_neighbSearch().tiGet_hash(cell_vec))
                pointer         = ti.u32(neighbObj.tiGet_module_neighbSearch().ptr_start_point[hash])
                if pointer == ti.u32(0xFFFFFFFF): continue
                neighb_part_num = neighbObj.tiGet_module_neighbSearch().cell_total_part_num[hash]
                for neighb_part_iter in range(neighb_part_num):
                    neighb_part_id  = neighbObj.tiGet_module_neighbSearch().part_id[pointer+neighb_part_iter]
                    neighb_part_pos = neighbObj.tiGetPos(neighb_part_id)
                    dist            = (part_pos - neighb_part_pos).norm()
                    if dist < self.search_range:
                        self.tiAdd_partNeighbSize(part_id, 1)

            self.tiSet_partNeighbObjBeginingPointer(part_id, neighbObj_id, ti.atomic_add(self.neighb_pool_used_space[None], self.tiGet_partNeighbSize(part_id)))
            self.tiSet_partNeighbObjSize           (part_id, neighbObj_id, self.tiGet_partNeighbSize(part_id))

            shift = 0
            for neighb_cell_iter in range(neighb_search_template.get_neighb_cell_num()):
                cell_vec        = located_cell_vec + neighb_search_template.get_neighb_cell_vec(neighb_cell_iter)
                hash            = ti.u32(neighbObj.tiGet_module_neighbSearch().tiGet_hash(cell_vec))
                pointer         = ti.u32(neighbObj.tiGet_module_neighbSearch().ptr_start_point[hash])
                if pointer == ti.u32(0xFFFFFFFF): continue
                neighb_part_num = neighbObj.tiGet_module_neighbSearch().cell_total_part_num[hash]
                for neighb_part_iter in range(neighb_part_num):
                    neighb_part_id  = neighbObj.tiGet_module_neighbSearch().part_id[pointer+neighb_part_iter]
                    neighb_part_pos = neighbObj.tiGetPos(neighb_part_id)
                    dist            = (part_pos - neighb_part_pos).norm()
                    if dist < self.search_range:
                        pool_pointer = self.tiGet_partNeighbObjBeginingPointer(part_id, neighbObj_id) + ti.atomic_add(shift, 1)
                        self.tiSet_neighbObjId  (pool_pointer, neighbObj_id)
                        self.tiSet_neighbPartId (pool_pointer, neighb_part_id)
                        self.tiSet_cachedDist   (pool_pointer, dist)
                        self.tiSet_cachedXijNorm(pool_pointer, (part_pos - neighb_part_pos) / dist)
                        self.tiSet_cachedW      (pool_pointer, spline_W(dist, self.tiGetObj().tiGetSphH(part_id), self.tiGetObj().tiGetSphSig(part_id)))
                        self.tiSet_cachedGradW  (pool_pointer, grad_spline_W(dist, self.tiGetObj().tiGetSphH(part_id), self.tiGetObj().tiGetSphSigInvH(part_id)) * self.tiGet_cachedXijNorm(pool_pointer))

    @ti.kernel
    def hash8log(self):
        self.hashed_part_cellId .fill(ti.u32(0xFFFFFFFF))
        self.ptr_start_point    .fill(ti.u32(0xFFFFFFFF))
        self.cell_total_part_num.fill(0)
        increment = 0
        for i in range(self.tiGetObj().tiGetStackTop()):
            cell_vec                    = self.locate_part_to_cell(self.tiGetObj().tiGetPos(i))
            self.hashed_part_cellId[i]  = self.tiGet_hash(cell_vec)
            self.cell_total_part_num[self.hashed_part_cellId[i]] += 1
        ti.sync()
        for i in range(self.cell_num):
            if self.cell_total_part_num[i] == 0: continue
            self.ptr_start_point[i] = ti.atomic_add(increment, self.cell_total_part_num[i])
            self.ptr_inc[i]         = self.ptr_start_point[i]
        ti.sync()
        for i in range(self.tiGetObj().tiGetStackTop()):
            self.part_id[ti.atomic_add(self.ptr_inc[self.hashed_part_cellId[i]],1)] = i

    @ti.kernel
    def hash(self):
        self.hashed_part_cellId.fill(ti.u32(0xFFFFFFFF))
        for i in range(self.tiGetObj().tiGetStackTop()):
            cell_vec               = self.locate_part_to_cell(self.tiGetObj().tiGetPos(i))
            self.hashed_part_cellId[i] = self.tiGet_hash(cell_vec)
    
    def sortPart(self):
        ti.algorithms.parallel_sort(self.hashed_part_cellId, self.part_id)

    @ti.kernel
    def computeStartPointAndPartNumInCell(self):
        self.ptr_start_point.fill(ti.u32(0xFFFFFFFF))
        self.cell_total_part_num.fill(0)
        self.ptr_start_point[self.hashed_part_cellId[0]] = 0
        for i in range(1, self.tiGetObj().tiGetStackTop()):
            if self.hashed_part_cellId[i] != self.hashed_part_cellId[i-1]:
                self.ptr_start_point[self.hashed_part_cellId[i]] = i
        for i in range(self.tiGetObj().tiGetStackTop()):
            self.cell_total_part_num[self.hashed_part_cellId[i]] += 1

    @ti.func
    def tiGet_hash(self, cell_vec) -> ti.u32:
        seed = ti.static(124232,4335221,4623421)
        hash_val = ti.u32(0)
        for j in ti.static(range(self.dim)):
            hash_val += ti.u32(cell_vec[j]) * seed[j] & ti.u32(0xFFFFFFFF)
        return ti.u32(hash_val % self.cell_num)
        

    @ti.func
    def locate_part_to_cell(
        self,
        part_pos: ti.template(),  # 粒子的位置
    ):
        return ti.floor((part_pos - self.lb) // self.cell_size).cast(ti.i32)

    @ti.func
    def tiGet_cachedDist(self, pointer: ti.i32):
        return self.neighb_pool.dist[pointer]
    def get_cachedDist(self, pointer: ti.i32):
        return self.neighb_pool.dist[pointer]
    @ti.func
    def tiGet_cachedGradW(self, pointer: ti.i32):
        return self.neighb_pool.gradW[pointer]
    @ti.func
    def tiGet_cachedW(self, pointer: ti.i32):
        return self.neighb_pool.W[pointer]
    @ti.func
    def tiGet_cachedXijNorm(self, pointer: ti.i32):
        return self.neighb_pool.xijNorm[pointer]
    @ti.func
    def tiGet_neighbPartId(self, pointer: ti.i32):
        return self.neighb_pool.neighbPartId[pointer]
    def get_neighbPartId(self, pointer: ti.i32):
        return self.neighb_pool.neighbPartId[pointer]
    @ti.func
    def tiGet_neighbObjId(self, pointer: ti.i32):
        return self.neighb_pool.neighbObjId[pointer]
    @ti.func
    def tiGet_nextPointer(self, pointer: ti.i32):
        return self.neighb_pool.next[pointer]
    @ti.func
    def tiGet_partNeighbObjBeginingPointer(self, partId: ti.i32, neighbObj_id: ti.i32):
        return self.partNeighbObj_begin[partId, neighbObj_id]
    def get_partNeighbObjBeginingPointer(self, partId: ti.i32, neighbObj_id: ti.i32):
        return self.partNeighbObj_begin[partId, neighbObj_id]
    @ti.func
    def tiGet_partNeighbObjSize(self, partId: ti.i32, neighbObj_id: ti.i32):
        return self.partNeighbObj_size[partId, neighbObj_id]
    def get_partNeighbObjSize(self, partId: ti.i32, neighbObj_id: ti.i32):
        return self.partNeighbObj_size[partId, neighbObj_id]
    @ti.func
    def tiGet_partNeighbSize(self, partId: ti.i32):
        return self.partNeighb_size[partId]

    @ti.func
    def tiSet_cachedDist(self, pointer: ti.i32, val: ti.f32):
        self.neighb_pool.dist[pointer] = val
    @ti.func
    def tiSet_cachedGradW(self, pointer: ti.i32, val: ti.template()):
        self.neighb_pool.gradW[pointer] = val
    @ti.func
    def tiSet_cachedW(self, pointer: ti.i32, val: ti.f32):
        self.neighb_pool.W[pointer] = val
    @ti.func
    def tiSet_cachedXijNorm(self, pointer: ti.i32, val: ti.template()):
        self.neighb_pool.xijNorm[pointer] = val
    @ti.func
    def tiSet_neighbPartId(self, pointer: ti.i32, val: ti.i32):
        self.neighb_pool.neighbPartId[pointer] = val
    @ti.func
    def tiSet_neighbObjId(self, pointer: ti.i32, val: ti.i32):
        self.neighb_pool.neighbObjId[pointer] = val
    @ti.func
    def tiSet_nextPointer(self, pointer: ti.i32, val: ti.i32):
        self.neighb_pool.next[pointer] = val
    @ti.func
    def tiSet_partNeighbObjBeginingPointer(self, partId: ti.i32, neighbObj_id: ti.i32, val: ti.i32):
        self.partNeighbObj_begin[partId, neighbObj_id] = val
    @ti.func
    def tiSet_partNeighbObjSize(self, partId: ti.i32, neighbObj_id: ti.i32, val: ti.i32):
        self.partNeighbObj_size[partId, neighbObj_id] = val
    @ti.func
    def tiSet_partNeighbSize(self, partId: ti.i32, val: ti.i32):
        self.partNeighb_size[partId] = val
    @ti.func
    def tiAdd_partNeighbSize(self, partId: ti.i32, val: ti.i32):
        ti.atomic_add(self.partNeighb_size[partId], val)    

    def get_cellSize(self):
        return self.cell_size