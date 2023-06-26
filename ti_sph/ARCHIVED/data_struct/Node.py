# data structure for particles and neighbor search cells

import taichi as ti
from .struct.constructor import *
from ..func_util import *


@ti.data_oriented
class Node:
    def __init__(
        self,
        dim,
        node_num,
        capacity_list,
        id=0,
    ):

        self.info = self.node() # some information for the entire Node object (not particle-wise)
        self.info.id[None] = id
        self.info.dim = dim                     # dimensions (e.g. 2D, 3D)
        self.info.node_num[None] = node_num     # total number of nodes (max number of particles)
        self.info.stack_top[None] = 0           # number of used nodes (actual number of particles)
        self.capacity_list = capacity_list

        node_construct(self, dim, node_num, capacity_list)

    # for initializing self.info (define the struct for self.info)
    def node(self):
        struct_node = ti.types.struct(
            id=ti.i32,
            node_num=ti.i32,
            stack_top=ti.i32,
            dim=ti.i32,
            neighb_cell_num=ti.i32,
        )
        return struct_node.field(shape=())

    # ====================================================================================
    # setting and modifying particle-wise attributes
    # field length and dimensions are NOT checked, need to make sure they match

    # set obj_attr[begin_index:begin_index+pushed_node_num] to element values in attr_seq
    @ti.kernel
    def push_attr_seq(
        self,
        obj_attr: ti.template(),
        attr_seq: ti.template(),
        begin_index: ti.i32,
        pushed_node_num: ti.i32,
    ):
        dim = ti.static(attr_seq.n)
        for i in range(pushed_node_num):
            i_p = i + begin_index
            for j in ti.static(range(dim)):
                obj_attr[i_p][j] = attr_seq[i][j]

    # set all elements in obj_attr to 0
    @ti.kernel
    def clear(
        self,
        obj_attr: ti.template(),
    ):
        for i in range(self.info.stack_top[None]):
            obj_attr[i] *= 0

    # set elements in obj_attr[begin_index:begin_index+pushed_node_num] to attr
    @ti.kernel
    def push_attr(
        self,
        obj_attr: ti.template(),
        attr: ti.template(),
        begin_index: ti.i32,
        pushed_node_num: ti.i32,
    ):
        for i in range(begin_index, begin_index + pushed_node_num):
            obj_attr[i] = attr

    # set all elements of obj_attr to val
    @ti.kernel
    def attr_set(
        self,
        obj_attr: ti.template(),
        val: ti.template(),
    ):
        for i in range(self.info.stack_top[None]):
            obj_attr[i] = val[None]

    # set all elements of obj_attr to element values in val_arr
    @ti.kernel
    def attr_set_arr(
        self,
        obj_attr: ti.template(),
        val_arr: ti.template(),
    ):
        for i in range(self.info.stack_top[None]):
            obj_attr[i] = val_arr[i]

    # add val to all elements of obj_attr
    @ti.kernel
    def attr_add(
        self,
        obj_attr: ti.template(),
        val: ti.template(),
    ):
        for i in range(self.info.stack_top[None]):
            obj_attr[i] += val[None]

    # add elements in val_arr to each element in obj_attr
    @ti.kernel
    def attr_add_arr(
        self,
        obj_attr: ti.template(),
        val_arr: ti.template(),
    ):
        for i in range(self.info.stack_top[None]):
            obj_attr[i] += val_arr[i]

    # multiply each element in obj_attr with val
    @ti.kernel
    def resize(
        self,
        obj_attr: ti.template(),
        val: ti.template(),
    ):
        for i in range(self.info.stack_top[None]):
            obj_attr[i] = obj_attr[i] * val

    # ====================================================================================
    # add particles with certain positions

    # add pushed_node_num particles, and set their positions to elements in pos_seq
    def push_pos_seq(
        self,
        obj,
        pushed_node_num,
        pos_seq,
    ):
        print("push ", pushed_node_num, " nodes")
        dim = pos_seq.shape[0]
        current_node_num = obj.info.stack_top[None]
        new_node_num = current_node_num + pushed_node_num
        pos_seq_ti = ti.Vector.field(dim, ti.Vector.field, pushed_node_num)
        pos_seq_ti.from_numpy(pos_seq)
        self.push_attr_seq(obj.basic.pos, pos_seq_ti, pushed_node_num, current_node_num)
        obj.info.stack_top[None] = new_node_num
        return pushed_node_num

    # add particles that form the shape of a cuboid (not cube), center of the particle with smallest coordinate in each dimension is aligned to lb
    # lb: corner with smallest coordinate in each dimension
    # rt: corner with largest coordinate in each dimension
    # span: distance between particles
    @ti.kernel
    def push_cube(
        self,
        lb: ti.template(),
        rt: ti.template(),
        span: ti.f32,
    ) -> ti.i32:
        current_node_num = self.info.stack_top[None]
        pushed_node_seq_coder = ti.Vector([0, 0, 0])
        pushed_node_seq = int(ti.ceil((rt - lb) / span))
        dim = ti.static(lb.n)
        for i in ti.static(range(dim)):
            if pushed_node_seq[i] == 0:
                pushed_node_seq[i] = 1  # at least push one
            # coder for seq
        tmp = 1
        for i in ti.static(range(dim)):
            pushed_node_seq_coder[i] = tmp
            tmp *= pushed_node_seq[i]
        # new node num
        pushed_node_num = 1
        for i in ti.static(range(dim)):
            pushed_node_num *= pushed_node_seq[i]
        new_node_num = current_node_num + pushed_node_num
        if new_node_num > self.info.node_num[None]:
            print("WARNING from push_cube(): overflow") # exceeded self.info.node_num[None]
        # inject pos [1/2]
        for i in range(pushed_node_num):
            tmp = i
            for j in ti.static(range(dim - 1, -1, -1)):
                self.basic.pos[i + current_node_num][j] = (
                    tmp // pushed_node_seq_coder[j]
                )
                tmp = tmp % pushed_node_seq_coder[j]
        # inject pos [2/2]
        # pos seq times node span minus lb
        for i in range(pushed_node_num):
            self.basic.pos[i + current_node_num] *= span
            self.basic.pos[i + current_node_num] += lb
        # inject volume_frac & rest_volume & color
        # for i in range(pushed_node_num):
        #     self.basic.rest_volume[i + current_node_num] = part_size**3
        #     self.basic.radius[i + current_node_num] = part_size
        # update ndoe num
        self.info.stack_top[None] = new_node_num
        return pushed_node_num

    # add particles that form the shape of a box, if a particle is placed inside the hollow part of the box, at the position with smallest coordinate in each dimension, its center is aligned to lb
    # lb: corner with smallest coordinate in each dimension (box's inside)
    # rt: corner with largest coordinate in each dimension (box's inside)
    # span: distance between particles
    # layers: layers of particles of the box
    @ti.kernel
    def push_box(
        self,
        lb: ti.template(),
        rt: ti.template(),
        span: ti.f32,
        layers: ti.i32,
    ) -> ti.i32:
        dim = ti.static(lb.n)

        current_node_num = self.info.stack_top[None]
        pushed_node_seq_coder = ti.Vector([0, 0, 0])

        pushed_node_seq = int(ti.ceil((rt - lb) / span))
        pushed_node_seq_offset = int(ti.ceil((rt - lb) / span)) + (layers * 2)

        for i in ti.static(range(dim)):
            if pushed_node_seq[i] == 0:
                pushed_node_seq[i] = 1  # at least push one

        tmp = 1
        for i in ti.static(range(dim)):
            pushed_node_seq_coder[i] = tmp
            tmp *= pushed_node_seq_offset[i]

        pushed_node_num = 1
        pushed_node_num_solid = 1
        for i in ti.static(range(dim)):
            pushed_node_num *= pushed_node_seq[i]
            pushed_node_num_solid *= pushed_node_seq_offset[i]
        pushed_node_num = pushed_node_num_solid - pushed_node_num
        new_node_num = current_node_num + pushed_node_num

        if new_node_num > self.info.node_num[None]:
            print("WARNING from push_box(): overflow")

        inc = ti.Vector([current_node_num])
        for i in range(pushed_node_num_solid):
            tmp = i
            a = rt - lb
            flag = True
            dim_check = 0
            for j in ti.static(range(dim - 1, -1, -1)):
                a[j] = tmp // pushed_node_seq_coder[j]
                tmp = tmp % pushed_node_seq_coder[j]
            if has_positive((a - 1) - pushed_node_seq) or has_negative(a - layers):
                index = ti.atomic_add(inc[0], 1)
                for j in ti.static(range(dim - 1, -1, -1)):
                    self.basic.pos[index][j] = (a[j] - layers) * span + lb[j]
        self.info.stack_top[None] += inc[0]
        return inc[0]

    def update_vel(
        self,
        dt,
    ):
        self.update_vel_ker(
            self,
            dt,
            self.basic.acc,
            self.basic.vel,
        )

    @ti.kernel
    def update_vel_ker(
        self,
        obj: ti.template(),
        dt: ti.f32,
        obj_acc: ti.template(),
        obj_output_vel: ti.template(),
    ):
        for i in range(obj.info.stack_top[None]):
            obj_output_vel[i] += obj_acc[i] * dt

    def update_pos(self, dt):
        self.update_pos_ker(
            self,
            dt,
            self.basic.vel,
            self.basic.pos,
        )

    @ti.kernel
    def update_pos_ker(
        self,
        obj: ti.template(),
        dt: ti.f32,
        obj_vel: ti.template(),
        obj_output_pos: ti.template(),
    ):
        for i in range(obj.info.stack_top[None]):
            obj_output_pos[i] += obj_vel[i] * dt

    def push_box_with_basic_attr(
        self,
        lb,
        rt,
        span,
        layers,
        size,
        rest_density,
        color=ti.Vector([0.3, 0.3, 0.3]),
    ):
        dim = self.basic.pos[0].n

        pushed_num = self.push_box(
            lb,
            rt,
            span,
            layers,
        )

        self.push_attr(
            self.basic.size,
            size,
            self.info.stack_top[None] - pushed_num,
            pushed_num,
        )
        self.push_attr(
            self.basic.rest_volume,
            size**dim,
            self.info.stack_top[None] - pushed_num,
            pushed_num,
        )
        self.push_attr(
            self.basic.rest_density,
            rest_density,
            self.info.stack_top[None] - pushed_num,
            pushed_num,
        )
        self.push_attr(
            self.basic.mass,
            rest_density * (size**dim),
            self.info.stack_top[None] - pushed_num,
            pushed_num,
        )
        self.push_attr(
            self.color.vec,
            color,
            self.info.stack_top[None] - pushed_num,
            pushed_num,
        )

        return pushed_num

    def push_cube_with_basic_attr(
        self,
        lb,
        rt,
        span,
        size,
        rest_density,
        color=ti.Vector([0.3, 0.3, 0.3]),
    ):
        dim = self.basic.pos[0].n

        pushed_num = self.push_cube(
            lb,
            rt,
            span,
        )

        self.push_attr(
            self.basic.size,
            size,
            self.info.stack_top[None] - pushed_num,
            pushed_num,
        )
        self.push_attr(
            self.basic.rest_volume,
            size**dim,
            self.info.stack_top[None] - pushed_num,
            pushed_num,
        )
        self.push_attr(
            self.basic.rest_density,
            rest_density,
            self.info.stack_top[None] - pushed_num,
            pushed_num,
        )
        self.push_attr(
            self.basic.mass,
            rest_density * (size**dim),
            self.info.stack_top[None] - pushed_num,
            pushed_num,
        )
        self.push_attr(
            self.color.vec,
            color,
            self.info.stack_top[None] - pushed_num,
            pushed_num,
        )

        return pushed_num

    def push_ply(
            self,
            pos_seq: ti.template(),
    ):
        current_node_num = self.info.stack_top[None]
        pushed_node_num = len(pos_seq)
        new_node_num = current_node_num + pushed_node_num
        if new_node_num > self.info.node_num[None]:
            print("WARNING from push_ply(): overflow")

        dim = ti.static(self.basic.pos.n)
        for i in range(pushed_node_num):
            for j in ti.static(range(dim)):
                self.basic.pos[i + current_node_num][j] = pos_seq[i, j]

        self.info.stack_top[None] = new_node_num
        return pushed_node_num

    def push_ply_with_basic_attr(
            self,
            pos_seq,
            size,
            rest_density,
            color=ti.Vector([0.3, 0.3, 0.3]),
    ):
        dim = self.basic.pos[0].n

        pushed_num = self.push_ply(pos_seq)

        self.push_attr(
            self.basic.size,
            size,
            self.info.stack_top[None] - pushed_num,
            pushed_num,
        )
        self.push_attr(
            self.basic.rest_volume,
            size ** dim,
            self.info.stack_top[None] - pushed_num,
            pushed_num,
        )
        self.push_attr(
            self.basic.rest_density,
            rest_density,
            self.info.stack_top[None] - pushed_num,
            pushed_num,
        )
        self.push_attr(
            self.basic.mass,
            rest_density * (size ** dim),
            self.info.stack_top[None] - pushed_num,
            pushed_num,
        )
        self.push_attr(
            self.color.vec,
            color,
            self.info.stack_top[None] - pushed_num,
            pushed_num,
        )

        return pushed_num
