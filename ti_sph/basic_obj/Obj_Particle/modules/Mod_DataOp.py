import taichi as ti
import numpy as np
from .Mod_GetAndSet import Mod_GetAndSet

@ti.data_oriented
class Mod_DataOp(Mod_GetAndSet):

    def update_stack_top(self, num: int):
        self.AddStackTop(num)

    def open_stack(self, open_num: int):
        if self.m_if_stack_open:
            raise Exception("Particle Stack already opened!")
            exit(0)

        if self.getStackTop() + open_num > self.m_part_num[None]:
            raise Exception("Particle Stack overflow!", self.getStackTop(), open_num, self.m_part_num[None])
            exit(0)

        self.setStackOpen(True)
        self.setStackOpenNum(open_num)

    def fill_open_stack_with_nparr(self, attr_: ti.template(), data: ti.types.ndarray()):
        data_dim = len(data.shape)
        if data_dim == 1:
            data_ti_container = ti.field(ti.f32, self.m_stack_open_num[None])
        elif data_dim == 2:
            data_ti_container = ti.Vector.field(data.shape[1], ti.f32, self.m_stack_open_num[None])
        else:
            raise Exception("Data dimension not supported!")
            exit(0)
        
        data_ti_container.from_numpy(data.astype(np.float32))
        self.fill_open_stack_with_arr(attr_, data_ti_container)

    @ti.kernel
    def fill_open_stack_with_arr(self, attr_: ti.template(), data: ti.template()):
        for i in range(self.m_stack_open_num[None]):
            attr_[i+self.tiGetStackTop()] = data[i]

    @ti.kernel
    def fill_open_stack_with_val(self, attr_: ti.template(), val: ti.template()):
        for i in range(self.tiGetStackOpenNum()):
            attr_[i+self.tiGetStackTop()] = val

    @ti.kernel
    def fill_open_stack_with_vals(self, attr_: ti.template(), val: ti.template()):
        for i in range(self.m_stack_open_num[None]):
            for j in range(ti.static(val.shape[0])):
                attr_[i+self.tiGetStackTop(), j] = val[j]

    def close_stack(self):
        if not self.m_if_stack_open:
            raise Exception("Particle Stack not opened!")
            exit(0)

        self.m_if_stack_open = False
        self.AddStackTop(self.getStackOpenNum())
        self.setStackOpenNum(0)


    @ti.func
    def has_negative(self, val: ti.template()):
        for dim in ti.static(range(self.m_world.dim[None])):
            if val[dim] < 0:
                return True
        return False

    @ti.func
    def has_positive(self, val: ti.template()):
        for dim in ti.static(range(self.m_world.dim[None])):
            if val[dim] > 0:
                return True
        return False


    @ti.kernel
    def clear(self, attr_: ti.template()):
        for i in range (self.tiGetStackTop()):
            attr_[i] *= 0

    @ti.kernel
    def copy_attr(self, from_attr: ti.template(), to_attr: ti.template()):
        for i in range(self.tiGetStackTop()):
            to_attr[i] = from_attr[i]

    def set_from_numpy(self, to: ti.template(), data: ti.types.ndarray()):
        num = data.shape[0]
        arr = to.to_numpy()
        arr[self.tiGetStackTop():num, :] = data
        to.from_numpy(arr)

    @ti.kernel
    def set_val(self, to_arr: ti.template(), num: ti.i32, val: ti.template()):
        for i in range(num):
            to_arr[i+self.tiGetStackTop()] = val[None]

    @ti.kernel
    def set(self, to_arr: ti.template(), val: ti.template()):
        for i in range(self.tiGetStackTop()):
            to_arr[i] = val[None]

    @ti.kernel
    def clamp_val_to_arr(self, arr: ti.template(), lower: ti.f32, upper: ti.f32, to: ti.template()):
        for i in range(self.tiGetStackTop()):
            for j in range(ti.static(to.n)):
                to[i][j] = ti.min(ti.max(arr[i] / (upper - lower),0),1)

    @ti.kernel
    def clamp_val(self, arr: ti.template(), lower: ti.f32, upper: ti.f32, to: ti.template()):
        for i in range(self.tiGetStackTop()):
            to[i] = ti.min(ti.max(arr[i] / (upper - lower),0),1)

    @ti.kernel
    def integ(self, arr: ti.template(), int_val: ti.f32, to: ti.template()):
        for i in range(self.tiGetStackTop()):
            to[i] += arr[i] * int_val