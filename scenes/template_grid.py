import taichi as ti
from ti_sph.basic_op import *
from ti_sph import *
import numpy as np

def grid_template(part_obj:Particle, world, verbose=False):
    part_obj.add_array("pos", vecxf(part_obj.getWorld().getDim()).field())
    part_obj.add_array("size", ti.field(ti.f32))
    part_obj.add_array("volume", ti.field(ti.f32))
    part_obj.add_array("mass", ti.field(ti.f32))
    part_obj.add_array("rgb", vecxf(3).field())
    part_obj.add_array("node_index", vecxi(part_obj.getWorld().getDim()).field())

    part_obj.add_array("sensed_density", ti.field(ti.f32))
    part_obj.add_array("vel", vecxf(part_obj.getWorld().getDim()).field())
    part_obj.add_array("strainRate", vecxm(part_obj.getWorld().getDim(),part_obj.getWorld().getDim()).field())

    sph = ti.types.struct(
        h=ti.f32,
        sig=ti.f32,
        sig_inv_h=ti.f32,
        density=ti.f32,
        compression_ratio=ti.f32,
        pressure=ti.f32,
        pressure_force=vecxf(part_obj.getWorld().getDim()),
        viscosity_force=vecxf(part_obj.getWorld().getDim()),
        gravity_force=vecxf(part_obj.getWorld().getDim()),
    )

    part_obj.add_struct("sph", sph)

    return part_obj
