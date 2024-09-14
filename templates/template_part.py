import taichi as ti
from ti_sph.basic_op import *
from ti_sph import *
import numpy as np

def part_template(part_obj:Particle, verbose=False):

    part_obj.add_array("mass",          ti.field(ti.f32))
    part_obj.add_array("size",          ti.field(ti.f32))
    part_obj.add_array("volume",        ti.field(ti.f32))
    part_obj.add_array("rest_density",  ti.field(ti.f32))
    part_obj.add_array("pressure",      ti.field(ti.f32))
    part_obj.add_array("k_vis",         ti.field(ti.f32))
    part_obj.add_array("vis_1",         ti.field(ti.f32))
    part_obj.add_array("vis_2",         ti.field(ti.f32))
    part_obj.add_array("vel_adv",       vecxf(part_obj.getWorld().getDim()).field())
    part_obj.add_array("vel",           vecxf(part_obj.getWorld().getDim()).field())
    part_obj.add_array("pos",           vecxf(part_obj.getWorld().getDim()).field())
    part_obj.add_array("acc",           vecxf(part_obj.getWorld().getDim()).field())
    part_obj.add_array("strainRate",    vecxm(part_obj.getWorld().getDim(),part_obj.getWorld().getDim()).field())
    part_obj.add_array("rgb",           vecxf(3).field())
    
    part_obj.add_attr("statistics_linear_momentum",     vecx_f(part_obj.getWorld().getDim()))
    part_obj.add_attr("statistics_angular_momentum",    vecx_f(3))
    part_obj.add_attr("statistics_kinetic_energy",      val_f(0))


    sph = ti.types.struct(
        h                               = ti.f32,
        sig                             = ti.f32,
        sig_inv_h                       = ti.f32,
        density                         = ti.f32,
        compression_ratio               = ti.f32,
        pressure                        = ti.f32,
        pressure_force                  = vecxf(part_obj.getWorld().getDim()),
        viscosity_force                 = vecxf(part_obj.getWorld().getDim()),
        gravity_force                   = vecxf(part_obj.getWorld().getDim()),
    )

    sph_df      = ti.types.struct(
        alpha_1                         = vecxf(part_obj.getWorld().getDim()),
        vel_adv                         = vecxf(part_obj.getWorld().getDim()),
        alpha_2                         = ti.f32,
        alpha                           = ti.f32,
        kappa_incomp                    = ti.f32,
        kappa_div                       = ti.f32,
        delta_density                   = ti.f32,
        delta_compression_ratio         = ti.f32,
    )
    sph_wc      = ti.types.struct(
        B                               = ti.f32,
    )

    phase       = ti.types.struct(
        val_frac                        = ti.f32,
        val_frac_in                     = ti.f32,
        val_frac_out                    = ti.f32,
        vel                             = vecxf(part_obj.getWorld().getDim()),
        drift_vel                       = vecxf(part_obj.getWorld().getDim()),
        acc                             = vecxf(part_obj.getWorld().getDim()),
    )
    mixture     = ti.types.struct(
        lamb                            = ti.f32,
        flag_negative_val_frac          = ti.i32,
        acc_pressure                    = vecxf(part_obj.getWorld().getDim()),
    )


    sph_elastic = ti.types.struct(
        force                           = vecxf(part_obj.getWorld().getDim()),
        F                               = matxf(part_obj.getWorld().getDim()),
        L                               = matxf(part_obj.getWorld().getDim()),
        L_inv                           = matxf(part_obj.getWorld().getDim()),
        R                               = matxf(part_obj.getWorld().getDim()),
        eps                             = matxf(part_obj.getWorld().getDim()),
        P                               = matxf(part_obj.getWorld().getDim()),
        svd_U                           = matxf(part_obj.getWorld().getDim()),
        svd_SIG                         = matxf(part_obj.getWorld().getDim()),
        svd_V_T                         = matxf(part_obj.getWorld().getDim()),
        # link_num                        = ti.i32,
        # link_num_0                      = ti.i32,
        # dissolve                        = ti.f32,
        # flag                            = ti.i32,
        K                               = ti.f32,
        G                               = ti.f32,
    )

    ''' Data Structure for each particle '''
    part_obj.add_struct("sph",          sph)
    part_obj.add_struct("sph_df",       sph_df)
    part_obj.add_struct("sph_wc",       sph_wc)
    part_obj.add_struct("sph_elastic",  sph_elastic)

    ''' Data Structure for each particle '''
    if hasattr(part_obj.getWorld(), 'g_phase_num'):
        part_obj.add_struct("phase",    phase, bundle=part_obj.getWorld().g_phase_num[None])
        part_obj.add_struct("mixture",  mixture)

    if verbose:
        part_obj.verbose_attrs("fluid_part")
        part_obj.verbose_arrays("fluid_part")
        part_obj.verbose_structs("fluid_part")

    return part_obj
