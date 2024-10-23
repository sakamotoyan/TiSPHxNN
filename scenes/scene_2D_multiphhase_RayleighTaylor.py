import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from scenes.scene_import import *

''' TAICHI SETTINGS '''
ti.init(arch=ti.cuda) 
# ti.init(arch=ti.cuda, device_memory_GB=6) 
''' GLOBAL SETTINGS SIMULATION '''
part_size                   = 0.02          # Unit: m
max_time_step               = part_size/50  # Unit: s
sim_time_limit              = 50.0          # Unit: s
kinematic_viscosity_fluid   = 0.01          # Unit: Pa s^-1
gravity_acc                 = -9.8          # Unit: m s^-2
phase_num                   = 3
fps                         = 30
sim_dimension               = 2
Cf                          = 0.0 
Cd                          = 0.0 
dpi                         = 400

''' INIT SIMULATION WORLD '''
world = tsph.World(dim=sim_dimension, lb=-5, rt=5)
world.setPartSize(part_size) 
world.setDt(max_time_step) 
world.set_multiphase(phase_num,[tsph.vec3f(0.2,0.0,0.8),tsph.vec3f(0.8,0,0.2),tsph.vec3f(0,0,1)],[500,1000,1000]) 
world.setGravityMagnitude(gravity_acc)

''' DATA SETTINGS FOR FLUID PARTICLE '''
pool_data = tsph.Squared_pool_2D_data(container_height=9, container_size=5, fluid_height=2.5, span=world.g_part_size[None]*1.0005, layer = 3, fluid_empty_width=0.5)
fluid_part_sphere_data = tsph.Sphere_2D_data(radius=0.6, pos=tsph.vec2f(0.0,0.6*1.1), span=world.g_part_size[None]*1.0005)
fluid_part_num = pool_data.fluid_part_num + fluid_part_sphere_data.fluid_part_num
bound_part_num = pool_data.bound_part_num
fluid_part_pos = pool_data.fluid_part_pos
bound_part_pos = pool_data.bound_part_pos
tsph.DEBUG("total fluid_part_num: "+str(fluid_part_num))

fluid_part = tsph.Particle(part_num=fluid_part_num, part_size=world.g_part_size, is_dynamic=True)
world.attachPartObj(fluid_part)
fluid_part.instantiate_from_template(part_template)
fluid_part.add_array("flag", ti.field(ti.i32))
fluid_part.open_stack(pool_data.fluid_part_num) # open the stack to feed data
fluid_part.fill_open_stack_with_val(fluid_part.flag, 0) # feed the flag data
fluid_part.fill_open_stack_with_val(fluid_part.k_vis, kinematic_viscosity_fluid)
fluid_part.fill_open_stack_with_nparr(fluid_part.pos, pool_data.fluid_part_pos)  # feed the position data
fluid_part.fill_open_stack_with_val(fluid_part.size, fluid_part.getPartSize()) # feed the particle size
fluid_part.fill_open_stack_with_val(fluid_part.volume, fluid_part.getPartSize()**world.g_dim[None]) # feed the particle volume
val_frac = ti.field(dtype=ti.f32, shape=phase_num) # create a field to store the volume fraction
val_frac[0], val_frac[1], val_frac[2] = 1.0,0.0,0.0 # set up the volume fraction
fluid_part.fill_open_stack_with_vals(fluid_part.phase.val_frac, val_frac) # feed the volume fraction
# fluid_part.fill_open_stack_with_val(fluid_part.vel, tsph.vec2f([1.0, 0.0])) # feed the initial velocity
fluid_part.close_stack() # close the stack

# fluid_part.open_stack(fluid_part_sphere_data.fluid_part_num) # open the stack to feed data for another cube
# fluid_part.fill_open_stack_with_val(fluid_part.flag, 1) # feed the flag data
# fluid_part.fill_open_stack_with_nparr(fluid_part.pos, fluid_part_sphere_data.fluid_part_pos) # feed the position data
# fluid_part.fill_open_stack_with_val(fluid_part.size, fluid_part.getPartSize()) # feed the particle size
# fluid_part.fill_open_stack_with_val(fluid_part.volume, fluid_part.getPartSize()**world.g_dim[None]) # feed the particle volume
# val_frac[0], val_frac[1], val_frac[2] = 0.0,1.0,0.0 # set up the volume fraction
# fluid_part.fill_open_stack_with_vals(fluid_part.phase.val_frac, val_frac) # feed the volume fraction
# # fluid_part.fill_open_stack_with_val(fluid_part.vel, tsph.vec2f([-1.0, 0.0])) # feed the initial velocity
# fluid_part.close_stack() # close the stack

bound_part = tsph.Particle(part_num=bound_part_num, part_size=world.g_part_size, is_dynamic=False)
world.attachPartObj(bound_part)
bound_part.instantiate_from_template(part_template)
bound_part.open_stack(bound_part_num)
bound_part.fill_open_stack_with_nparr(bound_part.pos, bound_part_pos)
bound_part.fill_open_stack_with_val(bound_part.size, bound_part.getPartSize())
bound_part.fill_open_stack_with_val(bound_part.volume, bound_part.getPartSize()**world.getDim())
bound_part.fill_open_stack_with_val(bound_part.mass, 1000*bound_part.getPartSize()**world.getDim())
bound_part.fill_open_stack_with_val(bound_part.rest_density, 1000)
bound_part.close_stack()

'''INIT NEIGHBOR SEARCH OBJECTS'''
neighb_list=[fluid_part, bound_part]
fluid_part.add_module_neighb_search([
                                    fluid_part,
                                    bound_part
                                   ])
bound_part.add_module_neighb_search([fluid_part, bound_part])

'''INIT SOLVER OBJECTS'''
# the shared solver
fluid_part.add_solver_adv()
fluid_part.add_solver_sph()
fluid_part.add_solver_elastic(lame_lambda=1e3, lame_mu=1e3)
fluid_part.add_solver_df(div_free_threshold=1e-4, incomp_warm_start=False, div_warm_start=False, incompressible_threshold=1e-4)
fluid_part.add_solver_ism(Cd=Cd, Cf=Cf, k_vis_inter=kinematic_viscosity_fluid, k_vis_inner=kinematic_viscosity_fluid)

bound_part.add_solver_sph()
bound_part.add_solver_df(div_free_threshold=2e-4)

world.init_modules()

gui2d = tsph.Gui2d(objs=[fluid_part, bound_part], radius=world.g_part_size[None]*0.5, lb=tsph.vec2f(-6,-6),rt=tsph.vec2f(6,6),dpi=dpi)
''' DATA PREPERATIONs '''
def prep():
    world.neighb_search() # perform the neighbor search
    fluid_part.m_solver_ism.update_rest_density_and_mass()
    fluid_part.m_solver_ism.update_color() # update the color
    fluid_part.m_solver_ism.recover_phase_vel_from_mixture() # recover the phase velocity from the mixture velocitye elastic force
    fluid_part.m_solver_elastic.init()

''' SIMULATION LOOPS '''
def loop():
    ''' update color based on the volume fraction '''
    
    # fluid_part.m_solver_ism.update_color()

    world.neighb_search()
    world.step_sph_compute_compression_ratio()
    world.step_df_compute_beta()
    # print('beta:', fluid_part.m_neighb_search.neighb_pool.xijNorm.to_numpy()[:1000])
    world.step_vfsph_div(update_vel=True)
    print('div_free iter:', fluid_part.m_solver_df.div_free_iter[None])

    '''  [TIME START] ISM Part 1 '''
    # fluid_part.m_solver_df.get_acc_pressure()
    # fluid_part.m_solver_ism.clear_phase_acc()
    # fluid_part.m_solver_ism.ditribute_acc_pressure_2_phase()
    # fluid_part.m_solver_ism.phase_acc_2_phase_vel()
    # fluid_part.m_solver_ism.update_vel_from_phase_vel()
    '''  [TIME END] ISM Part 1 '''
    
    '''  [TIME START] ISM Part 2 '''
    # fluid_part.m_solver_ism.clear_phase_acc()
    # fluid_part.m_solver_ism.add_phase_acc_gravity()
    # fluid_part.m_solver_ism.loop_neighb(fluid_part.m_neighb_search.neighb_pool, fluid_part, fluid_part.m_solver_ism.inloop_add_phase_acc_vis)
    # fluid_part.m_solver_ism.phase_acc_2_phase_vel() 
    # fluid_part.m_solver_ism.update_vel_from_phase_vel()
    '''  [TIME END] ISM Part 2 '''

    ''' [TIME START] Advection process '''
    world.clear_acc()
    fluid_part.getSolverAdv().add_acc_gravity()
    # fluid_part.get_module_neighbSearch().loop_neighb(fluid_part, fluid_part.getSolverAdv().inloop_accumulate_vis_acc)
    fluid_part.get_module_neighbSearch().loop_neighb(bound_part, fluid_part.getSolverAdv().inloop_accumulate_vis_acc)
    # fluid_part.m_solver_sph.loop_neighb(fluid_part.m_neighb_search.neighb_pool, bound_part, fluid_part.getSolverAdv().inloop_accumulate_vis_acc)
    fluid_part.getSolverElastic().step()
    # fluid_part.getSolverElastic().update_rest()
    world.acc2vel()
    ''' [TIME END] Advection process '''

    world.step_vfsph_incomp(update_vel=True)
    print('incomp iter:', fluid_part.m_solver_df.incompressible_iter[None])

    '''  [TIME START] ISM Part 3 '''
    # fluid_part.m_solver_df.get_acc_pressure()
    # fluid_part.m_solver_ism.clear_phase_acc()
    # fluid_part.m_solver_ism.ditribute_acc_pressure_2_phase()
    # fluid_part.m_solver_ism.phase_acc_2_phase_vel()
    # fluid_part.m_solver_ism.update_vel_from_phase_vel()
    '''  [TIME END] ISM Part 3 '''

    world.update_pos_from_vel()

    world.g_time[None] += world.g_dt[None]
    print('time:', world.g_time)
    # if world.getTime() < 0.5:
    
    '''  [TIME START] ISM Part 4 '''
    # fluid_part.m_solver_ism.update_val_frac()
    # fluid_part.m_solver_ism.update_vel_from_phase_vel()
    # fluid_part.m_solver_ism.regularize_val_frac()
    # fluid_part.m_solver_ism.update_rest_density_and_mass()
    # fluid_part.m_solver_ism.update_vel_from_phase_vel()
    '''  [TIME END] ISM Part 4 '''


    # world.cfl_dt(0.4, max_time_step)

    ''' statistical info '''
    # print(' ')
    # fluid_part.m_solver_ism.statistics_linear_momentum_and_kinetic_energy()
    # fluid_part.m_solver_ism.statistics_angular_momentum()
    # fluid_part.m_solver_ism.debug_check_val_frac()



def write_part_info_ply():
    for part_id in range(fluid_part.getStackTop()):
        fluid_part.pos[part_id]
        fluid_part.vel[part_id]
        for phase_id in range(phase_num):
            fluid_part.phase.val_frac[part_id, phase_id]
        fluid_part.rgb[part_id]

def export_pos_to_numpy(path):
    np.save(path, fluid_part.pos.to_numpy())
def export_flag_to_numpy(path):
    np.save(path, fluid_part.flag.to_numpy())

def export_pos_to_npz(path):
    np.savez_compressed(path, pos=fluid_part.pos.to_numpy())

''' Viusalization and run '''
def vis_run(loop):
    inv_fps = 1/fps
    timer = 0
    sim_time = 0
    loop_count = 0
    flag_write_img = False

    gui = tsph.Gui3d()
    while gui.window.running:

        gui.monitor_listen()

        if gui.op_system_run:
            loop()
            loop_count += 1
            sim_time += world.g_dt[None]
            
            if(sim_time > timer*inv_fps):
                if gui.op_write_file:
                    write_part_info_ply()
                timer += 1
                flag_write_img = True
        
        if gui.op_refresh_window:
            gui.scene_setup()
            gui.scene_add_parts_colorful(obj_pos=fluid_part.pos, obj_color=fluid_part.rgb,index_count=fluid_part.getStackTop(),size=world.g_part_size[None])
            gui.scene_add_parts_colorful(obj_pos=bound_part.pos, obj_color=bound_part.rgb,index_count=bound_part.getStackTop(),size=world.getPartSize())
            gui.canvas.scene(gui.scene)  # Render the scene

            if gui.op_save_img and flag_write_img:
                # gui.window.save_image('output/'+str(timer)+'.png')
                gui2d.save_img(path=os.path.join(parent_dir,'output/part_'+str(timer)+'.jpg'))
                flag_write_img = False

            gui.window.show()
        
        if sim_time > sim_time_limit:
            break

def run(loop):
    inv_fps = 1/fps
    timer = 0
    sim_time = 0
    loop_count = 0

    while sim_time < sim_time_limit:
        loop()
        loop_count += 1
        sim_time += world.g_dt[None]

        if(sim_time > timer*inv_fps):
            # write_part_info_ply()
            timer += 1
            gui2d.save_img(path=os.path.join(parent_dir,'output_vis/part_'+str(timer)+'.jpg'))
            export_pos_to_numpy(os.path.join(parent_dir,'output_data/part_'+str(timer)+'.npy'))
            export_flag_to_numpy(os.path.join(parent_dir,'output_data/flag_'+str(timer)+'.npy'))

''' RUN THE SIMULATION '''
prep()
run(loop)