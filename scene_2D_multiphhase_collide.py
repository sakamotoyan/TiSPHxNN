import taichi as ti
from ti_sph import *
from template_part import part_template
import time
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

''' TAICHI SETTINGS '''
# Use GPU, comment the below command to run this programme on CPU
ti.init(arch=ti.cuda, device_memory_GB=3) 
# Use CPU, uncomment the below command to run this programme if you don't have GPU
# ti.init(arch=ti.cpu) 

''' SOLVER SETTINGS '''
SOLVER_ISM = 0  # proposed method
SOLVER_JL21 = 1 # baseline method
solver = SOLVER_JL21 # choose the solver

''' SETTINGS OUTPUT DATA '''
# output fps
fps = 60
# max output frame number
output_frame_num = 2000

''' SETTINGS SIMULATION '''
# size of the particle
part_size = 0.05 
# number of phases
phase_num = 3 
# max time step size
if solver == SOLVER_ISM:
    max_time_step = part_size/10
elif solver == SOLVER_JL21:
    max_time_step = part_size/200
#  diffusion amount: Cf = 0 yields no diffusion
Cf = 0.0 
#  drift amount (for ism): Cd = 0 yields free driftand Cd = 1 yields no drift
Cd = 0.0 
# drag coefficient (for JL21): kd = 0 yields maximum drift 
kd = 0.0
# kinematic viscosity of fluid
kinematic_viscosity_fluid = 1e-4

''' INIT SIMULATION WORLD '''
# create a 2D world
world = World(dim=2) 
# set the particle diameter
world.set_part_size(part_size) 
# set the max time step size
world.set_dt(max_time_step) 
# set up the multiphase. The first argument is the number of phases. The second argument is the color of each phase (RGB). The third argument is the rest density of each phase.
world.set_multiphase(phase_num,[vec3f(0.8,0.2,0),vec3f(0,0.8,0.2),vec3f(0,0,1)],[500,500,1000]) 


''' DATA SETTINGS FOR FLUID PARTICLE '''
# generate the fluid particle data as two cubes. Should leave tiny space (1.001 of the part size) between particles to avoid SPH density error.
fluid_cube_data_1 = Cube_data(type=Cube_data.FIXED_CELL_SIZE, lb=vec2f(-3, -1.5), rt=vec2f(0, 1.5), span=world.g_part_size[None]*1.001)
fluid_cube_data_2 = Cube_data(type=Cube_data.FIXED_CELL_SIZE, lb=vec2f(0.5, -2.0), rt=vec2f(2.5, 2.0), span=world.g_part_size[None]*1.001)
fluid_part_num = val_i(fluid_cube_data_1.num + fluid_cube_data_2.num)
# get the number of fluid particles required to be generated
print("fluid_part_num", fluid_part_num)

'''INIT AN FLUID PARTICLE OBJECT'''
# create a fluid particle object. first argument is the number of particles. second argument is the size of the particle. third argument is whether the particle is dynamic or not.
fluid_part = world.add_part_obj(part_num=fluid_part_num[None], size=world.g_part_size, is_dynamic=True)
# set the particle data structure to the fluid particle object (see the file template_part.py)
fluid_part.instantiate_from_template(part_template, world)

''' FEED DATA TO THE FLUID PARTICLE OBJECT '''
fluid_part.open_stack(val_i(fluid_cube_data_1.num)) # open the stack to feed data
fluid_part.fill_open_stack_with_nparr(fluid_part.pos, fluid_cube_data_1.pos)  # feed the position data
fluid_part.fill_open_stack_with_val(fluid_part.size, fluid_part.get_part_size()) # feed the particle size
fluid_part.fill_open_stack_with_val(fluid_part.volume, val_f(fluid_part.get_part_size()[None]**world.g_dim[None])) # feed the particle volume
val_frac = ti.field(dtype=ti.f32, shape=phase_num) # create a field to store the volume fraction
val_frac[0], val_frac[1], val_frac[2] = 0.5,0.0,0.5 # set up the volume fraction
fluid_part.fill_open_stack_with_vals(fluid_part.phase.val_frac, val_frac) # feed the volume fraction
fluid_part.fill_open_stack_with_val(fluid_part.vel, vec2_f([1.0, 0.0])) # feed the initial velocity
fluid_part.close_stack() # close the stack
fluid_part.open_stack(val_i(fluid_cube_data_2.num)) # open the stack to feed data for another cube
fluid_part.fill_open_stack_with_nparr(fluid_part.pos, fluid_cube_data_2.pos) # feed the position data
fluid_part.fill_open_stack_with_val(fluid_part.size, fluid_part.get_part_size()) # feed the particle size
fluid_part.fill_open_stack_with_val(fluid_part.volume, val_f(fluid_part.get_part_size()[None]**world.g_dim[None])) # feed the particle volume
fluid_part.fill_open_stack_with_vals(fluid_part.phase.val_frac, val_frac) # feed the volume fraction
fluid_part.close_stack() # close the stack

'''INIT NEIGHBOR SEARCH OBJECTS'''
neighb_list=[fluid_part]
fluid_part.add_module_neighb_search()
fluid_part.add_neighb_objs(neighb_list)

'''INIT SOLVER OBJECTS'''
# the shared solver
fluid_part.add_solver_adv()
fluid_part.add_solver_sph()
# solvers for the proposed method
if solver == SOLVER_ISM:
    fluid_part.add_solver_df(div_free_threshold=1e-4, incomp_warm_start=True, div_warm_start=False)
    fluid_part.add_solver_ism(Cd=Cd, Cf=Cf, k_vis_inter=kinematic_viscosity_fluid, k_vis_inner=kinematic_viscosity_fluid)
# solvers for the baseline method
elif solver == SOLVER_JL21:
    fluid_part.add_solver_wcsph()
    fluid_part.add_solver_JL21(kd=kd,Cf=0.0,k_vis=kinematic_viscosity_fluid)

''' INIT ALL SOLVERS '''
world.init_modules()

''' DATA PREPERATIONs '''
def prep_ism():
    world.neighb_search() # perform the neighbor search
    fluid_part.m_solver_ism.update_rest_density_and_mass()
    fluid_part.m_solver_ism.update_color() # update the color
    fluid_part.m_solver_ism.recover_phase_vel_from_mixture() # recover the phase velocity from the mixture velocity

def prep_JL21():
    world.neighb_search() # perform the neighbor search
    fluid_part.m_solver_JL21.update_rest_density_and_mass()
    fluid_part.m_solver_JL21.update_color() # update the color
    fluid_part.m_solver_JL21.recover_phase_vel_from_mixture() # recover the phase velocity from the mixture velocity

''' SIMULATION LOOPS '''
def loop_ism():
    ''' update color based on the volume fraction '''
    fluid_part.m_solver_ism.update_color()

    ''' neighb search'''
    world.neighb_search()

    ''' dfsph pre-computation '''
    world.step_sph_compute_compression_ratio()
    world.step_df_compute_beta()

    ''' pressure accleration (divergence-free) '''
    world.step_vfsph_div(update_vel=False)
    print('div_free iter:', fluid_part.m_solver_df.div_free_iter[None])

    ''' [ISM] distribute pressure acc to phase acc and update phase vel '''
    fluid_part.m_solver_df.get_acc_pressure()
    fluid_part.m_solver_ism.clear_phase_acc()
    fluid_part.m_solver_ism.ditribute_acc_pressure_2_phase()
    fluid_part.m_solver_ism.phase_acc_2_phase_vel()
    fluid_part.m_solver_ism.update_vel_from_phase_vel()
    
    ''' viscosity & gravity (not requird in this scene)  accleration and update phase vel '''
    fluid_part.m_solver_ism.clear_phase_acc()
    # fluid_part.m_solver_ism.add_phase_acc_gravity()
    fluid_part.m_solver_ism.loop_neighb(fluid_part.m_neighb_search.neighb_pool, fluid_part, fluid_part.m_solver_ism.inloop_add_phase_acc_vis)
    fluid_part.m_solver_ism.phase_acc_2_phase_vel() 
    fluid_part.m_solver_ism.update_vel_from_phase_vel()

    ''' pressure accleration (divergence-free) '''
    world.step_vfsph_incomp(update_vel=False)
    print('incomp iter:', fluid_part.m_solver_df.incompressible_iter[None])

    ''' distribute pressure acc to phase acc and update phase vel '''
    fluid_part.m_solver_df.get_acc_pressure()
    fluid_part.m_solver_ism.clear_phase_acc()
    fluid_part.m_solver_ism.ditribute_acc_pressure_2_phase()
    fluid_part.m_solver_ism.phase_acc_2_phase_vel()
    fluid_part.m_solver_ism.update_vel_from_phase_vel()

    ''' update particle position from velocity '''
    world.update_pos_from_vel()

    ''' phase change '''
    fluid_part.m_solver_ism.update_val_frac()
    fluid_part.m_solver_ism.update_vel_from_phase_vel()

    ''' update mass and velocity '''
    fluid_part.m_solver_ism.regularize_val_frac()
    fluid_part.m_solver_ism.update_rest_density_and_mass()
    fluid_part.m_solver_ism.update_vel_from_phase_vel()

    ''' cfl condition update'''
    world.cfl_dt(0.4, max_time_step)

    ''' statistical info '''
    print(' ')
    fluid_part.m_solver_ism.statistics_linear_momentum_and_kinetic_energy()
    fluid_part.m_solver_ism.statistics_angular_momentum()
    fluid_part.m_solver_ism.debug_check_val_frac()

def loop_JL21():
    ''' update color based on the volume fraction '''
    fluid_part.m_solver_JL21.update_color()

    ''' compute number density '''
    world.neighb_search()
    world.step_sph_compute_number_density()

    ''' viscosity & gravity (not requird in this scene)  accleration and update phase vel '''
    world.clear_acc()
    # world.add_acc_gravity()
    fluid_part.m_solver_JL21.clear_vis_force()
    fluid_part.m_solver_JL21.loop_neighb(fluid_part.m_neighb_search.neighb_pool, fluid_part, fluid_part.m_solver_JL21.inloop_add_force_vis)

    ''' pressure force '''
    fluid_part.m_solver_JL21.clear_pressure_force()
    world.step_wcsph_add_acc_number_density_pressure()
    fluid_part.m_solver_JL21.loop_neighb(fluid_part.m_neighb_search.neighb_pool, fluid_part, fluid_part.m_solver_JL21.inloop_add_force_pressure)

    ''' update phase vel (from all accelerations) '''
    fluid_part.m_solver_JL21.update_phase_vel()
    
    ''' update particle position from velocity '''
    world.update_pos_from_vel()

    ''' phase change (spacial care with lambda scheme) '''
    fluid_part.m_solver_JL21.update_val_frac_lamb()    

    ''' statistical info '''
    print(' ')
    fluid_part.m_solver_JL21.statistics_linear_momentum_and_kinetic_energy()
    fluid_part.m_solver_JL21.statistics_angular_momentum()
    fluid_part.m_solver_JL21.debug_check_val_frac()

    dt, max_vec = world.get_cfl_dt_obj(fluid_part, 0.5, max_time_step)
    world.set_dt(dt)    

''' Viusalization and run '''
def vis_run(loop):
    inv_fps = 1/fps
    timer = 0
    sim_time = 0
    loop_count = 0
    flag_write_img = False

    gui = Gui3d()
    while gui.window.running:

        gui.monitor_listen()

        if gui.op_system_run:
            loop()
            loop_count += 1
            sim_time += world.g_dt[None]
            
            if(sim_time > timer*inv_fps):
                if gui.op_write_file:
                    pass
                timer += 1
                flag_write_img = True
        
        if gui.op_refresh_window:
            gui.scene_setup()
            gui.scene_add_parts_colorful(obj_pos=fluid_part.pos, obj_color=fluid_part.rgb,index_count=fluid_part.get_stack_top()[None],size=world.g_part_size[None])
            gui.canvas.scene(gui.scene)  # Render the scene

            if gui.op_save_img and flag_write_img:
                gui.window.save_image('output/'+str(timer)+'.png')
                flag_write_img = False

            gui.window.show()
        
        if timer > output_frame_num:
            break

''' RUN THE SIMULATION '''
if solver == SOLVER_ISM:
    prep_ism()
    vis_run(loop_ism)
elif solver == SOLVER_JL21:
    prep_JL21()
    vis_run(loop_JL21)







