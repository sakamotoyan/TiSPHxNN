import taichi as ti
from ti_sph import *
from template_part import part_template
import time
import os
import sys
import numpy as np
from Timing import Timing
from Statistics import Statistics
import argparse
from ply_util import *
np.set_printoptions(threshold=sys.maxsize)

''' TAICHI SETTINGS '''
# Use GPU, comment the below command to run this programme on CPU
ti.init(arch=ti.cuda, device_memory_GB=3) 
# Use CPU, uncomment the below command to run this programme if you don't have GPU
# ti.init(arch=ti.cpu) 

''' COMMAND ARGUMENTS '''
parser = argparse.ArgumentParser()
parser.add_argument('--solver', help='Solver to use: ISM or JL21')
parser.add_argument('--gui', help='Enable gui')
parser.add_argument('--ply', help='Enable ply')
parser.add_argument('--drag', help='Drag coefficient')
parser.add_argument('--drag_frame', help='Frame to change drag coefficient from default value to drag')
args = parser.parse_args()
print(args)

''' SOLVER SETTINGS '''
SOLVER_ISM = 0  # proposed method
SOLVER_JL21 = 1 # baseline method
if args.solver:
    if args.solver == "ISM":
        solver = SOLVER_ISM
    elif args.solver == "JL21":
        solver = SOLVER_JL21
    else:
        raise ValueError("wrong solver type: " + args.solver)
else:
    solver = SOLVER_ISM # choose the solver

''' FILE SETTINGS '''
file_identifier = ""
if args.solver:
    file_identifier += "_" + args.solver
if args.drag:
    file_identifier += "_" + args.drag
output_folder = "output"+file_identifier
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

''' SETTINGS OUTPUT DATA '''
# use gui
use_gui = False if args.gui and args.gui == "False" else True
# enable ply output
enable_ply = True if args.ply and args.ply == "True" else False
# output fps
fps = 60
# max output frame number
output_frame_num = 110

''' STATISITCS SETTINGS '''
timing = Timing("timing"+file_identifier+".csv")
if solver == SOLVER_ISM:
    timing.addGroup("SPH_NeighbSearch")
    timing.addGroup("SPH_Prepare")
    timing.addGroup("SPH_UpdatePos")
    timing.addGroup("SPH_CFL")
    timing.addGroup("SPH_Pressure_Com")
    timing.addGroup("SPH_Pressure_Div")
    timing.addGroup("ISM_Color")
    timing.addGroup("ISM_Gravity_Vis")
    timing.addGroup("ISM_Pressure_Com")
    timing.addGroup("ISM_Pressure_Div")
    timing.addGroup("ISM_PhaseChange")
    timing.addGroup("ISM_UpdateMassVel")
    timing.addGroup("Statistics")
elif solver == SOLVER_JL21:
    timing.addGroup("SPH_NeighbSearch")
    timing.addGroup("SPH_NumberDensity")
    timing.addGroup("SPH_Gravity")
    timing.addGroup("SPH_Viscosity")
    timing.addGroup("SPH_Pressure")
    timing.addGroup("SPH_UpdatePos")
    timing.addGroup("SPH_CFL")
    timing.addGroup("JL_Color")
    timing.addGroup("JL_UpdatePhaseVel")
    timing.addGroup("JL_PhaseChange")
    timing.addGroup("Statistics")

statistics = Statistics("statistics"+file_identifier+".csv")
statistics.addGroup("momentum_X")
statistics.addGroup("momentum_Y")
statistics.addGroup("momentum_Z")
statistics.addGroup("momentum_Len")
statistics.addGroup("kinetic_energy")
statistics.addGroup("ang_momentum_X")
statistics.addGroup("ang_momentum_Y")
statistics.addGroup("ang_momentum_Z")
statistics.addGroup("ang_momentum_Len")

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
Cd_fin = float(args.drag) if args.drag else 0.0
Cd = 1.0
# drag coefficient (for JL21): kd = 0 yields maximum drift 
kd = float(args.drag) if args.drag else 0.0
flag_start_drift = False
frame_change_drag = int(args.drag_frame) if args.drag_frame else 0
# kinematic viscosity of fluid
kinematic_viscosity_fluid = 0.0 

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
    timing.startStep()

    ''' color '''
    timing.startGroup("ISM_Color")
    fluid_part.m_solver_ism.update_color()
    timing.endGroup()

    ''' neighb search'''
    timing.startGroup("SPH_NeighbSearch")
    world.neighb_search()
    timing.endGroup()

    ''' sph pre-computation '''
    timing.startGroup("SPH_Prepare")
    world.step_sph_compute_compression_ratio()
    world.step_df_compute_beta()
    timing.endGroup()

    ''' pressure accleration (divergence-free) '''
    timing.startGroup("SPH_Pressure_Div")
    world.step_vfsph_div(update_vel=False)
    timing.endGroup()
    print('div_free iter:', fluid_part.m_solver_df.div_free_iter[None])

    ''' [ISM] distribute pressure acc to phase acc and update phase vel '''
    timing.startGroup("ISM_Pressure_Div")
    fluid_part.m_solver_df.get_acc_pressure()
    fluid_part.m_solver_ism.clear_phase_acc()
    fluid_part.m_solver_ism.ditribute_acc_pressure_2_phase()
    fluid_part.m_solver_ism.phase_acc_2_phase_vel()
    fluid_part.m_solver_ism.update_vel_from_phase_vel()
    timing.endGroup()
    

    ''' viscosity & gravity (not requird in this scene)  accleration and update phase vel '''
    timing.startGroup("ISM_Gravity_Vis")
    fluid_part.m_solver_ism.clear_phase_acc()
    # fluid_part.m_solver_ism.add_phase_acc_gravity()
    fluid_part.m_solver_ism.loop_neighb(fluid_part.m_neighb_search.neighb_pool, fluid_part, fluid_part.m_solver_ism.inloop_add_phase_acc_vis)
    fluid_part.m_solver_ism.phase_acc_2_phase_vel() 
    fluid_part.m_solver_ism.update_vel_from_phase_vel()
    timing.endGroup()

    ''' pressure accleration (incompressible) '''
    timing.startGroup("SPH_Pressure_Com")
    world.step_vfsph_incomp(update_vel=False)
    timing.endGroup()
    print('incomp iter:', fluid_part.m_solver_df.incompressible_iter[None])

    ''' distribute pressure acc to phase acc and update phase vel '''
    timing.startGroup("ISM_Pressure_Com")
    fluid_part.m_solver_df.get_acc_pressure()
    fluid_part.m_solver_ism.clear_phase_acc()
    fluid_part.m_solver_ism.ditribute_acc_pressure_2_phase()
    fluid_part.m_solver_ism.phase_acc_2_phase_vel()
    fluid_part.m_solver_ism.update_vel_from_phase_vel()
    timing.endGroup()

    ''' update particle position from velocity '''
    timing.startGroup("SPH_UpdatePos")
    world.update_pos_from_vel()
    timing.endGroup()

    ''' phase change '''
    timing.startGroup("ISM_PhaseChange")
    fluid_part.m_solver_ism.update_val_frac()
    fluid_part.m_solver_ism.update_vel_from_phase_vel()
    timing.endGroup()

    ''' update mass and velocity '''
    timing.startGroup("ISM_UpdateMassVel")
    fluid_part.m_solver_ism.regularize_val_frac()
    fluid_part.m_solver_ism.update_rest_density_and_mass()
    fluid_part.m_solver_ism.update_vel_from_phase_vel()
    timing.endGroup()

    timing.setStepLength(world.g_dt[None])

    ''' cfl condition update'''
    timing.startGroup("SPH_CFL")
    world.cfl_dt(0.4, max_time_step)
    timing.endGroup()

    ''' statistical info '''
    print(' ')
    timing.startGroup("Statistics")
    fluid_part.m_solver_ism.statistics_linear_momentum_and_kinetic_energy()
    fluid_part.m_solver_ism.statistics_angular_momentum()
    fluid_part.m_solver_ism.debug_check_val_frac()
    timing.endGroup()

    timing.endStep()

def loop_JL21():
    timing.startStep()

    ''' update color based on the volume fraction '''
    timing.startGroup("JL_Color")
    fluid_part.m_solver_JL21.update_color()
    timing.endGroup()

    ''' compute number density '''
    timing.startGroup("SPH_NeighbSearch")
    world.neighb_search()
    timing.endGroup()
    timing.startGroup("SPH_NumberDensity")
    world.step_sph_compute_number_density()
    timing.endGroup()

    ''' gravity (not requird in this scene) accleration '''
    timing.startGroup("SPH_Gravity")
    world.clear_acc()
    # world.add_acc_gravity()
    timing.endGroup()

    ''' viscosity force '''
    timing.startGroup("SPH_Viscosity")
    fluid_part.m_solver_JL21.clear_vis_force()
    fluid_part.m_solver_JL21.loop_neighb(fluid_part.m_neighb_search.neighb_pool, fluid_part, fluid_part.m_solver_JL21.inloop_add_force_vis)
    timing.endGroup()

    ''' pressure force '''
    timing.startGroup("SPH_Pressure")
    fluid_part.m_solver_JL21.clear_pressure_force()
    world.step_wcsph_add_acc_number_density_pressure()
    fluid_part.m_solver_JL21.loop_neighb(fluid_part.m_neighb_search.neighb_pool, fluid_part, fluid_part.m_solver_JL21.inloop_add_force_pressure)
    timing.endGroup()

    ''' update phase vel (from all accelerations) '''
    if flag_start_drift:
        timing.startGroup("JL_UpdatePhaseVel")
        fluid_part.m_solver_JL21.update_phase_vel()
        timing.endGroup()
    else:
        timing.startGroup("JL_UpdatePhaseVel")
        fluid_part.m_solver_JL21.vis_force_2_acc()
        fluid_part.m_solver_JL21.pressure_force_2_acc()
        fluid_part.m_solver_JL21.acc_2_vel()
        fluid_part.m_solver_JL21.vel_2_phase_vel()
        timing.endGroup()

    ''' update particle position from velocity '''
    timing.startGroup("SPH_UpdatePos")
    world.update_pos_from_vel()
    timing.endGroup()

    ''' phase change (spacial care with lambda scheme) '''
    timing.startGroup("JL_PhaseChange")
    fluid_part.m_solver_JL21.update_val_frac_lamb()    
    timing.endGroup()

    ''' statistical info '''
    print(' ')
    timing.startGroup("Statistics")
    fluid_part.m_solver_JL21.statistics_linear_momentum_and_kinetic_energy()
    fluid_part.m_solver_JL21.statistics_angular_momentum()
    fluid_part.m_solver_JL21.debug_check_val_frac()
    timing.endGroup()

    timing.setStepLength(world.g_dt[None])

    timing.startGroup("SPH_CFL")
    dt, max_vec = world.get_cfl_dt_obj(fluid_part, 0.5, max_time_step)
    world.set_dt(dt)    
    timing.endGroup() 

    timing.endStep()

def stats_record(sim_time):
    mom_x = fluid_part.statistics_linear_momentum[None].x
    mom_y = fluid_part.statistics_linear_momentum[None].y
    mom_z = 0.0
    k_energy = fluid_part.statistics_kinetic_energy[None]
    a_mom_x = fluid_part.statistics_angular_momentum[None].x
    a_mom_y = fluid_part.statistics_angular_momentum[None].y
    a_mom_z = fluid_part.statistics_angular_momentum[None].z
    statistics.recordStep(sim_time, 
                          momentum_X=mom_x,
                          momentum_Y=mom_y,
                          momentum_Z=mom_z,
                          momentum_Len=(mom_x * mom_x + mom_y * mom_y + mom_z * mom_z) ** 0.5,
                          kinetic_energy=k_energy,
                          ang_momentum_X=a_mom_x,
                          ang_momentum_Y=a_mom_y,
                          ang_momentum_Z=a_mom_z,
                          ang_momentum_Len=(a_mom_x * a_mom_x + a_mom_y * a_mom_y + a_mom_z * a_mom_z) ** 0.5)


''' Viusalization and run '''
def vis_run(loop):
    global flag_start_drift
    inv_fps = 1/fps
    timer = 0
    sim_time = 0
    loop_count = 0
    flag_write_img = False

    gui = Gui3d() if use_gui else None
    while gui is None or gui.window.running:

        if not gui is None:
            gui.monitor_listen()

        if gui is None or gui.op_system_run:
            loop()
            loop_count += 1
            sim_time += world.g_dt[None]
            
            if(sim_time > timer*inv_fps):
                if not gui is None and gui.op_write_file:
                    pass

                if timer == frame_change_drag:
                    if solver == SOLVER_ISM:
                        fluid_part.m_solver_ism.set_Cd(Cd_fin)
                        print("Change Cd to", fluid_part.m_solver_ism.Cd[None])
                    elif solver == SOLVER_JL21:
                        flag_start_drift = True
                        print("Start JL21 drift")
                
                timer += 1
                flag_write_img = True

                stats_record(sim_time)

                # if enable_ply:
                #     write_ply(path='output_ply', 
                #             frame_num=timer, 
                #             dim=world.g_dim[None], 
                #             num=fluid_part.m_part_num[None], 
                #             pos=fluid_part.pos.to_numpy(), 
                #             phase_num=phase_num, 
                #             volume_frac=fluid_part.phase.val_frac.to_numpy(), 
                #             phase_vel_flag=True, 
                #             phase_vel=fluid_part.phase.vel.to_numpy())
        
        if not gui is None and gui.op_refresh_window:
            gui.scene_setup()
            gui.scene_add_parts_colorful(obj_pos=fluid_part.pos, obj_color=fluid_part.rgb,index_count=fluid_part.get_stack_top()[None],size=world.g_part_size[None])
            gui.canvas.scene(gui.scene)  # Render the scene

            if gui.op_save_img and flag_write_img:
                gui.window.save_image(output_folder+'/'+str(timer)+'.png')
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







