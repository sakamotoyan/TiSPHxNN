import taichi as ti
from ti_sph import *
from template_part import part_template
from template_grid import grid_template
import time
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

''' TAICHI SETTINGS '''
ti.init(arch=ti.cuda, device_memory_GB=3) # Use GPU, comment this command to run this programme on CPU
# ti.init(arch=ti.cpu) # Use CPU, uncomment this command to run this programme if you don't have GPU

''' GLOBAL SETTINGS '''
fps = 60
output_frame_num = 2000
sense_res = 128
output_shift = 2000

part_size = 0.05
phase_num = 3
max_time_step = part_size/100
Cf = 0.0
Cd = 0.0
kinematic_viscosity_fluid_inter = 0.0
kinematic_viscosity_fluid_inner = 0.0

world = World(dim=2)
world.setWorldPartSize(part_size)
world.setWorldDt(max_time_step)
world.set_multiphase(phase_num,[vec3f(0.8,0.2,0),vec3f(0,0.8,0.2),vec3f(0,0,1)],[500,500,1000])

'''BASIC SETTINGS FOR FLUID'''
fluid_rest_density = val_f(10)
fluid_rest_density_2 = val_f(1000)
val_frac = ti.field(dtype=ti.f32, shape=phase_num)
# phase_vel = ti.Vector.field(world.g_dim[None], dtype=ti.f32, shape=phase_num)
# loop val_frac

fluid_cube_data_1 = Cube_data(type=Cube_data.FIXED_CELL_SIZE, lb=vec2f(-4+part_size, -4+part_size), rt=vec2f(4-part_size*3, -2), span=world.g_part_size[None]*1.001)
fluid_cube_data_2 = Cube_data(type=Cube_data.FIXED_CELL_SIZE, lb=vec2f(0, -1.8), rt=vec2f(3, 3.5), span=world.g_part_size[None]*1.001)
'''INIT AN FLUID PARTICLE OBJECT'''
fluid_part_num = val_i(fluid_cube_data_1.num + fluid_cube_data_2.num)
print("fluid_part_num", fluid_part_num)
fluid_part = world.add_part_obj(part_num=fluid_part_num[None], size=world.g_part_size, is_dynamic=True)
fluid_part.instantiate_from_template(part_template, world)
'''PUSH PARTICLES TO THE OBJECT'''
fluid_part.open_stack(val_i(fluid_cube_data_1.num))
fluid_part.fill_open_stack_with_nparr(fluid_part.pos, fluid_cube_data_1.pos)
fluid_part.fill_open_stack_with_val(fluid_part.size, fluid_part.getObjPartSize())
fluid_part.fill_open_stack_with_val(fluid_part.volume, val_f(fluid_part.getObjPartSize()[None]**world.g_dim[None]))
fluid_part.fill_open_stack_with_val(fluid_part.mass, val_f(fluid_rest_density_2[None]*fluid_part.getObjPartSize()[None]**world.g_dim[None]))
fluid_part.fill_open_stack_with_val(fluid_part.rest_density, fluid_rest_density_2)
fluid_part.fill_open_stack_with_val(fluid_part.rgb, vec3_f([0.0, 0.0, 1.0]))
# val_frac[0], val_frac[1], val_frac[2] = 1.0,0.0,0.0
val_frac[0], val_frac[1], val_frac[2] = 0.5,0.0,0.5
fluid_part.fill_open_stack_with_vals(fluid_part.phase.val_frac, val_frac)
fluid_part.fill_open_stack_with_val(fluid_part.vel, vec2_f([0.0, 1.0]))
fluid_part.close_stack()

fluid_part.open_stack(val_i(fluid_cube_data_2.num))
fluid_part.fill_open_stack_with_nparr(fluid_part.pos, fluid_cube_data_2.pos)
fluid_part.fill_open_stack_with_val(fluid_part.size, fluid_part.getObjPartSize())
fluid_part.fill_open_stack_with_val(fluid_part.volume, val_f(fluid_part.getObjPartSize()[None]**world.g_dim[None]))
fluid_part.fill_open_stack_with_val(fluid_part.mass, val_f(fluid_rest_density[None]*fluid_part.getObjPartSize()[None]**world.g_dim[None]))
fluid_part.fill_open_stack_with_val(fluid_part.rest_density, fluid_rest_density)
fluid_part.fill_open_stack_with_val(fluid_part.rgb, vec3_f([1.0, 0.0, 1.0]))
# val_frac[0], val_frac[1], val_frac[2] = 0.0,0.0,1.0
fluid_part.fill_open_stack_with_vals(fluid_part.phase.val_frac, val_frac)
fluid_part.close_stack()




'''INIT NEIGHBOR SEARCH OBJECTS'''
neighb_list=[fluid_part]

fluid_part.add_module_neighb_search()

fluid_part.add_neighb_objs(neighb_list)

fluid_part.add_solver_adv()
fluid_part.add_solver_sph()
fluid_part.add_solver_df(div_free_threshold=1e-4, incomp_warm_start=False, div_warm_start=False)
fluid_part.add_solver_ism(Cd=Cd, Cf=Cf, k_vis_inter=kinematic_viscosity_fluid_inter, k_vis_inner=kinematic_viscosity_fluid_inner)
fluid_part.add_solver_wcsph()
fluid_part.add_solver_JL21(kd=2,Cf=0.0,k_vis=kinematic_viscosity_fluid_inter)



world.init_modules()

world.neighb_search()

# print('DEBUG sense_output', sense_output.np_node_index_organized)
# save as numpy file
# np.save("pos_np.npy", sense_output.np_node_index_organized)
fluid_part.m_solver_ism.recover_phase_vel_from_mixture()
def loop():
    ''' color '''
    fluid_part.m_solver_ism.update_color()

    ''' neighb search'''
    world.update_pos_in_neighb_search()
    world.neighb_search()

    ''' sph pre-computation '''
    world.step_sph_compute_density()
    world.step_sph_compute_compression_ratio()
    world.step_df_compute_beta()
    # world.step_df_compute_alpha()

    ''' pressure (divergence-free) '''
    world.step_vfsph_div(update_vel=False)
    fluid_part.m_solver_df.get_acc_pressure()
    fluid_part.m_solver_ism.clear_phase_acc()
    fluid_part.m_solver_ism.ditribute_acc_pressure_2_phase()
    fluid_part.m_solver_ism.phase_acc_2_phase_vel()
    fluid_part.m_solver_ism.update_vel_from_phase_vel()
    print('div_free iter:', fluid_part.m_solver_df.div_free_iter[None])

    ''' gravity and viscosity '''
    fluid_part.m_solver_ism.clear_phase_acc()
    # fluid_part.m_solver_ism.add_phase_acc_gravity()
    # fluid_part.m_solver_ism.loop_neighb(fluid_part.m_neighb_search.neighb_pool, fluid_part, fluid_part.m_solver_ism.inloop_add_phase_acc_vis)
    fluid_part.m_solver_ism.phase_acc_2_phase_vel() 
    fluid_part.m_solver_ism.update_vel_from_phase_vel()

    ''' pressure (incompressible) '''
    world.step_vfsph_incomp(update_vel=False)
    fluid_part.m_solver_df.get_acc_pressure()
    fluid_part.m_solver_ism.clear_phase_acc()
    fluid_part.m_solver_ism.ditribute_acc_pressure_2_phase()
    fluid_part.m_solver_ism.phase_acc_2_phase_vel()
    fluid_part.m_solver_ism.update_vel_from_phase_vel()

    ''' pos update '''
    world.update_pos_from_vel()

    # fluid_part.m_solver_ism.zero_out_drift_vel() # DRBUG
    ''' phase change '''
    fluid_part.m_solver_ism.update_val_frac()
    fluid_part.m_solver_ism.update_vel_from_phase_vel()

    ''' update mass and velocity '''
    fluid_part.m_solver_ism.regularize_val_frac()
    fluid_part.m_solver_ism.update_rest_density_and_mass()
    fluid_part.m_solver_ism.update_vel_from_phase_vel()

    

    ''' cfl condition update'''
    world.cfl_dt(0.4, max_time_step)
    # dt = fluid_part.m_solver_ism.cfl_dt(0.5, max_time_step)
    # world.setWorldDt(dt)

    ''' debug info '''
    print('incomp iter:', fluid_part.m_solver_df.incompressible_iter[None])
    print(' ')
    # print('val_frac_sum: \n',fluid_part.phase.val_frac.to_numpy()[0].sum(), fluid_part.phase.val_frac.to_numpy()[1].sum())
    # print('drift vel: \n',fluid_part.phase.drift_vel.to_numpy()[0])
    # frac_np = fluid_part.phase.val_frac.to_numpy()
    # print("phase 1 total", frac_np[:,0].sum())
    # print("phase 2 total", frac_np[:,1].sum())
    # print("phase 3 total", frac_np[:,2].sum())
    # for i in range(fluid_part.getObjStackTop()[None]):
    #     if frac_np[i].sum() < 0.999:
    #         print('low val_frac: \n',frac_np[i])
    # print('dt', world.g_dt[None])   

    # fluid_part.m_solver_ism.draw_drift_vel(1)
    # fluid_part.m_solver_ism.debug_check_negative_phase()
    fluid_part.m_solver_ism.debug_check_val_frac()
    fluid_part.m_solver_ism.statistics_linear_momentum_and_kinetic_energy()

def loop_JL21():
    fluid_part.m_solver_JL21.update_color()

    world.update_pos_in_neighb_search()
    world.neighb_search()
    # world.step_sph_compute_density()
    world.step_sph_compute_number_density()

    # world.clear_acc()
    # world.add_acc_gravity()

    fluid_part.m_solver_JL21.clear_vis_force()
    fluid_part.m_solver_JL21.loop_neighb(fluid_part.m_neighb_search.neighb_pool, fluid_part, fluid_part.m_solver_JL21.inloop_add_force_vis)

    fluid_part.m_solver_JL21.clear_pressure_force()
    world.step_wcsph_add_acc_number_density_pressure()
    fluid_part.m_solver_JL21.loop_neighb(fluid_part.m_neighb_search.neighb_pool, fluid_part, fluid_part.m_solver_JL21.inloop_add_force_pressure)

    fluid_part.m_solver_JL21.update_phase_vel()
    
    world.update_pos_from_vel()

    fluid_part.m_solver_JL21.update_val_frac_lamb()    

    fluid_part.m_solver_JL21.statistics_linear_momentum_and_kinetic_energy()

    dt, max_vec = world.get_cfl_dt_obj(fluid_part, 0.5, max_time_step)
    world.setWorldDt(dt)    


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
            
            # print('loop count', loop_count, 'compressible ratio', 'incompressible iter', fluid_part_1.m_solver_df.incompressible_iter[None], ' ', fluid_part_2.m_solver_df.incompressible_iter[None])
            # print('comp ratio', fluid_part_1.m_solver_df.compressible_ratio[None], ' ', fluid_part_2.m_solver_df.compressible_ratio[None])
            # print('dt', world.g_dt[None])
            if(sim_time > timer*inv_fps):
                if gui.op_write_file:
                    sense_output.export_to_numpy(index=output_shift+timer,path='./output')
                timer += 1
                flag_write_img = True
        
        if gui.op_refresh_window:
            gui.scene_setup()
            if gui.show_bound:
                gui.scene_add_parts_colorful(obj_pos=fluid_part.pos, obj_color=fluid_part.rgb,index_count=fluid_part.getObjStackTop()[None],size=world.g_part_size[None])
                # gui.scene_add_parts(obj_pos=bound_part.pos, obj_color=(0,0.5,1),index_count=bound_part.getObjStackTop()[None],size=world.g_part_size[None])
            else:
                gui.scene_add_parts_colorful(obj_pos=sense_grid_part.pos, obj_color=sense_grid_part.rgb, index_count=sense_grid_part.getObjStackTop()[None], size=sense_grid_part.getObjPartSize()[None]*1.0)
            
            gui.canvas.scene(gui.scene)  # Render the scene

            if gui.op_save_img and flag_write_img:
                gui.window.save_image('output/'+str(timer)+'.png')
                flag_write_img = False

            gui.window.show()
        
        if timer > output_frame_num:
            break

loop_JL21()
vis_run(loop_JL21)







