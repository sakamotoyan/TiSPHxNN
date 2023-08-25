import taichi as ti
from ti_sph import *
from template_part import part_template
from template_grid import grid_template
import time
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

''' TAICHI SETTINGS '''
# ti.init(arch=ti.cuda, kernel_profiler=True) 
ti.init(arch=ti.cuda, device_memory_GB=3) # Use GPU
# ti.init(arch=ti.cpu) # Use CPU

''' GLOBAL SETTINGS '''
fps = 60
output_frame_num = 2000
sense_res = 128
output_shift = 2000

part_size = 0.05
max_time_step = part_size/100*0.4
world = World(dim=2)
world.set_part_size(part_size)
world.set_dt(max_time_step)

kinematic_viscosity_air = val_f(1.57e-5)
# kinematic_viscosity_air = val_f(0.01)
kinematic_viscosity_bound = val_f(0.01)

'''BASIC SETTINGS FOR FLUID'''
fluid_rest_density = val_f(1000)
fluid_rest_density_2 = val_f(100)
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
fluid_part.fill_open_stack_with_val(fluid_part.size, fluid_part.get_part_size())
fluid_part.fill_open_stack_with_val(fluid_part.volume, val_f(fluid_part.get_part_size()[None]**world.g_dim[None]))
fluid_part.fill_open_stack_with_val(fluid_part.mass, val_f(fluid_rest_density[None]*fluid_part.get_part_size()[None]**world.g_dim[None]))
fluid_part.fill_open_stack_with_val(fluid_part.rest_density, fluid_rest_density)
fluid_part.fill_open_stack_with_val(fluid_part.rgb, vec3_f([0.0, 0.0, 1.0]))
fluid_part.fill_open_stack_with_val(fluid_part.k_vis, kinematic_viscosity_air)
fluid_part.close_stack()

fluid_part.open_stack(val_i(fluid_cube_data_2.num))
fluid_part.fill_open_stack_with_nparr(fluid_part.pos, fluid_cube_data_2.pos)
fluid_part.fill_open_stack_with_val(fluid_part.size, fluid_part.get_part_size())
fluid_part.fill_open_stack_with_val(fluid_part.volume, val_f(fluid_part.get_part_size()[None]**world.g_dim[None]))
fluid_part.fill_open_stack_with_val(fluid_part.mass, val_f(fluid_rest_density_2[None]*fluid_part.get_part_size()[None]**world.g_dim[None]))
fluid_part.fill_open_stack_with_val(fluid_part.rest_density, fluid_rest_density_2)
fluid_part.fill_open_stack_with_val(fluid_part.rgb, vec3_f([1.0, 0.0, 1.0]))
fluid_part.fill_open_stack_with_val(fluid_part.k_vis, kinematic_viscosity_air)
fluid_part.close_stack()


''' INIT BOUNDARY PARTICLE OBJECT '''
box_data = Box_data(lb=vec2f(-4, -4), rt=vec2f(4, 4), span=world.g_part_size[None]*1.05, layers=3)
bound_rest_density = val_f(1000)
bound_part = world.add_part_obj(part_num=box_data.num, size=world.g_part_size, is_dynamic=False)
bound_part.instantiate_from_template(part_template, world)
bound_part.open_stack(val_i(box_data.num))
bound_part.fill_open_stack_with_arr(bound_part.pos, box_data.pos)
bound_part.fill_open_stack_with_val(bound_part.size, bound_part.get_part_size())
bound_part.fill_open_stack_with_val(bound_part.volume, val_f(bound_part.get_part_size()[None]**world.g_dim[None]))
bound_part.fill_open_stack_with_val(bound_part.mass, val_f(bound_rest_density[None]*bound_part.get_part_size()[None]**world.g_dim[None]))
bound_part.fill_open_stack_with_val(bound_part.rest_density, bound_rest_density)
bound_part.fill_open_stack_with_val(bound_part.k_vis, kinematic_viscosity_bound)
bound_part.close_stack()

# sense_cell_size = val_f(0.1/sense_res*64)
# sense_cube_data = Cube_data(type=Cube_data.FIXED_GRID_RES, span=sense_cell_size[None], grid_res=vec2i(sense_res,sense_res),grid_center=vec2f(0,0))
# sense_grid_part = world.add_part_obj(part_num=sense_cube_data.num, size=sense_cell_size, is_dynamic=False)
# sense_grid_part.instantiate_from_template(grid_template)
# sense_grid_part.open_stack(val_i(sense_cube_data.num))
# sense_grid_part.fill_open_stack_with_nparr(sense_grid_part.pos, sense_cube_data.pos)
# sense_grid_part.fill_open_stack_with_nparr(sense_grid_part.node_index, sense_cube_data.index)
# sense_grid_part.fill_open_stack_with_val(sense_grid_part.size, sense_grid_part.get_part_size())
# sense_grid_part.fill_open_stack_with_val(sense_grid_part.volume, val_f(sense_grid_part.get_part_size()[None]**world.g_dim[None]))
# sense_grid_part.close_stack()




'''INIT NEIGHBOR SEARCH OBJECTS'''
neighb_list=[fluid_part, bound_part]


fluid_part.add_module_neighb_search()
bound_part.add_module_neighb_search()
# sense_grid_part.add_module_neighb_search(max_neighb_num=val_i(fluid_part.get_part_num()[None]*32))

fluid_part.add_neighb_objs(neighb_list)
bound_part.add_neighb_objs(neighb_list)
# sense_grid_part.add_neighb_obj(neighb_obj=fluid_part, search_range=val_f(sense_cell_size[None]*2))


fluid_part.add_solver_adv()
fluid_part.add_solver_sph()
fluid_part.add_solver_wcsph()
# fluid_part.add_solver_df(div_free_threshold=2e-4)

bound_part.add_solver_sph()
bound_part.add_solver_wcsph()
# bound_part.add_solver_df(div_free_threshold=2e-4)

# sense_grid_part.add_solver_sph()

world.init_modules()

world.neighb_search()


# sense_output = Output_manager(format_type = Output_manager.type.SEQ, data_source = sense_grid_part)
# sense_output.add_output_dataType("pos",2)
# sense_output.add_output_dataType("node_index",2)
# sense_output.add_output_dataType("sensed_density",1)
# sense_output.add_output_dataType("vel",2)

# print('DEBUG sense_output', sense_output.np_node_index_organized)
# save as numpy file
# np.save("pos_np.npy", sense_output.np_node_index_organized)

def loop():
    world.update_pos_in_neighb_search()

    world.neighb_search()
    # world.step_sph_compute_density()
    world.step_sph_compute_number_density()

    world.clear_acc()
    world.add_acc_gravity()
    fluid_part.m_solver_sph.loop_neighb(fluid_part.m_neighb_search.neighb_pool, fluid_part, fluid_part.m_solver_adv.inloop_accumulate_vis_acc)
    fluid_part.m_solver_sph.loop_neighb(fluid_part.m_neighb_search.neighb_pool, bound_part, fluid_part.m_solver_adv.inloop_accumulate_vis_acc)
    
    # world.step_wcsph_add_acc_pressure()
    world.step_wcsph_add_acc_number_density_pressure()

    world.acc2vel()
    
    world.update_pos_from_vel()

    dt, max_vec = world.get_cfl_dt_obj(fluid_part, 0.5, max_time_step)
    world.set_dt(dt)



def run(loop):
    inv_fps = 1/fps
    timer = 0
    sim_time = 0
    loop_count = 0

    while True:
        loop()
        loop_count += 1
        print(f'loop {loop_count}')
        sim_time += world.g_dt[None]
        if(sim_time > timer*inv_fps):
            sense_output.export_to_numpy(index=output_shift+timer,path='./output')
            timer += 1
        if timer > output_frame_num:
            break

def vis_run(loop):
    inv_fps = 1/fps
    timer = 0
    sim_time = 0
    loop_count = 0
    gui = Gui3d()
    while gui.window.running:
        gui.monitor_listen()

        if gui.op_system_run:
            loop()
            print(f'loop {loop_count}')
            loop_count += 1
            sim_time += world.g_dt[None]
            # print('loop count', loop_count, 'compressible ratio', 'incompressible iter', fluid_part_1.m_solver_df.incompressible_iter[None], ' ', fluid_part_2.m_solver_df.incompressible_iter[None])
            # print('comp ratio', fluid_part_1.m_solver_df.compressible_ratio[None], ' ', fluid_part_2.m_solver_df.compressible_ratio[None])
            # print('dt', world.g_dt[None])
            if(sim_time > timer*inv_fps):
                if gui.op_write_file:
                    sense_output.export_to_numpy(index=output_shift+timer,path='./output')
                timer += 1
        
        if gui.op_refresh_window:
            gui.scene_setup()
            if gui.show_bound:
                gui.scene_add_parts_colorful(obj_pos=fluid_part.pos, obj_color=fluid_part.rgb,index_count=fluid_part.get_stack_top()[None],size=world.g_part_size[None])
                gui.scene_add_parts(obj_pos=bound_part.pos, obj_color=(0,0.5,1),index_count=bound_part.get_stack_top()[None],size=world.g_part_size[None])
            # else:
                # gui.scene_add_parts_colorful(obj_pos=sense_grid_part.pos, obj_color=sense_grid_part.rgb, index_count=sense_grid_part.get_stack_top()[None], size=sense_grid_part.get_part_size()[None]*1.0)
            
            gui.canvas.scene(gui.scene)  # Render the scene

            if(sim_time > (timer-1)*inv_fps):
                if gui.op_save_img:
                    gui.window.save_image('output/'+str(timer)+'.png')

            gui.window.show()
        
        if timer > output_frame_num:
            break

loop()
vis_run(loop)







