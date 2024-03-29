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
output_frame_num = 10
sense_res = 128
output_shift = 10

part_size = 0.05
max_time_step = part_size/100
k_vis = 1e-4
world = World(dim=2)
world.setPartSize(part_size)
world.setDt(max_time_step)

'''BASIC SETTINGS FOR FLUID'''
fluid_rest_density = 1000
fluid_cube_data_1 = Cube_data(type=Cube_data.FIXED_CELL_SIZE, lb=vec2f(-4+part_size, -4+part_size), rt=vec2f(4-part_size*3, -2), span=world.g_part_size[None]*1.001)
fluid_cube_data_2 = Cube_data(type=Cube_data.FIXED_CELL_SIZE, lb=vec2f(0, -1.8), rt=vec2f(3, 3.5), span=world.g_part_size[None]*1.001)
'''INIT AN FLUID PARTICLE OBJECT'''
fluid_part_num = fluid_cube_data_1.num + fluid_cube_data_2.num
print("fluid_part_num", fluid_part_num)
fluid_part = Particle(part_num=fluid_part_num, part_size=world.g_part_size, is_dynamic=True)
world.attachPartObj(fluid_part)
fluid_part.instantiate_from_template(part_template, world)
'''PUSH PARTICLES TO THE OBJECT'''
fluid_part.open_stack(fluid_cube_data_1.num)
fluid_part.fill_open_stack_with_nparr(fluid_part.pos, fluid_cube_data_1.pos)
fluid_part.fill_open_stack_with_val(fluid_part.size, fluid_part.getPartSize())
fluid_part.fill_open_stack_with_val(fluid_part.volume, fluid_part.getPartSize()**world.getDim())
fluid_part.fill_open_stack_with_val(fluid_part.mass, fluid_rest_density*fluid_part.getPartSize()**world.getDim())
fluid_part.fill_open_stack_with_val(fluid_part.rest_density, fluid_rest_density)
fluid_part.fill_open_stack_with_val(fluid_part.rgb, vec3f([0.0, 0.0, 1.0]))
fluid_part.fill_open_stack_with_val(fluid_part.k_vis, k_vis)
fluid_part.close_stack()

fluid_part.open_stack(fluid_cube_data_2.num)
fluid_part.fill_open_stack_with_nparr(fluid_part.pos, fluid_cube_data_2.pos)
fluid_part.fill_open_stack_with_val(fluid_part.size, fluid_part.getPartSize())
fluid_part.fill_open_stack_with_val(fluid_part.volume, fluid_part.getPartSize()**world.getDim())
fluid_part.fill_open_stack_with_val(fluid_part.mass, fluid_rest_density*fluid_part.getPartSize()**world.getDim())
fluid_part.fill_open_stack_with_val(fluid_part.rest_density, fluid_rest_density)
fluid_part.fill_open_stack_with_val(fluid_part.rgb, vec3f([1.0, 0.0, 0.0]))
fluid_part.fill_open_stack_with_val(fluid_part.k_vis, k_vis)
fluid_part.close_stack()


''' INIT BOUNDARY PARTICLE OBJECT '''
box_data = Box_data(lb=vec2f(-4, -4), rt=vec2f(4, 4), span=world.g_part_size[None]*1.05, layers=3)
bound_rest_density = 1000
bound_part = Particle(part_num=box_data.num, part_size=world.g_part_size, is_dynamic=False)
world.attachPartObj(bound_part)
bound_part.instantiate_from_template(part_template, world)
bound_part.open_stack(box_data.num)
bound_part.fill_open_stack_with_arr(bound_part.pos, box_data.pos)
bound_part.fill_open_stack_with_val(bound_part.size, bound_part.getPartSize())
bound_part.fill_open_stack_with_val(bound_part.volume, bound_part.getPartSize()**world.getDim())
bound_part.fill_open_stack_with_val(bound_part.mass, bound_rest_density*bound_part.getPartSize()**world.getDim())
bound_part.fill_open_stack_with_val(bound_part.rest_density, bound_rest_density)
fluid_part.fill_open_stack_with_val(bound_part.rgb, vec3f([0.0, 0.0, 0.0]))
bound_part.fill_open_stack_with_val(bound_part.k_vis, k_vis)
bound_part.close_stack()

sense_cell_size = val_f(7.0/sense_res)
sense_cube_data = Cube_data(type=Cube_data.FIXED_GRID_RES, span=sense_cell_size[None], grid_res=vec2i(sense_res,sense_res),grid_center=vec2f(0,0))
sense_grid_part = Particle(part_num=sense_cube_data.num, part_size=sense_cell_size, is_dynamic=False)
world.attachPartObj(sense_grid_part)
sense_grid_part.instantiate_from_template(grid_template, world)
sense_grid_part.open_stack(sense_cube_data.num)
sense_grid_part.fill_open_stack_with_nparr(sense_grid_part.pos, sense_cube_data.pos)
sense_grid_part.fill_open_stack_with_nparr(sense_grid_part.node_index, sense_cube_data.index)
sense_grid_part.fill_open_stack_with_val(sense_grid_part.size, sense_grid_part.getPartSize())
sense_grid_part.fill_open_stack_with_val(sense_grid_part.volume, sense_grid_part.getPartSize()**world.getDim())
sense_grid_part.close_stack()


'''INIT NEIGHBOR SEARCH OBJECTS'''
neighb_list=[fluid_part, bound_part]


fluid_part.add_module_neighb_search()
bound_part.add_module_neighb_search()
sense_grid_part.add_module_neighb_search(max_neighb_num=val_i(fluid_part.getPartNum()*32))

fluid_part.add_neighb_objs(neighb_list)
bound_part.add_neighb_objs(neighb_list)
sense_grid_part.add_neighb_obj(neighb_obj=fluid_part, search_range=val_f(sense_cell_size[None]*2))


fluid_part.add_solver_adv()
fluid_part.add_solver_sph()
fluid_part.add_solver_df(div_free_threshold=2e-4)

bound_part.add_solver_sph()
bound_part.add_solver_df(div_free_threshold=2e-4)

sense_grid_part.add_solver_sph()

world.init_modules()

world.neighb_search()

sense_output = Output_manager(format_type = Output_manager.type.SEQ, data_source = sense_grid_part)
sense_output.add_output_dataType("pos",2)
sense_output.add_output_dataType("node_index",2)
sense_output.add_output_dataType("sensed_density",1)
sense_output.add_output_dataType("vel",2)

# print('DEBUG sense_output', sense_output.np_node_index_organized)
# save as numpy file
# np.save("pos_np.npy", sense_output.np_node_index_organized)

def loop():
    world.update_pos_in_neighb_search()

    world.neighb_search()
    world.step_sph_compute_density()
    world.step_df_compute_alpha()
    world.step_df_div()
    print('div_free iter:', fluid_part.m_solver_df.div_free_iter[None])

    world.clear_acc()
    world.add_acc_gravity()
    fluid_part.m_solver_sph.loop_neighb(fluid_part.m_neighb_search.neighb_pool, fluid_part, fluid_part.m_solver_adv.inloop_accumulate_vis_acc)
    fluid_part.m_solver_sph.loop_neighb(fluid_part.m_neighb_search.neighb_pool, bound_part, fluid_part.m_solver_adv.inloop_accumulate_vis_acc)
    world.acc2vel()

    world.step_df_incomp()
    print('incomp iter:', fluid_part.m_solver_df.incompressible_iter[None])

    world.update_pos_from_vel()

    world.cfl_dt(0.5, max_time_step)

    print(' ')



def run(loop):
    inv_fps = 1/fps
    timer = 0
    sim_time = 0
    loop_count = 0

    while True:
        loop()
        loop_count += 1
        sim_time += world.g_dt[None]
        if(sim_time > timer*inv_fps):
            sense_output.export_to_numpy(index=output_shift+timer,path='./output')
            timer += 1
        if timer > output_frame_num:
            break

def run(loop):
    inv_fps = 1/fps
    timer = int(0)
    sim_time = float(0.0)
    loop_count = int(0)

    gui2d_part = Gui2d(objs=[fluid_part, bound_part], radius=world.getPartSize(), lb=vec2f([-8,-8]),rt=vec2f([8,8]))
    gui2d_grid = Gui2d(objs=[sense_grid_part], radius=sense_grid_part.getPartSize(), lb=vec2f([-8,-8]),rt=vec2f([8,8]))

    while sim_time < 30.0:
        loop()
        loop_count += 1
        sim_time += world.getdDt()
        if(sim_time > timer*inv_fps):
            
            sense_grid_part.copy_attr(from_attr=sense_grid_part.sph.density, to_attr=sense_grid_part.sensed_density)
            sense_grid_part.m_solver_sph.sph_avg_velocity(sense_grid_part.m_neighb_search.neighb_pool)
            sense_output.export_to_numpy(index=output_shift+timer,path='./output')

            gui2d_part.save_img(path='./output/part_'+str(output_shift+timer)+'.png')
            
            sense_grid_part.clamp_val_to_arr(sense_grid_part.sph.density, 0, 1000, sense_grid_part.rgb)
            gui2d_grid.save_img(path='./output/grid_'+str(output_shift+timer)+'.png')

            timer += 1



run(loop)







