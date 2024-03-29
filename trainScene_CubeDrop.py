import taichi as ti
from ti_sph import *
from template_part import part_template
from template_grid import grid_template
import time
import os
import sys
import numpy as np
from Dataset_processing import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--scene', type=int, default=0)
args = parser.parse_args()

output_path = './output'
clear_folder(output_path)


np.set_printoptions(threshold=sys.maxsize)

''' TAICHI SETTINGS '''
# ti.init(arch=ti.cuda, kernel_profiler=True) 
ti.init(arch=ti.cuda) # Use GPU
# ti.init(arch=ti.vulkan) # Use CPU

''' GLOBAL SETTINGS '''
output_shift = 0
output_frame_num = 2000
fps = 30
sense_res = 518

sense_cell_size = val_f(7.0/sense_res)
part_size = sense_cell_size[None] / 3

max_time_step = part_size/100
k_vis = 1e-4
world = World(dim=2, lb=-9, rt=9)
world.setPartSize(part_size)
world.setDt(max_time_step)

print('default dt', world.getdDt())

'''BASIC SETTINGS FOR FLUID'''
pool_data = Squared_pool_2D_data(container_height=16, container_size=8, fluid_height=8, span=world.g_part_size[None]*1.001, layer=3)
sense_cube_data = Cube_data(type=Cube_data.FIXED_GRID_RES, span=sense_cell_size[None], grid_res=vec2i(sense_res,sense_res),grid_center=vec2f(0,-4))
fluid_rest_density = 1000
if args.scene == 0:
    flui_cube_data_list = [
        Cube_data(type=Cube_data.FIXED_CELL_SIZE, lb=vec2f(-3.5,  0.2), rt=vec2f(-3.0,  7.9), span=world.g_part_size[None]*1.001),
        Cube_data(type=Cube_data.FIXED_CELL_SIZE, lb=vec2f(-2.8,  0.2), rt=vec2f(-2.0,  7.0), span=world.g_part_size[None]*1.001),
        Cube_data(type=Cube_data.FIXED_CELL_SIZE, lb=vec2f( 1.5,  0.2), rt=vec2f( 3.0,  6.0), span=world.g_part_size[None]*1.001),
    ]
elif args.scene == 1:
    flui_cube_data_list = [
        Cube_data(type=Cube_data.FIXED_CELL_SIZE, lb=vec2f(-3.9,  0.2), rt=vec2f(-0.0,  7.9), span=world.g_part_size[None]*1.001)
    ]
elif args.scene == 2:
    flui_cube_data_list = [
        Cube_data(type=Cube_data.FIXED_CELL_SIZE, lb=vec2f(-3.9,  0.2), rt=vec2f(-0.0,  7.9), span=world.g_part_size[None]*1.001),
        Cube_data(type=Cube_data.FIXED_CELL_SIZE, lb=vec2f( 1.5,  0.2), rt=vec2f( 3.9,  6.0), span=world.g_part_size[None]*1.001),
    ]

# fluid_cube_data_3 = Cube_data(type=Cube_data.FIXED_CELL_SIZE, lb=vec2f(1, -3), rt=vec2f(3.5, 3), span=world.g_part_size[None]*1.001)
'''INIT AN FLUID PARTICLE OBJECT'''
fluid_part_num = pool_data.fluid_part_num + sum([flui_cube_data_list[i].num for i in range(len(flui_cube_data_list))])
print("fluid_part_num", fluid_part_num)
fluid_part = Particle(part_num=fluid_part_num, part_size=world.g_part_size, is_dynamic=True)
world.attachPartObj(fluid_part)
fluid_part.instantiate_from_template(part_template, world)
'''PUSH PARTICLES TO THE OBJECT'''
fluid_part.open_stack(pool_data.fluid_part_num)
fluid_part.fill_open_stack_with_nparr(fluid_part.pos, pool_data.fluid_part_pos)
fluid_part.fill_open_stack_with_val(fluid_part.size, fluid_part.getPartSize())
fluid_part.fill_open_stack_with_val(fluid_part.volume, fluid_part.getPartSize()**world.getDim())
fluid_part.fill_open_stack_with_val(fluid_part.mass, fluid_rest_density*fluid_part.getPartSize()**world.getDim())
fluid_part.fill_open_stack_with_val(fluid_part.rest_density, fluid_rest_density)
fluid_part.fill_open_stack_with_val(fluid_part.rgb, vec3f([0.0, 0.0, 1.0]))
fluid_part.fill_open_stack_with_val(fluid_part.k_vis, k_vis)
fluid_part.close_stack()

for i in range(len(flui_cube_data_list)):
    fluid_part.open_stack(flui_cube_data_list[i].num)
    fluid_part.fill_open_stack_with_nparr(fluid_part.pos, flui_cube_data_list[i].pos)
    fluid_part.fill_open_stack_with_val(fluid_part.size, fluid_part.getPartSize())
    fluid_part.fill_open_stack_with_val(fluid_part.volume, fluid_part.getPartSize()**world.getDim())
    fluid_part.fill_open_stack_with_val(fluid_part.mass, fluid_rest_density*fluid_part.getPartSize()**world.getDim())
    fluid_part.fill_open_stack_with_val(fluid_part.rest_density, fluid_rest_density)
    fluid_part.fill_open_stack_with_val(fluid_part.rgb, vec3f([0.0, 0.0, 1.0]))
    fluid_part.fill_open_stack_with_val(fluid_part.k_vis, k_vis)
    fluid_part.close_stack()

# fluid_part.open_stack(fluid_cube_data_1.num)
# fluid_part.fill_open_stack_with_nparr(fluid_part.pos, fluid_cube_data_1.pos)
# fluid_part.fill_open_stack_with_val(fluid_part.size, fluid_part.getObjPartSize())
# fluid_part.fill_open_stack_with_val(fluid_part.volume, fluid_part.getObjPartSize()**world.getWorldDim())
# fluid_part.fill_open_stack_with_val(fluid_part.mass, fluid_rest_density*fluid_part.getObjPartSize()**world.getWorldDim())
# fluid_part.fill_open_stack_with_val(fluid_part.rest_density, fluid_rest_density)
# fluid_part.fill_open_stack_with_val(fluid_part.rgb, vec3f([0.0, 0.0, 1.0]))
# fluid_part.fill_open_stack_with_val(fluid_part.k_vis, k_vis)
# fluid_part.close_stack()

# fluid_part.open_stack(fluid_cube_data_2.num)
# fluid_part.fill_open_stack_with_nparr(fluid_part.pos, fluid_cube_data_2.pos)
# fluid_part.fill_open_stack_with_val(fluid_part.size, fluid_part.getObjPartSize())
# fluid_part.fill_open_stack_with_val(fluid_part.volume, fluid_part.getObjPartSize()**world.getWorldDim())
# fluid_part.fill_open_stack_with_val(fluid_part.mass, fluid_rest_density*fluid_part.getObjPartSize()**world.getWorldDim())
# fluid_part.fill_open_stack_with_val(fluid_part.rest_density, fluid_rest_density)
# fluid_part.fill_open_stack_with_val(fluid_part.rgb, vec3f([1.0, 0.0, 0.0]))
# fluid_part.fill_open_stack_with_val(fluid_part.k_vis, k_vis)
# fluid_part.close_stack()

# fluid_part.open_stack(fluid_cube_data_3.num)
# fluid_part.fill_open_stack_with_nparr(fluid_part.pos, fluid_cube_data_3.pos)
# fluid_part.fill_open_stack_with_val(fluid_part.size, fluid_part.getObjPartSize())
# fluid_part.fill_open_stack_with_val(fluid_part.volume, fluid_part.getObjPartSize()**world.getWorldDim())
# fluid_part.fill_open_stack_with_val(fluid_part.mass, fluid_rest_density*fluid_part.getObjPartSize()**world.getWorldDim())
# fluid_part.fill_open_stack_with_val(fluid_part.rest_density, fluid_rest_density)
# fluid_part.fill_open_stack_with_val(fluid_part.rgb, vec3f([1.0, 0.0, 0.0]))
# fluid_part.fill_open_stack_with_val(fluid_part.k_vis, k_vis)
# fluid_part.close_stack()


''' INIT BOUNDARY PARTICLE OBJECT '''
bound_rest_density = 1000
bound_part = Particle(part_num=pool_data.bound_part_num, part_size=world.g_part_size, is_dynamic=False)
world.attachPartObj(bound_part)
bound_part.instantiate_from_template(part_template, world)
bound_part.open_stack(pool_data.bound_part_num)
bound_part.fill_open_stack_with_nparr(bound_part.pos, pool_data.bound_part_pos)
bound_part.fill_open_stack_with_val(bound_part.size, bound_part.getPartSize())
bound_part.fill_open_stack_with_val(bound_part.volume, bound_part.getPartSize()**world.getDim())
bound_part.fill_open_stack_with_val(bound_part.mass, bound_rest_density*bound_part.getPartSize()**world.getDim())
bound_part.fill_open_stack_with_val(bound_part.rest_density, bound_rest_density)
fluid_part.fill_open_stack_with_val(bound_part.rgb, vec3f([0.0, 0.0, 0.0]))
bound_part.fill_open_stack_with_val(bound_part.k_vis, k_vis)
bound_part.close_stack()

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
fluid_part.add_solver_df(div_free_threshold=1e-4)

bound_part.add_solver_sph()
bound_part.add_solver_df(div_free_threshold=1e-4)

sense_grid_part.add_solver_sph()

world.init_modules()

world.neighb_search()

sense_output = Output_manager(format_type = Output_manager.type.SEQ, data_source = sense_grid_part)
sense_output.add_output_dataType("pos")
sense_output.add_output_dataType("node_index")
sense_output.add_output_dataType("sensed_density")
sense_output.add_output_dataType("vel")
sense_output.add_output_dataType("strainRate")

# print('DEBUG sense_output', sense_output.np_node_index_organized)
# save as numpy file
# np.save("pos_np.npy", sense_output.np_node_index_organized)

def loop(is_log_loop=False):
    world.update_pos_in_neighb_search()

    world.neighb_search()
    world.step_sph_compute_density()
    world.step_df_compute_alpha()
    world.step_df_div()
    # print('div_free iter:', fluid_part.m_solver_df.div_free_iter[None])
    
    if(is_log_loop):
        fluid_part.m_solver_sph.sph_compute_strainRate(fluid_part, fluid_part.m_neighb_search.neighb_pool)

    world.clear_acc()
    world.add_acc_gravity()
    fluid_part.m_solver_sph.loop_neighb(fluid_part.m_neighb_search.neighb_pool, fluid_part, fluid_part.m_solver_adv.inloop_accumulate_vis_acc)
    fluid_part.m_solver_sph.loop_neighb(fluid_part.m_neighb_search.neighb_pool, bound_part, fluid_part.m_solver_adv.inloop_accumulate_vis_acc)
    world.acc2vel()
    
    world.step_df_incomp()
    # print('incomp iter:', fluid_part.m_solver_df.incompressible_iter[None])

    world.update_pos_from_vel()

    world.cfl_dt(0.5, max_time_step)

    # print(' ')


def run(loop):
    inv_fps = 1/fps
    timer = int(0)
    sim_time = float(0.0)
    loop_count = int(0)

    gui2d_part = Gui2d(objs=[fluid_part, bound_part], radius=world.getPartSize(), lb=vec2f([-8,-8]),rt=vec2f([8,8]))
    gui2d_grid = Gui2d(objs=[sense_grid_part], radius=sense_grid_part.getPartSize(), lb=vec2f([-8,-8]),rt=vec2f([8,8]))

    while timer < output_frame_num:
        is_log_loop = sim_time > timer*inv_fps

        loop(is_log_loop)
        loop_count += 1
        sim_time += world.getdDt()

        if(is_log_loop):
            
            sense_grid_part.copy_attr(from_attr=sense_grid_part.sph.density, to_attr=sense_grid_part.sensed_density)
            sense_grid_part.m_solver_sph.sph_avg_velocity(sense_grid_part.m_neighb_search.neighb_pool)
            
            sense_grid_part.m_solver_sph.sph_avg_strainRate(sense_grid_part.m_neighb_search.neighb_pool)

            sense_output.export_to_numpy(index=output_shift+timer,path=output_path)

            gui2d_part.save_img(path=os.path.join(output_path,'part_'+str(output_shift+timer)+'.png'))
            
            sense_grid_part.clamp_val_to_arr(sense_grid_part.sph.density, 0, 1000, sense_grid_part.rgb)
            gui2d_grid.save_img(path=os.path.join(output_path,'grid_'+str(output_shift+timer)+'.png'))

            timer += 1

run(loop)







