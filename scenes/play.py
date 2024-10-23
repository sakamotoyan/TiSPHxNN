import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from scenes.scene_import import *

# ------------------------------------------------
# [1] 模拟参数设置
# 粒子大小、时间步大小、模拟时间限制、粘度、重力加速度、相数、帧率、模拟空间维度、扩散和漂移参数、2D可视化界面像素
# ------------------------------------------------
''' TAICHI SETTINGS '''
# ti.init(arch=ti.gpu) 
# ti.init(arch=ti.cuda, device_memory_GB=6) 
# ti.init(arch=ti.cuda,device_memory_fraction=0.9)
ti.init(arch=ti.vulkan)
''' GLOBAL SETTINGS SIMULATION '''
part_size                   = 0.1            # Unit: m
max_time_step               = part_size/50    # Unit: s
sim_time_limit              = 50.0            # Unit: s
kinematic_viscosity_fluid   = 0.001           # Unit: Pa s^-1
gravity_acc                 = -9.8            # Unit: m s^-2
phase_num                   = 3
fps                         = 60
sim_dimension               = 3
Cf                          = 0.0 
Cd                          = 0.0 
dpi                         = 200

# ------------------------------------------------
# [2] 世界参数设置
# 世界边界、多相流密度、颜色设置，参数传递（维度、粒子大小、时间步、重力加速度）
# ------------------------------------------------
''' INIT SIMULATION WORLD '''
world = tsph.World(dim=sim_dimension, lb=-5, rt=5)
world.setPartSize(part_size) 
world.setDt(max_time_step) 
world.set_multiphase(phase_num,[tsph.vec3f(0.2,0.0,0.8),tsph.vec3f(0.8,0,0.2),tsph.vec3f(0,0,1)],[500,1000,1000]) 
world.setGravityMagnitude(gravity_acc)

# ------------------------------------------------
# [3] 模拟物质参数设置
# 物质形状-->粒子数量，粒子位置-->为粒子属性数组赋值（位置、粒子大小、体积、相分数等等）
# ------------------------------------------------
''' DATA SETTINGS FOR FLUID PARTICLE '''
fluid_cube_data = tsph.Cube_data(type=tsph.Cube_data.FIXED_CELL_SIZE, lb=tsph.vec3f(1.2,-1,-1), rt=tsph.vec3f(2.9,0.8,1.8), span=world.g_part_size[None]*1.001)
boundary_box_data = tsph.Box_data(lb=tsph.vec3f(-1.1,-1.1,-1.1), rt=tsph.vec3f(3, 1, 2), span=world.g_part_size[None]*1.001,layers=2)
#粒子数较少，会炸
elastic_cube_data=tsph.Cube_data(type=tsph.Cube_data.FIXED_CELL_SIZE, lb=tsph.vec3f(-1,-1,-1), rt=tsph.vec3f(-0.7, -0.7,-0.7), span=world.g_part_size[None]*1.001)
#粒子数较多，正常
# elastic_cube_data=tsph.Cube_data(type=tsph.Cube_data.FIXED_CELL_SIZE, lb=tsph.vec3f(-1,-1,-1), rt=tsph.vec3f(1, 0.75,1), span=world.g_part_size[None]*1.001)


fluid_part_num = fluid_cube_data.num
bound_part_num = boundary_box_data.num
elastic_part_num=elastic_cube_data.num

print("fluid_part_num",fluid_part_num)
print("elastic_part_num",elastic_part_num)
print("bound_part_num",bound_part_num)

# --- 流体部分 ---
fluid_part = tsph.Particle(part_num=fluid_part_num, part_size=world.g_part_size, is_dynamic=True)
world.attachPartObj(fluid_part)
fluid_part.instantiate_from_template(part_template)
fluid_part.open_stack(fluid_part_num) # open the stack to feed data
fluid_part.fill_open_stack_with_nparr(fluid_part.pos, fluid_cube_data.pos)  # feed the position data
fluid_part.fill_open_stack_with_val(fluid_part.size, fluid_part.getPartSize()) # feed the particle size
fluid_part.fill_open_stack_with_val(fluid_part.volume, fluid_part.getPartSize()**world.g_dim[None]) # feed the particle volume
val_frac = ti.field(dtype=ti.f32, shape=phase_num) # create a field to store the volume fraction
val_frac[0], val_frac[1], val_frac[2] = 1.0,0.0,0.0 # set up the volume fraction
fluid_part.fill_open_stack_with_vals(fluid_part.phase.val_frac, val_frac) # feed the volume fraction
fluid_part.close_stack() # close the stack

# --- 边界部分 ---
bound_part = tsph.Particle(part_num=bound_part_num, part_size=world.g_part_size, is_dynamic=False)
world.attachPartObj(bound_part)
bound_part.instantiate_from_template(part_template)
bound_part.open_stack(bound_part_num)
bound_part.fill_open_stack_with_nparr(bound_part.pos, boundary_box_data.pos.to_numpy())
bound_part.fill_open_stack_with_val(bound_part.size, bound_part.getPartSize())
bound_part.fill_open_stack_with_val(bound_part.volume, bound_part.getPartSize()**world.getDim())
bound_part.fill_open_stack_with_val(bound_part.mass, 1000*bound_part.getPartSize()**world.getDim())
bound_part.fill_open_stack_with_val(bound_part.rest_density, 1000)
bound_part.close_stack()

# --- 固体部分 ---
elastic_part = tsph.Particle(part_num=elastic_part_num,part_size=world.g_part_size, is_dynamic=True)
world.attachPartObj(elastic_part)
elastic_part.instantiate_from_template(part_template)
elastic_part.open_stack(elastic_part_num) # open the stack to feed data
elastic_part.fill_open_stack_with_nparr(elastic_part.pos, elastic_cube_data.pos) # feed the position data
elastic_part.fill_open_stack_with_val(elastic_part.size, elastic_part.getPartSize()) # feed the particle size
elastic_part.fill_open_stack_with_val(elastic_part.volume, elastic_part.getPartSize()**world.getDim()) # feed the particle volume
elastic_part.fill_open_stack_with_val(elastic_part.mass, 1000*elastic_part.getPartSize()**world.getDim())
elastic_part.fill_open_stack_with_val(elastic_part.rest_density,1000)
elastic_part.close_stack()

# ------------------------------------------------
# [4] 初始化邻居搜索模块
# 需要进行邻居搜索的物体列表，为物体增加相应的邻居列表
# ------------------------------------------------
'''INIT NEIGHBOR SEARCH OBJECTS'''
neighb_list=[fluid_part, bound_part, elastic_part]
fluid_part.add_module_neighb_search(neighb_list)
bound_part.add_module_neighb_search(neighb_list)
elastic_part.add_module_neighb_search(neighb_list)

# ------------------------------------------------
# [5] 初始化物体使用的各种求解器
# 流体一般会有平流求解器、sph基础求解器、DF求解器、ism（多相流）求解器
# ------------------------------------------------
'''INIT SOLVER OBJECTS'''
# the shared solver
# --- 流体部分 ---
fluid_part.add_solver_adv()
fluid_part.add_solver_sph()
fluid_part.add_solver_df(div_free_threshold=1e-4, incomp_warm_start=False, div_warm_start=False, incompressible_threshold=1e-3)
fluid_part.add_solver_ism(Cd=Cd, Cf=Cf, k_vis_inter=kinematic_viscosity_fluid, k_vis_inner=kinematic_viscosity_fluid)
# --- 边界部分 ---
bound_part.add_solver_sph()
bound_part.add_solver_df(div_free_threshold=2e-4)
# --- 固体部分 ---
elastic_part.add_solver_adv()
elastic_part.add_solver_sph()
elastic_part.add_solver_df(div_free_threshold=1e-4, incomp_warm_start=False, div_warm_start=False, incompressible_threshold=1e-3)
# elastic_part.add_solver_plastic(isFracture=False,lame_lambda=5e6,lame_mu=5e6,s_max=0.07,alpha=0)
# elastic_part.add_solver_elastic(lame_lambda=1e4, lame_mu=1e4)

# ------------------------------------------------
# [6] 世界初始化各种模块并定义GUI界面
# 流体一般会有平流求解器、sph基础求解器、DF求解器、ism（多相流）求解器
# ------------------------------------------------
world.init_modules()

# gui2d = tsph.Gui2d(objs=[fluid_part, bound_part], radius=world.g_part_size[None]*0.5, lb=tsph.vec2f(-6,-6),rt=tsph.vec2f(6,6),dpi=dpi)

# ------------------------------------------------
# [7] 进入正式loop前的预计算阶段
# 世界邻居搜索、更新多相流密度和质量、更新多相流颜色、恢复相速度（从混合物中）、初始化弹性体等
# ------------------------------------------------
''' DATA PREPERATIONs '''
def prep():
    world.neighb_search() # perform the neighbor search
    # elastic_part.get_module_neighbSearch().check_neighb(10)

    fluid_part.m_solver_ism.update_rest_density_and_mass()
    fluid_part.m_solver_ism.update_color() # update the color
    fluid_part.m_solver_ism.recover_phase_vel_from_mixture() # recover the phase velocity from the mixture velocitye elastic force
    # elastic_part.m_solver_plastic.init()
    # elastic_part.m_solver_elastic.init()
    # # print("self.obj.peridynamics.bond_count",elastic_part.peridynamics.bond_count)

# ------------------------------------------------
# [8] 正式loop
# 世界邻居搜索、计算压缩率、计算压强求解器参数、加重力加速度等等
# ------------------------------------------------
''' SIMULATION LOOPS '''
def loop():
    ''' update color based on the volume fraction '''
    
    # fluid_part.m_solver_ism.update_color()

    world.neighb_search()
    world.step_sph_compute_compression_ratio()
    print(elastic_part.getSphCompressionRatioArr())
    world.step_df_compute_beta()
    # # print('beta:', fluid_part.m_neighb_search.neighb_pool.xijNorm.to_numpy()[:1000])
    # world.step_vfsph_div(update_vel=True)
    # tsph.DEBUG('div_free iter:', fluid_part.m_solver_df.div_free_iter[None])

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
    # fluid_part.m_solver_sph.loop_neighb(fluid_part.m_neighb_search.neighb_pool, bound_part, fluid_part.getSolverAdv().inloop_accumulate_vis_acc)
    # elastic_part.m_solver_plastic.step()
    # elastic_part.m_solver_elastic.step()
    elastic_part.getSolverAdv().add_acc_gravity()
    # # print("afrer gravity self.obj.acc",elastic_part.acc)

    world.acc2vel()
    # print("afrer world.acc2vel self.obj.vel",elastic_part.vel)
    ''' [TIME END] Advection process '''

    world.step_vfsph_incomp(update_vel=True)
    # # print('incomp iter:', fluid_part.m_solver_df.incompressible_iter[None])

    '''  [TIME START] ISM Part 3 '''
    # fluid_part.m_solver_df.get_acc_pressure()
    # fluid_part.m_solver_ism.clear_phase_acc()
    # fluid_part.m_solver_ism.ditribute_acc_pressure_2_phase()
    # fluid_part.m_solver_ism.phase_acc_2_phase_vel()
    # fluid_part.m_solver_ism.update_vel_from_phase_vel()
    '''  [TIME END] ISM Part 3 '''
    # print("afrer step_vfsph_incomp self.obj.vel",elastic_part.vel)
    world.update_pos_from_vel()

    '''  [TIME START] ISM Part 4 '''
    # fluid_part.m_solver_ism.update_val_frac()
    # fluid_part.m_solver_ism.update_vel_from_phase_vel()
    # fluid_part.m_solver_ism.regularize_val_frac()
    # fluid_part.m_solver_ism.update_rest_density_and_mass()
    # fluid_part.m_solver_ism.update_vel_from_phase_vel()
    '''  [TIME END] ISM Part 4 '''


    # world.cfl_dt(0.4, max_time_step)

    ''' statistical info '''
    # # print(' ')
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
    # gui.op_system_run=True
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
            # gui.op_system_run=False
        if gui.op_refresh_window:
            gui.scene_setup()
            gui.scene_add_parts_colorful(obj_pos=fluid_part.pos, obj_color=fluid_part.rgb,index_count=fluid_part.getStackTop(),size=world.getPartSize()/2)
            gui.scene_add_parts(obj_pos=bound_part.pos, obj_color=(0.86,0.86,0.86),index_count=bound_part.getStackTop(),size=world.getPartSize()/12)
            gui.scene_add_parts(obj_pos=elastic_part.pos, obj_color=(0.9,0.9,0.05),index_count=elastic_part.getStackTop(),size=world.getPartSize()/2)
            gui.canvas.scene(gui.scene)  # Render the scene

            # if gui.op_save_img and flag_write_img:
            #     # gui.window.save_image('output/'+str(timer)+'.png')
            #     gui2d.save_img(path=os.path.join(parent_dir,'output/part_'+str(timer)+'.jpg'))
            #     flag_write_img = False

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
            # export_pos_to_numpy(os.path.join(parent_dir,'output_data/part_'+str(timer)+'.npy'))
            # export_flag_to_numpy(os.path.join(parent_dir,'output_data/flag_'+str(timer)+'.npy'))

''' RUN THE SIMULATION '''
prep()
# run(loop)
vis_run(loop)








