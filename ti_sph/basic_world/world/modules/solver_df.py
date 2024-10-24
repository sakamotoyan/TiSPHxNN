import taichi as ti
import numpy as np
from typing import List
from ....basic_obj.Obj_Particle import Particle

def init_solver_df(self):
    self.df_solver_list = []
    for part_obj in self.part_obj_list:
        if part_obj.getSolverDF() is not None:
            self.df_solver_list.append(part_obj)
    self.df_incompressible_states = [False for _ in range(len(self.df_solver_list))]
    self.df_divergence_free_states = [False for _ in range(len(self.df_solver_list))]

def step_df_compute_alpha(self):
    for part_obj in self.df_solver_list:
        part_obj: Particle
        part_obj.getSolverDF().compute_alpha(part_obj.m_neighb_search.neighb_pool)

def step_df_compute_beta(self):
    for part_obj in self.df_solver_list:
        part_obj: Particle
        part_obj.getSolverDF().compute_beta(part_obj.m_neighb_search.neighb_pool)

def step_df_incomp(self):
    for part_obj in self.df_solver_list:
        part_obj: Particle
        part_obj.getSolverDF().incompressible_iter[None] = 0

        if part_obj.m_is_dynamic:
            part_obj.getSolverDF().set_vel_adv(part_obj.vel)
            self.df_incompressible_states[self.df_solver_list.index(part_obj)] = False
        else:
            self.df_incompressible_states[self.df_solver_list.index(part_obj)] = True
        
    while True:
        for part_obj in self.df_solver_list:
            if not part_obj.m_is_dynamic:
                continue
            
            part_obj.getSolverDF().incompressible_iter[None] += 1

            part_obj.getSolverDF().compute_delta_density()

            for neighb_obj in part_obj.get_module_neighbSearch().neighb_obj_list:
                part_obj.get_module_neighbSearch().loop_neighb(neighb_obj, part_obj.getSolverDF().inloop_update_delta_density_from_vel_adv)
            part_obj.getSolverDF().ReLU_delta_density()
            part_obj.getSolverDF().update_df_compressible_ratio()

            if part_obj.getSolverDF().compressible_ratio[None] < part_obj.getSolverDF().incompressible_threshold[None] \
                or part_obj.getSolverDF().incompressible_iter[None] > part_obj.getSolverDF().incompressible_iter_max[None]:
                self.df_incompressible_states[self.df_solver_list.index(part_obj)] = True

        for part_obj in self.df_solver_list:
            if part_obj.m_is_dynamic:
                for neighb_obj in part_obj.get_module_neighbSearch().neighb_obj_list:
                    part_obj.get_module_neighbSearch().loop_neighb(neighb_obj, part_obj.getSolverDF().inloop_update_vel_adv_from_alpha)
            
        if all(self.df_incompressible_states):
            break

    for part_obj in self.df_solver_list:
        if part_obj.m_is_dynamic:
            part_obj.getSolverDF().update_vel(part_obj.vel)

def step_df_div(self):
    for part_obj in self.df_solver_list:
        part_obj: Particle
        part_obj.getSolverDF().div_free_iter[None] = 0

        if part_obj.m_is_dynamic:
            part_obj.getSolverDF().set_vel_adv(part_obj.getVelArr())
            self.df_divergence_free_states[self.df_solver_list.index(part_obj)] = False
        else:
            self.df_divergence_free_states[self.df_solver_list.index(part_obj)] = True
        
    while True:
        for part_obj in self.df_solver_list:
            if not part_obj.m_is_dynamic:
                continue

            part_obj.getSolverDF().div_free_iter[None] += 1

            part_obj.getSolverDF().compute_delta_density()

            for neighb_obj in part_obj.get_module_neighbSearch().neighb_obj_list:
                part_obj.get_module_neighbSearch().loop_neighb(neighb_obj, part_obj.getSolverDF().inloop_update_delta_density_from_vel_adv)
            part_obj.getSolverDF().ReLU_delta_density()
            part_obj.getSolverDF().update_df_compressible_ratio()
            # print('compressible ratio during', part_obj.getSolverDF().compressible_ratio[None])

            if part_obj.getSolverDF().compressible_ratio[None] < part_obj.getSolverDF().div_free_threshold[None] \
                or part_obj.getSolverDF().div_free_iter[None] > part_obj.getSolverDF().div_free_iter_max[None]:
                self.df_divergence_free_states[self.df_solver_list.index(part_obj)] = True
    
        for part_obj in self.df_solver_list:
            if part_obj.m_is_dynamic:
                for neighb_obj in part_obj.get_module_neighbSearch().neighb_obj_list:
                    part_obj.get_module_neighbSearch().loop_neighb(neighb_obj, part_obj.getSolverDF().inloop_update_vel_adv_from_alpha)
        
        if all(self.df_divergence_free_states):
            break

    for part_obj in self.df_solver_list:
        if part_obj.m_is_dynamic:
            part_obj.getSolverDF().update_vel(part_obj.vel)

def step_dfsph_incomp(self):
    for part_obj in self.df_solver_list:
        part_obj: Particle
        part_obj.getSolverDF().incompressible_iter[None] = 0

        if part_obj.m_is_dynamic:
            part_obj.getSolverDF().set_vel_adv(part_obj.vel)
            self.df_incompressible_states[self.df_solver_list.index(part_obj)] = False
        else:
            self.df_incompressible_states[self.df_solver_list.index(part_obj)] = True
    
    # Warm start
    for part_obj in self.df_solver_list:
        if part_obj.m_is_dynamic and part_obj.getSolverDF().incomp_warm_start:
            part_obj.clear(part_obj.sph_df.alpha_2)
            for neighb_obj in part_obj.get_module_neighbSearch().neighb_obj_list:
                part_obj.get_module_neighbSearch().loop_neighb(neighb_obj, part_obj.getSolverDF().inloop_df_update_vel_adv_from_kappa_incomp)

    while True:
        for part_obj in self.df_solver_list:
            if not part_obj.m_is_dynamic:
                continue
            
            part_obj.getSolverDF().incompressible_iter[None] += 1

            part_obj.getSolverDF().compute_delta_density()

            for neighb_obj in part_obj.get_module_neighbSearch().neighb_obj_list:
                part_obj.get_module_neighbSearch().loop_neighb(neighb_obj, part_obj.getSolverDF().inloop_update_delta_density_from_vel_adv)
            part_obj.getSolverDF().ReLU_delta_density()
            part_obj.getSolverDF().update_df_compressible_ratio()

            if part_obj.getSolverDF().compressible_ratio[None] < part_obj.getSolverDF().incompressible_threshold[None] \
                or part_obj.getSolverDF().incompressible_iter[None] > part_obj.getSolverDF().incompressible_iter_max[None]:
                self.df_incompressible_states[self.df_solver_list.index(part_obj)] = True

        for part_obj in self.df_solver_list:
            if part_obj.m_is_dynamic:
                part_obj.getSolverDF().compute_kappa_incomp_from_delta_density()

        for part_obj in self.df_solver_list:
            if part_obj.m_is_dynamic:
                for neighb_obj in part_obj.get_module_neighbSearch().neighb_obj_list:
                    part_obj.get_module_neighbSearch().loop_neighb(neighb_obj, part_obj.getSolverDF().inloop_df_update_vel_adv_from_kappa_incomp)
            
        if all(self.df_incompressible_states):
            break

    for part_obj in self.df_solver_list:
        if part_obj.m_is_dynamic:
            part_obj.getSolverDF().log_kappa_incomp()

    for part_obj in self.df_solver_list:
        if part_obj.m_is_dynamic:
            part_obj.getSolverDF().update_vel(part_obj.vel)

def step_dfsph_div(self):
    for part_obj in self.df_solver_list:
        part_obj: Particle
        part_obj.getSolverDF().div_free_iter[None] = 0

        if part_obj.m_is_dynamic:
            part_obj.getSolverDF().set_vel_adv(part_obj.vel)
            self.df_divergence_free_states[self.df_solver_list.index(part_obj)] = False
        else:
            self.df_divergence_free_states[self.df_solver_list.index(part_obj)] = True
    
    # Warm start
    for part_obj in self.df_solver_list:
        if part_obj.m_is_dynamic and part_obj.getSolverDF().div_warm_start:
            part_obj.clear(part_obj.sph_df.alpha_2)
            for neighb_obj in part_obj.get_module_neighbSearch().neighb_obj_list:
                part_obj.get_module_neighbSearch().loop_neighb(neighb_obj, part_obj.getSolverDF().inloop_df_update_vel_adv_from_kappa_div)

    while True:
        for part_obj in self.df_solver_list:
            if not part_obj.m_is_dynamic:
                continue

            part_obj.getSolverDF().div_free_iter[None] += 1

            part_obj.getSolverDF().compute_delta_density()

            for neighb_obj in part_obj.get_module_neighbSearch().neighb_obj_list:
                part_obj.get_module_neighbSearch().loop_neighb(neighb_obj, part_obj.getSolverDF().inloop_update_delta_density_from_vel_adv)
            part_obj.getSolverDF().ReLU_delta_density()
            part_obj.getSolverDF().update_df_compressible_ratio()
            # print('compressible ratio during', part_obj.getSolverDF().compressible_ratio[None])

            if part_obj.getSolverDF().compressible_ratio[None] < part_obj.getSolverDF().div_free_threshold[None] \
                or part_obj.getSolverDF().div_free_iter[None] > part_obj.getSolverDF().div_free_iter_max[None]:
                self.df_divergence_free_states[self.df_solver_list.index(part_obj)] = True
    
        for part_obj in self.df_solver_list:
            if part_obj.m_is_dynamic:
                part_obj.getSolverDF().compute_kappa_div_from_delta_density()
        
        for part_obj in self.df_solver_list:
            if part_obj.m_is_dynamic:
                for neighb_obj in part_obj.get_module_neighbSearch().neighb_obj_list:
                    part_obj.get_module_neighbSearch().loop_neighb(neighb_obj, part_obj.getSolverDF().inloop_df_update_vel_adv_from_kappa_div)
        
        if all(self.df_divergence_free_states):
            break

    for part_obj in self.df_solver_list:
        if part_obj.m_is_dynamic:
            part_obj.getSolverDF().log_kappa_div()

    for part_obj in self.df_solver_list:
        if part_obj.m_is_dynamic:
            part_obj.getSolverDF().update_vel(part_obj.vel)

def step_vfsph_incomp(self, update_vel=True):
    for part_obj in self.df_solver_list:
        part_obj: Particle
        part_obj.getSolverDF().incompressible_iter[None] = 0

        if part_obj.m_is_dynamic:
            part_obj.getSolverDF().set_vel_adv(part_obj.getVelArr())
            self.df_incompressible_states[self.df_solver_list.index(part_obj)] = False
        else:
            self.df_incompressible_states[self.df_solver_list.index(part_obj)] = True
    
    # Warm start
    for part_obj in self.df_solver_list:
        if part_obj.m_is_dynamic and part_obj.getSolverDF().incomp_warm_start:
            part_obj.clear(part_obj.sph_df.alpha_2)
            for neighb_obj in part_obj.get_module_neighbSearch().neighb_obj_list:
                part_obj.get_module_neighbSearch().loop_neighb(neighb_obj, part_obj.getSolverDF().inloop_vf_update_vel_adv_from_kappa_incomp)

    while True:
        for part_obj in self.df_solver_list:
            # if not part_obj.m_is_dynamic:
            #     continue
            
            part_obj.getSolverDF().incompressible_iter[None] += 1

            part_obj.getSolverDF().compute_delta_compression_ratio()

            for neighb_obj in part_obj.get_module_neighbSearch().neighb_obj_list:
                part_obj.get_module_neighbSearch().loop_neighb(neighb_obj, part_obj.getSolverDF().inloop_update_delta_compression_ratio_from_vel_adv)
            part_obj.getSolverDF().ReLU_delta_compression_ratio()
            part_obj.getSolverDF().update_vf_compressible_ratio()

            if part_obj.getSolverDF().compressible_ratio[None] < part_obj.getSolverDF().incompressible_threshold[None] \
                or part_obj.getSolverDF().incompressible_iter[None] > part_obj.getSolverDF().incompressible_iter_max[None]:
                self.df_incompressible_states[self.df_solver_list.index(part_obj)] = True

        # for part_obj in self.df_solver_list:
            part_obj.getSolverDF().compute_kappa_incomp_from_delta_compression_ratio()
            if part_obj.m_is_dynamic:
                for neighb_obj in part_obj.get_module_neighbSearch().neighb_obj_list:
                    part_obj.get_module_neighbSearch().loop_neighb(neighb_obj, part_obj.getSolverDF().inloop_vf_update_vel_adv_from_kappa_incomp)
            
        if all(self.df_incompressible_states) and part_obj.getSolverDF().incompressible_iter[None]>1:
            break

    for part_obj in self.df_solver_list:
        if part_obj.m_is_dynamic:
            part_obj.getSolverDF().log_kappa_incomp()

    if update_vel:
        for part_obj in self.df_solver_list:
            if part_obj.m_is_dynamic:
                part_obj.getSolverDF().update_vel(part_obj.vel)

def step_vfsph_div(self, update_vel=True):
    for part_obj in self.df_solver_list:
        part_obj: Particle
        part_obj.getSolverDF().div_free_iter[None] = 0

        if part_obj.m_is_dynamic:
            part_obj.getSolverDF().set_vel_adv(part_obj.vel)
            self.df_divergence_free_states[self.df_solver_list.index(part_obj)] = False
        else:
            self.df_divergence_free_states[self.df_solver_list.index(part_obj)] = True
    
    # Warm start
    for part_obj in self.df_solver_list:
        if part_obj.m_is_dynamic and part_obj.getSolverDF().div_warm_start:
            part_obj.clear(part_obj.sph_df.alpha_2)
            for neighb_obj in part_obj.get_module_neighbSearch().neighb_obj_list:
                part_obj.get_module_neighbSearch().loop_neighb(neighb_obj, part_obj.getSolverDF().inloop_vf_update_vel_adv_from_kappa_div)

    while True:
        for part_obj in self.df_solver_list:
            # if not part_obj.m_is_dynamic:
            #     continue
            
            part_obj.getSolverDF().div_free_iter[None] += 1
            
            part_obj.getSolverDF().compute_delta_compression_ratio()

            for neighb_obj in part_obj.get_module_neighbSearch().neighb_obj_list:
                part_obj.get_module_neighbSearch().loop_neighb(neighb_obj, part_obj.getSolverDF().inloop_update_delta_compression_ratio_from_vel_adv)
            part_obj.getSolverDF().ReLU_delta_compression_ratio()
            part_obj.getSolverDF().update_vf_compressible_ratio()

            if part_obj.getSolverDF().compressible_ratio[None] < part_obj.getSolverDF().div_free_threshold[None] \
                or part_obj.getSolverDF().div_free_iter[None] > part_obj.getSolverDF().div_free_iter_max[None]:
                self.df_divergence_free_states[self.df_solver_list.index(part_obj)] = True

        # for part_obj in self.df_solver_list:
            part_obj.getSolverDF().compute_kappa_div_from_delta_compression_ratio()
            if part_obj.m_is_dynamic:
                for neighb_obj in part_obj.get_module_neighbSearch().neighb_obj_list:
                    part_obj.get_module_neighbSearch().loop_neighb(neighb_obj, part_obj.getSolverDF().inloop_vf_update_vel_adv_from_kappa_div)
            
        if all(self.df_divergence_free_states):
            break

    for part_obj in self.df_solver_list:
        if part_obj.m_is_dynamic:
            part_obj.getSolverDF().log_kappa_div()

    if update_vel:
        for part_obj in self.df_solver_list:
            if part_obj.m_is_dynamic:
                part_obj.getSolverDF().update_vel(part_obj.vel)