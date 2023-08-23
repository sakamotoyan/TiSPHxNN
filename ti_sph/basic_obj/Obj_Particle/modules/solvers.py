import taichi as ti
from ....basic_solvers.Solver_adv import Adv_slover
from ....basic_solvers.Solver_df import DF_solver
from ....basic_solvers.Solver_sph import SPH_solver
from ....basic_solvers.Solver_wcsph import WCSPH_solver
from ....basic_solvers.Solver_ism import Implicit_mixture_solver

def add_solver_adv(self):
    self.m_solver_adv = Adv_slover(self)

def add_solver_sph(self):
    self.m_solver_sph = SPH_solver(self)

def add_solver_df(self, incompressible_threshold: ti.f32 = 1e-4, div_free_threshold: ti.f32 = 1e-3, incompressible_iter_max: ti.i32 = 100, div_free_iter_max: ti.i32 = 50, incomp_warm_start: bool = False, div_warm_start: bool = False):
    self.m_solver_df = DF_solver(self, incompressible_threshold, div_free_threshold, incompressible_iter_max, div_free_iter_max, incomp_warm_start, div_warm_start)

def add_solver_wcsph(self, gamma: ti.f32 = 7, max_vel: ti.f32 = 10, eta: ti.f32 = 0.01):
    self.m_solver_wcsph = WCSPH_solver(self, gamma, max_vel, eta)

def add_solver_ism(self, Cd, Cf):
    self.m_solver_ism = Implicit_mixture_solver(self, Cd, Cf, self.m_world)