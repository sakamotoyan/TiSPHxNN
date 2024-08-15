from ....basic_obj.Obj_Particle import Particle

def init_solver_elastic(self):
    self.elastic_solver_list = []
    for part_obj in self.part_obj_list:
        part_obj: Particle
        if part_obj.getSolverElastic() is not None:
            self.elastic_solver_list.append(part_obj)

def step_elastic_clear_force(self):
    for part_obj in self.elastic_solver_list:
        part_obj: Particle
        part_obj.getSolverElastic().clear_force()