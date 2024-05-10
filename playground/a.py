import taichi as ti
ti.init()
a = ti.Vector.field(3, ti.f32, (10))
print(a.n)