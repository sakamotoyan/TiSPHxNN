import taichi as ti

ti.init(arch=ti.gpu)

@ti.kernel
def randVec(vec: ti.template(), lower: ti.f32, upper: ti.f32):
    for i in range(vec.shape[0]):
        vec[i] = ti.Vector([ti.random() * (upper - lower) + lower, ti.random() * (upper - lower) + lower])

@ti.kernel
def randScalar(scalar: ti.template(), lower: ti.f32, upper: ti.f32):
    for i in range(scalar.shape[0]):
        scalar[i] = ti.random() * (upper - lower) + lower

@ti.kernel
def naturalSeq(scalar: ti.template()):
    for i in range(scalar.shape[0]):
        scalar[i] = i

@ti.kernel
def naturalVec(vec: ti.template()):
    for i in range(vec.shape[0]):
        for n in ti.static(range(vec.n)):
            vec[i][n] = i

@ti.kernel
def hashVec(vec: ti.template(), hashed: ti.template()):
    seed = ti.static(124232,43351221,4623421)
    for i in range(vec.shape[0]):
        for j in ti.static(range(vec.n)):
            hashed[i] += ti.u32(vec[i][j]) * seed[j] & ti.u32(0xFFFFFFFF)

@ti.kernel
def hashMod(hashVal: ti.template()):
    for i in range(hashVal.shape[0]):
        hashVal[i] = hashVal[i] % hashVal.shape[0]
           

@ti.kernel
def computeStartPoint(hashed_part_id: ti.template(), start_point: ti.template()):
    start_point.fill(ti.u32(0xFFFFFFFF))
    start_point[hashed_part_id[0]] = 0
    for i in range(1, hashed_part_id.shape[0]):
        if hashed_part_id[i] != hashed_part_id[i-1]:
            start_point[hashed_part_id[i]] = i

# a_type = ti.types.struct(
#     pos = ti.types.vector(2, dtype=ti.f32),
#     id  = ti.u32,
# )

# a               = a_type.field(shape=10)
# hashed_a        = ti.field(dtype=ti.u32, shape=a.shape[0])
# start_point_a   = ti.field(dtype=ti.u32, shape=a.shape[0])
# naturalVec(a.pos)
# naturalSeq(a.id)
# a.pos[0].fill(99)
# a.pos[3].fill(99)
# hashVec(a.pos, hashed_a)
# hashMod(hashed_a)
# ti.algorithms.parallel_sort(hashed_a, a)
# computeStartPoint(hashed_a, start_point_a)
# print(a.pos)
# print(a.id)
# print(hashed_a)
# print(start_point_a)
# print(start_point_a[0] == 0xFFFFFFFF)

@ti.kernel
def iden(a: ti.template()):
    a[0] = ti.math.eye(a.n)

a = ti.Matrix.field(2, 2, dtype=ti.f32, shape=10)
a[0] = [[1,2],[3,4]]
b=ti.Vector([1,2])
print((a[0]*a[0]).sum())
