from plyfile import *

def write_ply(path, frame_num, dim, num, pos, phase_num, volume_frac, phase_vel_flag, phase_vel):
    if dim == 3:
        list_pos = []
        for i in range(num):
            pos_tmp = [pos[i, 0], pos[i, 1], pos[i, 2]]
            for j in range(phase_num):
                pos_tmp.append(volume_frac[i, j])
            if phase_vel_flag:
                for j in range(dim):
                    pos_tmp.append(phase_vel[i, 0, j])
                for j in range(dim):
                    pos_tmp.append(phase_vel[i, 1, j])
            list_pos.append(tuple(pos_tmp))
    elif dim == 2:
        list_pos = [(pos[i, 0], pos[i, 1], 0) for i in range(num)]
    else:
        print('write_ply(): dim exceeds default values')
        return
    data_type = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    for k in range(phase_num):
        data_type.append(('f'+str(k+1),'f4'))
    if phase_vel_flag:
        for j in range(dim):
            data_type.append(('pv0_'+str(j+1),'f4'))
        for j in range(dim):
            data_type.append(('pv1_'+str(j+1),'f4'))
    np_pos = np.array(list_pos, dtype=data_type)
    el_pos = PlyElement.describe(np_pos, 'vertex')
    PlyData([el_pos]).write(str(path) + '_' + str(frame_num) + '.ply')