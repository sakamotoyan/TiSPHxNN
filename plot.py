import json
from plyfile import PlyData
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

path_ply = "./" 
path_img = "./" 
lb = (-8,-6)
rt = (8,6)
radius = 0.02

for i in range(110, 111):
    
    path = (path_ply+"fluid_pos_"+str(i)+".ply")
    plydata = PlyData.read(path)
    verts = plydata['vertex'].data
    xlist = verts['x']
    ylist = verts['y']
    colorr = verts['r']
    colorg = verts['g']
    colorb = verts['b']
    
    colors = np.column_stack((colorr, colorg, colorb))
    
    plt.figure()
    
    plt.xlim(lb[0], rt[0])
    plt.ylim(lb[1], rt[1])
    
    plt.xlabel('X-axis (m)')
    plt.ylabel('Y-axis (m)')
    
    plt.title('2D Position Plot with RGB Colors')
    
    plt.scatter(xlist, ylist, s=radius, c=colors/255, edgecolors='none') 
    plt.savefig(path_img+str(i)+'.png', dpi=1200)
    # plt.show()