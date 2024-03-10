import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, colors

def plot_hsv_cone_with_cap(samples=360):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Cone base circle
    radius = 1
    theta = np.linspace(0, 2*np.pi, samples)
    x_base = radius * np.cos(theta)
    y_base = radius * np.sin(theta)
    z_base = np.zeros(samples)

    # Cone surface
    u = np.linspace(0, 2 * np.pi, samples)
    v = np.linspace(0, 1, samples)
    U, V = np.meshgrid(u, v)
    X = V * np.cos(U)
    Y = V * np.sin(U)
    Z = V  # Height of the cone

    # HSV to RGB for cone surface
    HSV_surface = np.dstack((U / (2 * np.pi), V, Z))
    RGB_surface = colors.hsv_to_rgb(HSV_surface)

    # Cap (top disk)
    x_cap = radius * V * np.cos(U)
    y_cap = radius * V * np.sin(U)
    z_cap = np.ones_like(x_cap)  # Cap at the top of the cone (Value = 1)

    # HSV to RGB for cap
    HSV_cap = np.dstack((U / (2 * np.pi), V, np.ones_like(Z)))
    RGB_cap = colors.hsv_to_rgb(HSV_cap)

    # Plotting
    ax.plot(x_base, y_base, z_base, color='k')  # Base circle
    ax.plot_surface(X, Y, Z, facecolors=RGB_surface, linewidth=0)  # Cone surface
    ax.plot_surface(x_cap, y_cap, z_cap, facecolors=RGB_cap, linewidth=0, alpha=0.6)  # Cap

    # Labels and title
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    ax.set_zlabel('Value')
    ax.set_title('HSV Color Cone with Cap')

    # Hide the axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    plt.show()

plot_hsv_cone_with_cap()