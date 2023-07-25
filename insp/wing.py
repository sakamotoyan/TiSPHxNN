import numpy as np
import matplotlib.pyplot as plt

def camber_line(m, p, c, x):
    return np.where((x>=0)&(x<=(c*p)),
                    m * (x / np.power(p,2)) * (2.0 * p - (x / c)),
                    m * ((c - x) / np.power(1-p,2)) * (1.0 + (x / c) - 2.0 * p ))

def thickness(t, c, x):
    term1 =  0.2969 * np.sqrt(x / c)
    term2 = -0.1260 * (x / c)
    term3 = -0.3516 * np.power(x / c, 2)
    term4 =  0.2843 * np.power(x / c, 3)
    term5 = -0.1015 * np.power(x / c, 4)
    return t * c * (term1 + term2 + term3 + term4 + term5) / 0.2

# Airfoil parameters
m = 0.02  # Maximum camber
p = 0.4   # Location of maximum camber
t = 0.12  # Maximum thickness as a fraction of the chord
c = 1.0   # Chord length
x = np.linspace(0, c, 200)  # x coordinates

# Calculate camber line and thickness
yc = camber_line(m, p, c, x)
yt = thickness(t, c, x)

# Combine camber and thickness to form the airfoil surfaces
yu = yc + yt
yl = yc - yt

############### Method 1 Plot airfoil ###############
# plt.figure(figsize=(10, 5))
# plt.plot(x, yu, 'r', label='Upper Surface')
# plt.plot(x, yl, 'b', label='Lower Surface')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('NACA 2412 Airfoil')
# plt.legend()
# plt.grid(True)
# plt.axis('equal')  # Equal scaling
# plt.show()

############### Method 2 Plot airfoil with particles ###############
# Diameter of the particles
d = 0.02

# Create a grid
grid_x, grid_y = np.mgrid[0:c:d, -c:c:d]

# Create a mask for points within the airfoil
mask = (grid_y >= np.interp(grid_x, x, yl)) & (grid_y <= np.interp(grid_x, x, yu))

# Extract particle locations
particle_x = grid_x[mask]
particle_y = grid_y[mask]

print('particle_x', particle_x)

# Plot airfoil
plt.figure(figsize=(10, 5))
plt.plot(x, yu, 'r', label='Upper Surface')
plt.plot(x, yl, 'b', label='Lower Surface')
plt.scatter(particle_x, particle_y, s=10, color='green', label='Particles')
plt.xlabel('x')
plt.ylabel('y')
plt.title('NACA 2412 Airfoil with Particles')
plt.legend()
plt.grid(True)
plt.axis('equal')  # Equal scaling
plt.show()