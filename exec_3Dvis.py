import numpy as np
import pyvista as pv

input_path = '../organized_output'

# # Create a PyVista UniformGrid object
# grid_density = pv.ImageData()

# density_field = np.load('../organized_output/density_30.npz')['arr_0']

# # Set the dimensions, spacing, and origin of the grid
# grid_density.dimensions = np.array(density_field.shape) + 1
# grid_density.spacing = (1, 1, 1)  # Assuming the spacing between your points is 1 in all directions
# grid_density.origin = (0, 0, 0)  # Assuming the origin is at (0, 0, 0)

# # Assign the density field to the grid
# grid_density.cell_data["values"] = density_field.flatten(order="F")  # Flatten the array in Fortran order

# # Create the volume plot
# plotter_density = pv.Plotter()
# plotter_density.add_volume(grid_density, scalars="values")

# # Display the plot
# plotter_density.show()


# grid_velocity = pv.ImageData()

# velocity_field = np.load('../organized_output/velocity_30.npz')['arr_0']

# # Set the dimensions, spacing, and origin of the grid
# grid_velocity.dimensions = np.array(velocity_field.shape[:-1]) + 1
# grid_velocity.spacing = (1, 1, 1)  # Assuming the spacing between your points is 1 in all directions
# grid_velocity.origin = (0, 0, 0)  # Assuming the origin is at (0, 0, 0)

# # Assign the density field to the grid
# velocity_vectors = velocity_field.reshape(-1, 3, order="F")
# grid_velocity.cell_data["velocity"] = velocity_vectors
# glyphs = grid_velocity.glyph(orient="velocity", scale="velocity", factor=0.1)

# # Create the volume plot
# plotter_velocity = pv.Plotter(off_screen=True)
# plotter_velocity.add_mesh(glyphs, color="black")

# # Display the plot
# plotter_velocity.show()

def update_volume(frame, plotter, meshes):
    plotter.clear()
    plotter.add_volume(meshes[frame], scalars="values")

plotter_density = pv.Plotter(off_screen=True)
plotter_density.open_gif("animation1.gif")
for i in range(0, 1):
    grid_density = pv.ImageData()

    # field = np.load(f'../organized_output/density_{i}.npz')['arr_0']
    field = np.load(f'../organized_output/velocity_{i}.npz')['arr_0']
    field = np.sqrt(field[...,0]**2+field[...,1]**2+field[...,2]**2)

    grid_density.dimensions = np.array(field.shape) + 1
    grid_density.spacing = (1, 1, 1)  # Assuming the spacing between your points is 1 in all directions
    grid_density.origin = (0, 0, 0)  # Assuming the origin is at (0, 0, 0)
    grid_density.cell_data["values"] = field.flatten(order="F")  # Flatten the array in Fortran order
    plotter_density.clear()
    plotter_density.add_volume(grid_density, scalars="values")
    outline = grid_density.outline_corners()
    plotter_density.add_mesh(outline, color="black", show_edges=True)
    plotter_density.camera_position = 'zy'
    plotter_density.camera.elevation = 30
    plotter_density.write_frame()  

plotter_density.close()

