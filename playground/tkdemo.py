import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Step 1: Create a figure and add a subplot with labels
fig = Figure(figsize=(5, 4), dpi=100)
plot = fig.add_subplot(1, 1, 1)
plot.set_title('Sample Plot')
plot.set_xlabel('X axis')
plot.set_ylabel('Y axis')

# Just a simple plot for demonstration
x = [1, 2, 3, 4, 5]
y = [10, 1, 20, 5, 3]
plot.plot(x, y)

# Step 2: Create a Tkinter window
root = tk.Tk()
root.title("Matplotlib in Tkinter")

# Step 3: Create a FigureCanvasTkAgg object
canvas = FigureCanvasTkAgg(fig, master=root)  # A tk.DrawingArea.

# Step 4: Draw the canvas
canvas.draw()

# Step 5: Pack the canvas widget into the Tkinter window
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# Tkinter event loop
root.mainloop()
