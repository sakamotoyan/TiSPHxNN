import numpy as np
import os
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import scrolledtext
from tkinter import Toplevel, Label
from PIL import Image, ImageTk

data_path = '../output'
attr_name = 'bottleneck'

def inspect(path, attr_name):
    files = os.listdir(path)
    relevant_files = [f for f in files if f.startswith(attr_name) and f.endswith(".npy")]
    indices = [int(f.split('_')[-1].split('.')[0]) for f in relevant_files]
    indices.sort()
    start_number = min(indices) if indices else None
    end_number = max(indices) if indices else None
    return len(relevant_files)

def load_arrays(path, attr_name, n_frames):
    frames = []
    for i in range(n_frames):
        # Assuming the files are named 'frame0.npy', 'frame1.npy', ..., 'frame127.npy'
        filename = os.path.join(path, f'{attr_name}_{i}.npy')
        frame = np.load(filename)
        frames.append(frame)
    return frames

def mse(frame1, frame2):
    return np.mean((frame1 - frame2) ** 2)

def compute_similarity_matrix(frames):
    n_frames = len(frames)
    similarity_matrix = np.zeros((n_frames, n_frames))
    
    for i in range(n_frames):
        for j in range(n_frames):
            similarity_matrix[i, j] = mse(frames[i], frames[j])
            
    return similarity_matrix

def find_lowest_mse_pairs(similarity_matrix):
    pairs = []
    n = similarity_matrix.shape[0]  # Number of frames
    
    # Create a copy of the similarity matrix to modify
    adjusted_matrix = np.copy(similarity_matrix)
    
    # Set the diagonal (self-comparisons) to infinity to ignore them
    np.fill_diagonal(adjusted_matrix, np.inf)
    
    for i in range(n):
        # Find the index of the minimum MSE excluding itself
        min_index = np.argmin(adjusted_matrix[i])
        min_mse = adjusted_matrix[i, min_index]
        pairs.append((i, min_index, min_mse))
    
    return pairs

# Function to find n closest frames
def find_n_closest_frames(frame_number, n, exclude_local_frames, similarity_matrix):
    mse_values = np.copy(similarity_matrix[frame_number])
    
    # Set the MSE for the local range to infinity to exclude them
    start = max(0, frame_number - exclude_local_frames)
    end = min(len(mse_values), frame_number + exclude_local_frames + 1)
    mse_values[start:end] = np.inf
    
    # Find the indexes of the n smallest MSE values excluding the local range
    closest_frames = np.argsort(mse_values)[:n]
    return closest_frames, mse_values[closest_frames]

def visualize_similarity_matrix(matrix):
    plt.figure(figsize=(10, 8))
    plt.imshow(matrix, cmap='hot', interpolation='nearest')
    plt.title('Similarity Matrix based on MSE')
    plt.colorbar()
    plt.show()

# Tooltip class for displaying frame information on hover
class Tooltip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tipwindow = None

    def show_tip(self):
        "Display text in tooltip window"
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20
        self.tipwindow = Toplevel(self.widget)
        self.tipwindow.wm_overrideredirect(True)
        self.tipwindow.wm_geometry(f"+{x}+{y}")
        label = Label(self.tipwindow, text=self.text, justify=tk.LEFT, background="#ffffe0", relief=tk.SOLID, borderwidth=1, font=("tahoma", "8", "normal"))
        label.pack(ipadx=1)

    def hide_tip(self):
        if self.tipwindow:
            self.tipwindow.destroy()
        self.tipwindow = None

# Function to create a tooltip for a widget
def create_tooltip(widget, text):
    tooltip = Tooltip(widget, text)
    widget.bind('<Enter>', lambda event: tooltip.show_tip())
    widget.bind('<Leave>', lambda event: tooltip.hide_tip())

# Function to handle image click event
def on_image_click(frame_number, entry_frame_number, find_closest_frames_func):
    """
    Update the frame number input field and refresh the closest frames display.
    """
    def handler(event):
        entry_frame_number.delete(0, tk.END)  # Clear the current entry
        entry_frame_number.insert(0, str(frame_number))  # Update with the clicked frame number
        find_closest_frames_func()  # Trigger the refresh of the display
    return handler

# GUI Application with Image Display
class MSEGUIWithImages:
    def __init__(self, master, path, similarity_matrix):
        self.master = master
        master.title("Closest Frames Finder with Image Rows")

        self.path = path
        self.similarity_matrix = similarity_matrix

        # Frame number input
        self.label_frame_number = tk.Label(master, text="Frame Number:")
        self.label_frame_number.pack()
        
        self.entry_frame_number = tk.Entry(master)
        self.entry_frame_number.pack()

        # Number of closest frames input
        self.label_n_closest = tk.Label(master, text="Number of closest frames:")
        self.label_n_closest.pack()
        
        self.entry_n_closest = tk.Entry(master)
        self.entry_n_closest.pack()

        # Exclude local frames input
        self.label_exclude_local = tk.Label(master, text="Exclude local frames range:")
        self.label_exclude_local.pack()
        
        self.entry_exclude_local = tk.Entry(master)
        self.entry_exclude_local.pack()

        # Find button
        self.find_button = tk.Button(master, text="Find Closest Frames", command=self.find_closest_frames)
        self.find_button.pack()

        # Results text area
        self.results_text = scrolledtext.ScrolledText(master, height=10)
        self.results_text.pack()

        # Initialize an empty list for dynamically created row frames
        self.row_frames = []

    def find_closest_frames(self):
        frame_number = int(self.entry_frame_number.get())
        n_closest = int(self.entry_n_closest.get())
        exclude_local_frames = int(self.entry_exclude_local.get())
        
        closest_frames, mse_values = find_n_closest_frames(frame_number, n_closest, exclude_local_frames, self.similarity_matrix)
        
        # Clear previous results
        self.results_text.delete('1.0', tk.END)

        # Remove old row frames
        for frame in self.row_frames:
            frame.destroy()
        self.row_frames = []


        row_frame = tk.Frame(self.master)
        row_frame.pack()
        self.row_frames.append(row_frame)
        image_path = os.path.join(self.path, f'sci_output_vorticity_{frame_number}.png')  # Update 'path' to your actual images directory
        img = Image.open(image_path)
        img = img.resize((100, 100))  # Resize for display, adjust as needed
        img_tk = ImageTk.PhotoImage(img)
        label = tk.Label(row_frame, image=img_tk)
        label.image = img_tk  # Keep a reference
        label.pack(side=tk.LEFT)

        # Display new results and arrange images in rows
        max_per_row = 5  # Maximum number of images per row
        for i, (frame, mse) in enumerate(zip(closest_frames, mse_values)):
            row_index = i // max_per_row
            col_index = i % max_per_row
            
            # Create a new row frame if needed
            if col_index == 0:
                row_frame = tk.Frame(self.master)
                row_frame.pack()
                self.row_frames.append(row_frame)
            
            self.results_text.insert(tk.END, f"{i+1}. Frame {frame} with MSE: {mse:.4f}\n")
            
            # Load and display the image in the current row frame
            image_path = os.path.join(self.path, f'sci_output_vorticity_{frame}.png')  # Update 'path' to your actual images directory
            img = Image.open(image_path)
            img = img.resize((100, 100))  # Resize for display, adjust as needed
            img_tk = ImageTk.PhotoImage(img)

            
            
            # Create a label for the image in the current row frame and display it
            label = tk.Label(row_frame, image=img_tk)
            label.image = img_tk  # Keep a reference
            label.pack(side=tk.LEFT)

            # Bind the click event to the label
            label.bind('<Button-1>', on_image_click(frame, self.entry_frame_number, self.find_closest_frames))

            frame_info = f"Frame {frame}, MSE: {mse:.4f}"
            create_tooltip(label, frame_info)

num_files = inspect(data_path, attr_name)
arrays = load_arrays(data_path, attr_name, num_files)
similarity_matrix = compute_similarity_matrix(arrays)


# Main loop
root = tk.Tk()
gui = MSEGUIWithImages(root, data_path, similarity_matrix)
root.mainloop()
# image_path = os.path.join(self.path, f'sci_output_vorticity_{frame}.png')

# visualize_similarity_matrix(similarity_matrix)


