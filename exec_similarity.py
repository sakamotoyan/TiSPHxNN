import numpy as np
import os
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import scrolledtext
from tkinter import Toplevel, Label
from PIL import Image, ImageTk

# data_path = '../output'
# attr_name = 'bottleneck'

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

def cosine_distance(frame1, frame2):
    # Normalize the frames to unit vectors
    norm1 = np.linalg.norm(frame1)
    norm2 = np.linalg.norm(frame2)
    frame1_unit = frame1 / (norm1 if norm1 > 0 else 1)
    frame2_unit = frame2 / (norm2 if norm2 > 0 else 1)
    
    # Compute cosine similarity and then return cosine distance
    cosine_similarity = np.dot(frame1_unit, frame2_unit)
    cosine_distance = 1 - cosine_similarity
    return cosine_distance

def compute_similarity_matrix(frames, mesurement):
    n_frames = len(frames)
    similarity_matrix = np.zeros((n_frames, n_frames))
    if mesurement == 'mse':
        for i in range(n_frames):
            for j in range(n_frames):
                similarity_matrix[i, j] = mse(frames[i], frames[j])
    elif mesurement == 'cosine_distance':
        for i in range(n_frames):
            for j in range(n_frames):
                similarity_matrix[i, j] = cosine_distance(frames[i].flatten(), frames[j].flatten())
            
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
    
    def create(self):
        self.widget.bind('<Enter>', lambda event: self.show_tip())
        self.widget.bind('<Leave>', lambda event: self.hide_tip())


# GUI Application with Image Display
class MSEGUIWithImages:
    def __init__(self, master, path, attr_name):
        self.master = master
        master.title("Closest Frames Finder with Image Rows")

        self.path = path
        self.attr_name = attr_name

        # Frame number input
        self.label_frame_number = tk.Label(master, text="Frame Number:")
        self.label_frame_number.pack()
        
        self.entry_frame_number = tk.Entry(master)
        self.entry_frame_number.pack()
        self.entry_frame_number.insert(0, "50")

        # Number of closest frames input
        self.label_n_closest = tk.Label(master, text="Number of closest frames:")
        self.label_n_closest.pack()
        
        self.entry_n_closest = tk.Entry(master)
        self.entry_n_closest.pack()
        self.entry_n_closest.insert(0, "30")

        # Exclude local frames input
        self.label_exclude_local = tk.Label(master, text="Exclude local frames range:")
        self.label_exclude_local.pack()
        
        self.entry_exclude_local = tk.Entry(master)
        self.entry_exclude_local.pack()
        self.entry_exclude_local.insert(0, "5")

        # Measurement method dropdown
        self.label_measurement_method = tk.Label(master, text="Measurement Method:")
        self.label_measurement_method.pack()
        self.measurement_method_tag = tk.StringVar(master)
        self.measurement_method_tag.set("MSE")  # default value
        self.measurement_dropdown = tk.OptionMenu(master, self.measurement_method_tag, "MSE", "Cosine_distance", command=self.act_on_change_measurement)
        self.measurement_dropdown.pack()

        # similarity matrix visualization button
        self.visualize_button = tk.Button(master, text="Visualize Similarity Matrix", command=self.act_on_visualize)
        self.visualize_button.pack()

        # Image source dropdown
        self.label_image_source = tk.Label(master, text="Image Source:")
        self.label_image_source.pack()

        self.image_source_tag = tk.StringVar(master)
        self.image_source_tag.set("sci_output_vorticity")  # default value
        self.image_source_dropdown = tk.OptionMenu(master, self.image_source_tag, "sci_output_vorticity", "sci_input_vorticity", "sci_input_velocity", "sci_output_velocity",
                                                   command=self.act_on_change_source)
        self.image_source_dropdown.pack()

        # Find button
        self.label_measurement_method = tk.Label(master, text=" ")
        self.label_measurement_method.pack()
        self.find_button = tk.Button(master, text="Find Closest Frames", command=self.act_on_find)
        self.find_button.pack()

        # Create a canvas and a scrollbar
        self.canvas = tk.Canvas(master)
        self.scrollbar = tk.Scrollbar(master, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas)
        # Configure the canvas to be scrollable
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        # Pack the canvas and scrollbar into the GUI
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Initialize an empty list for dynamically created row frames
        self.row_frames = []

        self.load_data()
        self.compute_similarity()
        self.compute_closest_frames()
        self.show_imgs()

    def load_data(self, _=None):
        self.frame_number         = int(self.entry_frame_number.get())
        self.n_closest            = int(self.entry_n_closest.get())
        self.exclude_local_frames = int(self.entry_exclude_local.get())
        self.image_source         = self.image_source_tag.get()
        self.measurement_method   = self.measurement_method_tag.get().lower().replace(" ", "_")
    
    def compute_similarity(self, _=None):
        # Load frames and compute similarity matrix based on the selected method
        self.num_files            = inspect(self.path, self.attr_name)
        self.arrays               = load_arrays(self.path, self.attr_name, self.num_files)
        self.similarity_matrix    = compute_similarity_matrix(self.arrays, self.measurement_method)
    
    def compute_closest_frames(self, _=None):
        self.closest_frames, self.mse_values \
                                  = find_n_closest_frames(self.frame_number, self.n_closest, self.exclude_local_frames, self.similarity_matrix)

        
    def show_imgs(self, _=None):
        source = self.image_source

        # Clear previous images and rows from the scrollable frame
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        # Reset the list that tracks row frames
        self.row_frames = []

        # Create the first row for the selected frame
        selected_row_frame = tk.Frame(self.scrollable_frame)
        selected_row_frame.pack()
        self.row_frames.append(selected_row_frame)

        # Load and display the selected frame
        selected_image_path = os.path.join(self.path, f'{source}_{self.frame_number}.png')
        selected_img = Image.open(selected_image_path)
        selected_img = selected_img.resize((100, 100))  # Resize for display
        selected_img_tk = ImageTk.PhotoImage(selected_img)

        selected_label = tk.Label(selected_row_frame, image=selected_img_tk)
        selected_label.image = selected_img_tk  # Keep a reference
        selected_label.pack(side=tk.LEFT)

        selected_frame_info = f"Selected Frame {self.frame_number}"
        Tooltip(selected_label, selected_frame_info).create()

        # Initialize variables for creating new rows
        row_frame = None
        max_per_row = 10  # Maximum number of images per row

        # Iterate through each closest frame to display
        for i, (frame, mse) in enumerate(zip(self.closest_frames, self.mse_values)):
            col_index = i % max_per_row

            # Create a new row frame if needed
            if col_index == 0:
                row_frame = tk.Frame(self.scrollable_frame)
                row_frame.pack()
                self.row_frames.append(row_frame)

            # Construct the image path and load the image
            image_path = os.path.join(self.path, f'{source}_{frame}.png')
            img = Image.open(image_path)
            img = img.resize((100, 100))  # Resize for display
            img_tk = ImageTk.PhotoImage(img)

            # Create a label for the image in the current row frame and display it
            label = tk.Label(row_frame, image=img_tk)
            label.image = img_tk  # Keep a reference to avoid garbage collection
            label.pack(side=tk.LEFT)

            # Bind the click event to the label to refresh on click
            label.bind('<Button-1>', self.click_image_handler(frame))

            # Create a tooltip for the image to show frame info
            frame_info = f"Frame {frame}, Distance: {mse:.4f}"
            Tooltip(label, frame_info).create()

    def show_similarity_matrix(self, _=None):
        plt.figure(figsize=(10, 8))
        plt.imshow(self.similarity_matrix, cmap='hot', interpolation='nearest')
        plt.title('Similarity Matrix based on MSE')
        plt.colorbar()
        plt.show()

    def act_on_change_source(self, _=None):
        self.load_data()
        self.show_imgs()

    def act_on_find(self, _=None):
        self.load_data()
        self.compute_closest_frames()
        self.show_imgs()
    
    def act_on_change_measurement(self, _=None):
        self.load_data()
        self.compute_similarity()
        self.compute_closest_frames()
        self.show_imgs()

    def act_on_visualize(self, _=None):
        self.show_similarity_matrix()
    
    def click_image_handler(self, frame_number):
        def handler(event):
            self.entry_frame_number.delete(0, tk.END)
            self.entry_frame_number.insert(0, str(frame_number))
            self.load_data()
            self.compute_closest_frames()
            self.show_imgs()
        return handler


data_path = '../output'
attr_name = 'bottleneck'

# Main loop
root = tk.Tk()
gui = MSEGUIWithImages(root, data_path, attr_name)
root.mainloop()


