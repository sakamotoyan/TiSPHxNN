import numpy as np
import os
import matplotlib.pyplot as plt
import tkinter as ttk
from tkinter import scrolledtext
from tkinter import Toplevel, Label
from PIL import Image, ImageTk
from tkinter import font
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.gridspec import GridSpec
import math

from ctypes import windll
windll.shcore.SetProcessDpiAwareness(1)

lable_width = 25
status_width = 12
warp_length = status_width*10

class Frame_SimilaritySearchLoad:
    def __init__(self, root):
        self.flag_load = False
        self.flag_success = True

        self.num_files = None
        self.arr_bottleneck = None
        self.attrType_menu = ["Input Velocity", "Input Vorticity", "Output Velocity", "Output Vorticity"]

        self.root = root
        bold_font = font.Font(family="Helvetica", weight="bold")
        self.frame = ttk.LabelFrame(self.root, text="Step 1: Data Loading", font=bold_font)

        self.label_mainDir =                ttk.Label(self.frame, text="Main Directory:",        anchor='w', width=lable_width)
        self.label_bottleneck_name =        ttk.Label(self.frame, text="Bottleneck Name:",       anchor='w', width=lable_width)
        self.lable_attr_names =            [ttk.Label(self.frame, text=f"{attr} Name:", anchor='w', width=lable_width) for attr in self.attrType_menu]
        self.load_status =                  ttk.Label(self.frame, text="Load Status", width=status_width)
        self.lable_status_bottleneck =      ttk.Label(self.frame, text=" ", width=status_width, anchor='w', wraplength=warp_length)
        self.lable_status_input_velocity =  ttk.Label(self.frame, text=" ", width=status_width, anchor='w', wraplength=warp_length)
        self.lable_status_input_vorticity = ttk.Label(self.frame, text=" ", width=status_width, anchor='w', wraplength=warp_length)
        self.lable_status_output_velocity = ttk.Label(self.frame, text=" ", width=status_width, anchor='w', wraplength=warp_length)
        self.lable_status_output_vorticity =ttk.Label(self.frame, text=" ", width=status_width, anchor='w', wraplength=warp_length)
        self.lable_load =                   ttk.Label(self.frame, text="Load Status: ", width=lable_width, anchor='w',)
        self.entry_mainDir =                ttk.Entry(self.frame)
        self.entry_bottleneck_name =        ttk.Entry(self.frame)
        self.entry_input_velocity_name =    ttk.Entry(self.frame)
        self.entry_input_vorticity_name =   ttk.Entry(self.frame)
        self.entry_output_velocity_name =   ttk.Entry(self.frame)
        self.entry_output_vorticity_name =  ttk.Entry(self.frame)
        self.botton_load =                  ttk.Button(self.frame, text="Load", command=self.load)
        
        self.load_status.grid                   (row=0, column=2, sticky="w")
        self.label_mainDir.grid                 (row=1, column=0, sticky="w")
        self.label_bottleneck_name.grid         (row=2, column=0, sticky="w")
        for i in range(3, 7):
            self.lable_attr_names[i-3].grid     (row=i, column=0, sticky="w")
        self.entry_mainDir.grid                 (row=1, column=1, sticky="w")
        self.entry_bottleneck_name.grid         (row=2, column=1, sticky="w")
        self.entry_input_velocity_name.grid     (row=3, column=1, sticky="w")
        self.entry_input_vorticity_name.grid    (row=4, column=1, sticky="w")
        self.entry_output_velocity_name.grid    (row=5, column=1, sticky="w")
        self.entry_output_vorticity_name.grid   (row=6, column=1, sticky="w")
        self.lable_status_bottleneck.grid       (row=2, column=2, sticky="w")
        self.lable_status_input_velocity.grid   (row=3, column=2, sticky="w")
        self.lable_status_input_vorticity.grid  (row=4, column=2, sticky="w")
        self.lable_status_output_velocity.grid  (row=5, column=2, sticky="w")
        self.lable_status_output_vorticity.grid (row=6, column=2, sticky="w")
        self.botton_load.grid                   (row=7, column=0, sticky="w")
        self.lable_load.grid                    (row=7, column=1, sticky="w", columnspan=2)

        self.entry_mainDir.insert(0, "../output/")
        self.entry_bottleneck_name.insert(0, "bottleneck")
        self.entry_input_velocity_name.insert(0, "sci_input_velocity")
        self.entry_input_vorticity_name.insert(0, "sci_input_vorticity")
        self.entry_output_velocity_name.insert(0, "sci_output_velocity")
        self.entry_output_vorticity_name.insert(0, "sci_output_vorticity")

        # for i in range(8):
        #     self.frame.grid_rowconfigure(i, weight=1)
        # for i in range(3):
        #     self.frame.grid_columnconfigure(i, weight=1)

    def load(self):
        self.lable_load.config(text="Load Status: Loading...")
        self.main_path = os.path.join(self.entry_mainDir.get())
        self.attr_bottleneck = self.entry_bottleneck_name.get()
        self.attr_input_velocity = self.entry_input_velocity_name.get()
        self.attr_input_vorticity = self.entry_input_vorticity_name.get()
        self.attr_output_velocity = self.entry_output_velocity_name.get()
        self.attr_output_vorticity = self.entry_output_vorticity_name.get()
        self.attr_list = [self.attr_input_velocity, self.attr_input_vorticity, self.attr_output_velocity, self.attr_output_vorticity]

        self.lable_status_bottleneck.config         (text=self.inspect(self.main_path, self.attr_bottleneck))
        self.lable_status_input_velocity.config     (text=self.inspect(self.main_path, self.attr_input_velocity))
        self.lable_status_input_vorticity.config    (text=self.inspect(self.main_path, self.attr_input_vorticity))
        self.lable_status_output_velocity.config    (text=self.inspect(self.main_path, self.attr_output_velocity))
        self.lable_status_output_vorticity.config   (text=self.inspect(self.main_path, self.attr_output_vorticity))

        self.flag_load = True
    
        self.passed = self.flag_success and self.flag_load

        if self.passed:
            self.num_files = self.inspect_num(self.main_path, self.attr_bottleneck)
            self.arr_bottleneck = self.load_arrays(self.main_path, self.attr_bottleneck, self.num_files)
            self.lable_load.config(text="Load Status: Successfull")
        else:
            self.lable_load.config(text="Load Status: Failed")

    def inspect(self, path, attr_name, _=None):    
        files = os.listdir(path)
        relevant_files = [f for f in files if f.startswith(attr_name)]
        indices = [int(f.split('_')[-1].split('.')[0]) for f in relevant_files]
        indices.sort()
        start_number = min(indices) if indices else None
        end_number = max(indices) if indices else None
        if len(indices) == 0:
            return "No files found"
            self.flag_success = False
        missing_files = []
        for i in range(max(indices) + 1):
            if i not in indices:
                missing_files.append(f"{i}")
        # print(f"SeqData.inspect(): In {path}, found {len(relevant_files)} files for attribute {attr_name}, ranging [{start_number}, {end_number}].")
        # if len(missing_files) > 0:
        #     raise Exception(f"Missing files for attribute {attr_name}: {missing_files}")
        if len(missing_files) == 0:
            return f"[{start_number}, {end_number}]"
        else:
            return f"[{start_number}, {end_number}], Missing files: {missing_files}"
            self.flag_success = False

    def inspect_num(self, path, attr_name, _=None):    
        files = os.listdir(path)
        relevant_files = [f for f in files if f.startswith(attr_name)]
        indices = [int(f.split('_')[-1].split('.')[0]) for f in relevant_files]
        indices.sort()
        start_number = min(indices) if indices else None
        end_number = max(indices) if indices else None
        return len(relevant_files)
    
    def load_arrays(self, path, attr_name, n_frames):
        frames = []
        for i in range(n_frames):
            # Assuming the files are named 'frame0.npy', 'frame1.npy', ..., 'frame127.npy'
            filename = os.path.join(path, f'{attr_name}_{i}.npy')
            frame = np.load(filename)
            frames.append(frame)
        return frames

class Frame_SimilaritySearchCompute:
    def __init__(self, root, frame_SimilaritySearchLoad:Frame_SimilaritySearchLoad):
        self.root = root
        self.frame_SimilaritySearchLoad = frame_SimilaritySearchLoad
        self.similarity_matrix_list = None
        self.similarity_matrix = None
        self.method_menu = ["MSE", "Cosine_distance"]

        bold_font =                     font.Font(family="Helvetica", weight="bold")
        self.frame =                    ttk.LabelFrame(self.root, text="Step 2: Compute Similarity Matrix", font=bold_font)

        self.button_compute_all =       ttk.Button(self.frame, text="Compute All", command=self.compute_all_similarity_matrix)
        self.label_compute_status =     ttk.Label(self.frame, text="Compute Status:", width=lable_width, anchor='w')
        self.label_measurement_method = ttk.Label(self.frame, text="Measurement Method:", anchor='w', width=lable_width)
        self.measurement_method_tag =   ttk.StringVar(self.frame)
        self.measurement_method_tag.set("None") 
        self.measurement_dropdown =     ttk.OptionMenu(self.frame, self.measurement_method_tag, *self.method_menu, command=self.draw_similarity_matrix)
        
        self.figure = Figure(figsize=(4, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, self.frame)
        self.figure_2 = Figure(figsize=(4, 1.5), dpi=100)
        self.canvas_2 = FigureCanvasTkAgg(self.figure_2, self.frame)

        self.button_compute_all.grid(           row=0, column=0, sticky="w")
        self.label_compute_status.grid(         row=0, column=1, sticky="w")
        self.label_measurement_method.grid(     row=1, column=0, sticky="w")
        self.measurement_dropdown.grid(         row=1, column=1, sticky="we")
        self.canvas.get_tk_widget().grid(       row=2, column=0, columnspan=2, sticky="nsew")
        self.canvas_2.get_tk_widget().grid(     row=3, column=0, columnspan=2, sticky="nsew")

        self.frame.grid_rowconfigure(2, weight=1)
        self.frame.grid_rowconfigure(3, weight=1)
        
        
    def mse(self, frame1, frame2):
        return np.mean((frame1 - frame2) ** 2)

    def cosine_distance(self, frame1, frame2):
        # Normalize the frames to unit vectors
        norm1 = np.linalg.norm(frame1)
        norm2 = np.linalg.norm(frame2)
        frame1_unit = frame1 / (norm1 if norm1 > 0 else 1)
        frame2_unit = frame2 / (norm2 if norm2 > 0 else 1)
        
        # Compute cosine similarity and then return cosine distance
        cosine_similarity = np.dot(frame1_unit, frame2_unit)
        cosine_distance = 1 - cosine_similarity
        return cosine_distance
    
    def compute_all_similarity_matrix(self, _=None):
        self.similarity_matrix_list = []
        for method in self.method_menu:
            self.similarity_matrix_list.append(self.compute_similarity_matrix(method))
        self.label_compute_status.config(text="Compute Status: Successfull")
    
    def compute_similarity_matrix(self, method):
        n_frames = self.frame_SimilaritySearchLoad.num_files
        similarity_matrix = np.zeros((n_frames, n_frames))
        for i in range(n_frames):
            for j in range(n_frames):
                frame1 = self.frame_SimilaritySearchLoad.arr_bottleneck[i]
                frame2 = self.frame_SimilaritySearchLoad.arr_bottleneck[j]
                if method == self.method_menu[0]:
                    similarity_matrix[i, j] = self.mse(frame1, frame2)
                elif method == self.method_menu[1]:
                    similarity_matrix[i, j] = self.cosine_distance(frame1.flatten(), frame2.flatten())
        return similarity_matrix
    
    def draw_similarity_matrix(self, selected_frame=None, closest_frames=None, _=None):
        self.similarity_matrix = self.similarity_matrix_list[self.method_menu.index(self.measurement_method_tag.get())]
        if self.similarity_matrix is None or self.frame_SimilaritySearchLoad.arr_bottleneck is None:
            return
        self.figure.clear()
        self.figure_2.clear()

        gridsize = (5, 1)
        # Create two subplots
        self.plot = self.figure.add_subplot(1,1,1)  # 1 rows, 1 column, 1st subplot
        self.plot_2 = self.figure_2.add_subplot(1,1,1)  # 2 rows, 1 column, 2nd subplot

        # Plot the similarity matrix
        im = self.plot.imshow(self.similarity_matrix, cmap='viridis')
        self.plot.set_xlabel('Frames')
        self.plot.set_ylabel('Frames')
        self.plot.set_title('Similarity Matrix')

        # self.cbar = self.figure.colorbar(im, ax=self.plot)

        if selected_frame is not None:
            self.plot.scatter(selected_frame, selected_frame, color='red', s=10, edgecolor='black')

        # Mark the positions of the closest frames
        if closest_frames is not None:
            for frame in closest_frames:
                self.plot.scatter(frame, selected_frame, color='white', s=10, edgecolor='black')
                self.plot.scatter(selected_frame, frame, color='white', s=10, edgecolor='black')

        # Plot the distance line
        if selected_frame is not None and closest_frames is not None:
            distances = self.similarity_matrix[selected_frame, :]
            self.plot_2.plot(distances, '-o', markersize=1, zorder=1)
            self.plot_2.scatter(selected_frame, distances[selected_frame], color='red', s=10, edgecolor='black', zorder=2)
            self.plot_2.set_xlabel('Frame')
            self.plot_2.set_ylabel('Distance')
            self.plot_2.set_title('Distances from Selected Frame')
            

        # Join the x-axes of the two subplots
        self.plot.get_shared_x_axes().join(self.plot, self.plot_2)

        # Draw the canvas
        self.canvas.draw()
        self.canvas_2.draw()



class Frame_SimilaritySearchNClosest:
    def __init__(self, root, frame_SimilaritySearchLoad:Frame_SimilaritySearchLoad, frame_SimilaritySearchCompute:Frame_SimilaritySearchCompute):
        self.root = root
        self.frame_SimilaritySearchLoad = frame_SimilaritySearchLoad
        self.frame_SimilaritySearchCompute = frame_SimilaritySearchCompute
        self.attrType_menu = ["Input Velocity", "Input Vorticity", "Output Velocity", "Output Vorticity"]
        self.selected_imgs = []
        self.row_frames = []
        self.selected_img_size = 256
        self.close_img_size = math.floor(256/5*4)
        self.img_per_row = 5

        bold_font = font.Font(family="Helvetica", weight="bold")
        self.frame = ttk.LabelFrame(self.root, text="Step 3: Find Similar Frames", font=bold_font)

        self.frame_botton =                 ttk.Frame(self.frame)
        self.frame_closest_imgs =           ttk.LabelFrame(self.frame, text="Closest Frames")
        self.label_frame_number =           ttk.Label(self.frame, text="Selected Frame:",            anchor='w', width=lable_width)
        self.label_n_closest =              ttk.Label(self.frame, text="Number of closest frames:",  anchor='w', width=lable_width)
        self.label_exclude_local =          ttk.Label(self.frame, text="Exclude local frames range:",anchor='w', width=lable_width)
        self.entry_frame_number =           ttk.Entry(self.frame)
        self.entry_n_closest =              ttk.Entry(self.frame)
        self.entry_exclude_local =          ttk.Entry(self.frame)
        self.botton_find_n_closest =        ttk.Button(self.frame_botton, text="Find", command=self.find)
        self.botton_find_n_closes_next =    ttk.Button(self.frame_botton, text="Find Next", command=self.find_next)
        self.botton_find_n_closes_prev =    ttk.Button(self.frame_botton, text="Find Previous", command=self.find_prev)
        self.display_tag =                  ttk.StringVar(self.frame_botton)
        self.display_dropdown =             ttk.OptionMenu(self.frame_botton, self.display_tag, *self.attrType_menu)
        self.display_tag.set(self.attrType_menu[0])
        self.canvas_selected =              ttk.Canvas(self.frame, height=self.selected_img_size, width=self.selected_img_size*4)
        self.canvas_closest_imgs =          ttk.Canvas(self.frame_closest_imgs)
        self.scrollbar =                    ttk.Scrollbar(self.frame_closest_imgs, orient="vertical", command=self.canvas_closest_imgs.yview)
        self.scrollable_frame =             ttk.Frame(self.canvas_closest_imgs)

        self.label_frame_number.grid(       row=2, column=0, sticky="w")
        self.label_n_closest.grid(          row=3, column=0, sticky="w")
        self.label_exclude_local.grid(      row=4, column=0, sticky="w")
        self.entry_frame_number.grid(       row=2, column=1, sticky="w")
        self.entry_n_closest.grid(          row=3, column=1, sticky="w")
        self.entry_exclude_local.grid(      row=4, column=1, sticky="w")
        self.frame_botton.grid(             row=5, column=0, columnspan=2, sticky="w")
        self.canvas_selected.grid(          row=6, column=0, columnspan=2, sticky="nsew")
        self.frame_closest_imgs.grid(       row=7, column=0, columnspan=2, sticky="nsew")
        self.botton_find_n_closes_prev.grid(row=0, column=0, sticky="w")
        self.botton_find_n_closest.grid(    row=0, column=1, sticky="w")
        self.botton_find_n_closes_next.grid(row=0, column=2, sticky="w")
        self.display_dropdown.grid(         row=0, column=3, sticky="we")
        self.frame.grid_rowconfigure(7, weight=1)
        self.frame.grid_columnconfigure(0, weight=1)

        self.entry_frame_number.insert(0, "50")
        self.entry_n_closest.insert(0, "30")
        self.entry_exclude_local.insert(0, "5")

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas_closest_imgs.configure(
                scrollregion=self.canvas_closest_imgs.bbox("all")
            )
        )
        self.canvas_closest_imgs.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas_closest_imgs.configure(yscrollcommand=self.scrollbar.set)
        self.canvas_closest_imgs.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.canvas_closest_imgs.bind_all("<MouseWheel>", self.on_mousewheel)  # For Windows
        self.canvas_closest_imgs.bind_all("<Button-4>", self.on_mousewheel)  # For Linux
        self.canvas_closest_imgs.bind_all("<Button-5>", self.on_mousewheel)  # For Linux

    def find(self):
        closest_frames, distances = self.find_n_closest_frames()
        self.display_closest_frames(closest_frames, distances)
        self.frame_SimilaritySearchCompute.draw_similarity_matrix(int(self.entry_frame_number.get()), closest_frames)
    
    def find_next(self):
        next_number = int(self.entry_frame_number.get()) + 1
        self.entry_frame_number.delete(0, ttk.END)
        self.entry_frame_number.insert(0, str(next_number))
        self.find()

    def find_prev(self):
        prev_number = int(self.entry_frame_number.get()) - 1
        self.entry_frame_number.delete(0, ttk.END)
        self.entry_frame_number.insert(0, str(prev_number))
        self.find()

    def find_n_closest_frames(self):
        frame_number = int(self.entry_frame_number.get())
        n = int(self.entry_n_closest.get())
        exclude_local_frames = int(self.entry_exclude_local.get())

        distance_values = np.copy(self.frame_SimilaritySearchCompute.similarity_matrix[frame_number])
        
        # Set the MSE for the local range to infinity to exclude them
        start = max(0, frame_number - exclude_local_frames)
        end = min(len(distance_values), frame_number + exclude_local_frames + 1)
        distance_values[start:end] = np.inf
        
        # Find the indexes of the n smallest MSE values excluding the local range
        closest_frames = np.argsort(distance_values)[:n]
        return closest_frames, distance_values[closest_frames]
    
    def display_closest_frames(self, closest_frames, distance_values):
        # Display the selected frame
        selected_frame_number = int(self.entry_frame_number.get())
        selected_img_path_list = [os.path.join(self.frame_SimilaritySearchLoad.main_path, f"{attr_name}_{selected_frame_number}.png") for attr_name in self.frame_SimilaritySearchLoad.attr_list]
        img_list = [Image.open(path).resize((self.selected_img_size, self.selected_img_size), Image.ANTIALIAS) for path in selected_img_path_list]
        self.selected_imgs.clear()
        for img in img_list:
            img_tk = ImageTk.PhotoImage(img)
            self.selected_imgs.append(img_tk)
        self.canvas_selected.delete(ttk.ALL)
        for i in range(len(img_list)):
            self.canvas_selected.create_image(i*self.selected_img_size, 0, anchor='nw', image=self.selected_imgs[i])

        # Display the closest frames
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        for i, (frame, distance) in enumerate(zip(closest_frames, distance_values)):
            col_index = i % self.img_per_row
            if col_index == 0:
                row_frame = ttk.Frame(self.scrollable_frame)
                row_frame.pack()
                self.row_frames.append(row_frame)
            image_path = os.path.join(self.frame_SimilaritySearchLoad.main_path, f"{self.frame_SimilaritySearchLoad.attr_list[self.attrType_menu.index(self.display_tag.get())]}_{frame}.png")
            img = Image.open(image_path)
            img = img.resize((self.close_img_size, self.close_img_size))  # Resize for display
            img_tk = ImageTk.PhotoImage(img)
            label = ttk.Label(row_frame, image=img_tk)
            label.image = img_tk
            label.pack(side=ttk.LEFT)
            label.bind('<Button-1>', self.click_close_image_handler(frame))
            # frame_info = f"Frame {frame}, Distance: {distance:.4f}"
            frame_info = f"Frame {frame}, Distance: {distance:.4f}"
            self.Tooltip(label, frame_info).create()
        
    def click_close_image_handler(self, frame_number):
        def handler(event):
            self.entry_frame_number.delete(0, ttk.END)
            self.entry_frame_number.insert(0, str(frame_number))
            closest_frames, distances = self.find_n_closest_frames()
            self.display_closest_frames(closest_frames, distances)
            self.frame_SimilaritySearchCompute.draw_similarity_matrix(int(self.entry_frame_number.get()), closest_frames)
        return handler
    
    def on_mousewheel(self, event):
        """Handle mouse wheel scroll for different platforms."""
        if event.num == 4 or event.delta > 0:  # Scroll up
            self.canvas_closest_imgs.yview_scroll(-1, "units")
        elif event.num == 5 or event.delta < 0:  # Scroll down
            self.canvas_closest_imgs.yview_scroll(1, "units")
    
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
            label = Label(self.tipwindow, text=self.text, justify=ttk.LEFT, background="#ffffe0", relief=ttk.SOLID, borderwidth=1, font=("tahoma", "8", "normal"))
            label.pack(ipadx=1)

        def hide_tip(self):
            if self.tipwindow:
                self.tipwindow.destroy()
            self.tipwindow = None
        
        def create(self):
            self.widget.bind('<Enter>', lambda event: self.show_tip())
            self.widget.bind('<Leave>', lambda event: self.hide_tip())

class Frame_SimilaritySearchBottleneckCustomize:
    def __init__(self, root, 
                 frame_SimilaritySearchLoad:Frame_SimilaritySearchLoad, 
                 frame_SimilaritySearchCompute:Frame_SimilaritySearchCompute,
                 frame_SimilaritySearchNClosest:Frame_SimilaritySearchNClosest):
        self.root = root
        self.frame_SimilaritySearchLoad = frame_SimilaritySearchLoad
        self.frame_SimilaritySearchCompute = frame_SimilaritySearchCompute
        self.frame_SimilaritySearchNClosest = frame_SimilaritySearchNClosest
        
        bold_font = font.Font(family="Helvetica", weight="bold")
        self.frame = ttk.LabelFrame(self.root, text="Step 4: Customize Bottleneck", font=bold_font)
        self.feature_vector_size = 0
        self.feature_vectors_per_row = 5

        self.frame_buttons =            ttk.Frame(self.frame)
        self.frame_feature_vectors =    ttk.LabelFrame(self.frame, text="Feature Vectors")
        self.button_refresh =           ttk.Button(self.frame_buttons, text="Refresh", command=self.refresh)
        self.button_compute =           ttk.Button(self.frame_buttons, text="Compute", command=self.compute)

        self.frame_buttons.grid(         row=0, column=0, sticky="w")
        self.frame_feature_vectors.grid( row=1, column=0, sticky="nsew")
        self.button_refresh.grid(        row=0, column=0, sticky="w")
        
    def refresh(self, _=None):
        row_frame_list = []
        for widget in self.frame_feature_vectors.winfo_children():
            widget.destroy()
        self.feature_vector_size = self.frame_SimilaritySearchLoad.arr_bottleneck[0].flatten().shape[0]
        self.activate_mask = np.zeros(self.feature_vector_size, dtype=bool)

        for i in range(self.feature_vector_size):
            col_index = i % self.feature_vectors_per_row
            if col_index == 0:
                row_frame = ttk.Frame(self.frame_feature_vectors)
                row_frame.grid(row=i // self.feature_vectors_per_row, column=0, sticky="w")
                row_frame_list.append(row_frame)
            check_button = ttk.Checkbutton(row_frame, text=f"Feature {i}", width=8, command=self.ticked(i))
            check_button.grid(row=0, column=col_index, sticky="w")

    def ticked(self, i, _=None):
        def handler():
            self.activate_mask[i] = not self.activate_mask[i]
            print(f"Feature {i} is {'activated' if self.activate_mask[i] else 'deactivated'}")
        return handler
    
    def 