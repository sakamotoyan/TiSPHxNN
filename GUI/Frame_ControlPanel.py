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
import torch

from ctypes import windll
windll.shcore.SetProcessDpiAwareness(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lable_width = 25
status_width = 12
warp_length = status_width*10

class Frame_ControlPanel:
    def __init__(self, root):
        self.root = root
        self.frame_control = ttk.LabelFrame(self.root, text="Control Panel", font=font.Font(family="Helvetica", weight="bold"))
        self.subframe_load = ttk.LabelFrame(self.frame_control, text="Data Loading")
        self.subframe_similarity = ttk.LabelFrame(self.frame_control, text="Compute Similarity Matrix")
        self.subframe_find_n_closest = ttk.LabelFrame(self.frame_control, text="Find N Closest Frames")
        self.subframe_customize_bottleneck = ttk.LabelFrame(self.frame_control, text="Customize Bottleneck")

        self.frame_datavis = ttk.LabelFrame(self.root, text="Data Visualization", font=font.Font(family="Helvetica", weight="bold"))
        self.subframe_similarity_matrix = ttk.LabelFrame(self.frame_datavis, text="Similarity Matrix")
        self.subframe_distance_plot = ttk.LabelFrame(self.frame_datavis, text="Distance Plot")

        self.selected_img_size = 256
        self.close_img_size = math.floor(256/5*4)
        self.img_per_row = 5
        self.feature_vector_size = 0
        self.feature_vectors_per_row = 4

        self.num_files = None
        self.arr_bottleneck = None
        self.similarity_matrix = None
        self.customized_similarity_matrix = None
        self.similarity_matrix_list = []
        self.customized_similarity_matrix_list = []
        self.attrType_menu = ["Input Velocity", "Input Vorticity", "Output Velocity", "Output Vorticity"]
        self.method_menu = ["MSE", "Cosine_distance"]

        self.widgets_init_load(self.subframe_load)
        self.widgets_init_similarity(self.subframe_similarity)
        self.widgets_init_find_n_closest(self.subframe_find_n_closest)
        self.widgets_init_customize_bottleneck(self.subframe_customize_bottleneck)
        self.subframe_load.grid(row=0, column=0, sticky="nsew")
        ttk.Label(self.frame_control, text=" ", width=lable_width, anchor='w').grid(row=1, column=0, sticky="w")
        self.subframe_similarity.grid(row=2, column=0, sticky="nsew")
        ttk.Label(self.frame_control, text=" ", width=lable_width, anchor='w').grid(row=3, column=0, sticky="w")
        self.subframe_find_n_closest.grid(row=4, column=0, sticky="nsew")
        ttk.Label(self.frame_control, text=" ", width=lable_width, anchor='w').grid(row=5, column=0, sticky="w")
        self.subframe_customize_bottleneck.grid(row=6, column=0, sticky="nsew")
        
        self.widgets_init_similarity_matrix(self.subframe_similarity_matrix)
        self.widgets_init_distance_plot(self.subframe_distance_plot)
        self.subframe_similarity_matrix.grid(row=0, column=0, sticky="nsew")
        self.subframe_distance_plot.grid(row=1, column=0, sticky="nsew")

    def widgets_init_load(self, frame):
        self.flag_load = False
        self.flag_success = True

        self.num_files = None
        self.arr_bottleneck = None
        self.attrType_menu = ["Input Velocity", "Input Vorticity", "Output Velocity", "Output Vorticity"]

        self.label_mainDir =                    ttk.Label(frame, text="Main Directory:",        anchor='w', width=lable_width)
        self.label_bottleneck_name =            ttk.Label(frame, text="Bottleneck Name:",       anchor='w', width=lable_width)
        self.lable_attr_names =                [ttk.Label(frame, text=f"{attr} Name:", anchor='w', width=lable_width) for attr in self.attrType_menu]
        self.label_load_status =                ttk.Label(frame, text="Statistics", width=status_width)
        self.lable_status_bottleneck =          ttk.Label(frame, text=" ", width=status_width, anchor='w', wraplength=warp_length)
        self.lable_status_input_velocity =      ttk.Label(frame, text=" ", width=status_width, anchor='w', wraplength=warp_length)
        self.lable_status_input_vorticity =     ttk.Label(frame, text=" ", width=status_width, anchor='w', wraplength=warp_length)
        self.lable_status_output_velocity =     ttk.Label(frame, text=" ", width=status_width, anchor='w', wraplength=warp_length)
        self.lable_status_output_vorticity =    ttk.Label(frame, text=" ", width=status_width, anchor='w', wraplength=warp_length)
        self.lable_load =                       ttk.Label(frame, text="Load Status: ", width=lable_width, anchor='w',)
        self.entry_mainDir =                    ttk.Entry(frame)
        self.entry_bottleneck_name =            ttk.Entry(frame)
        self.entry_input_velocity_name =        ttk.Entry(frame)
        self.entry_input_vorticity_name =       ttk.Entry(frame)
        self.entry_output_velocity_name =       ttk.Entry(frame)
        self.entry_output_vorticity_name =      ttk.Entry(frame)
        self.button_load =                      ttk.Button(frame, text="Load", command=self.load)

        self.label_load_status.grid             (row=0, column=2, sticky="w")
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
        self.button_load.grid                   (row=7, column=0, sticky="w")
        self.lable_load.grid                    (row=7, column=1, sticky="w", columnspan=2)

        self.entry_mainDir.insert(0, "../output/")
        self.entry_bottleneck_name.insert(0, "bottleneck")
        self.entry_input_velocity_name.insert(0, "sci_input_velocity")
        self.entry_input_vorticity_name.insert(0, "sci_input_vorticity")
        self.entry_output_velocity_name.insert(0, "sci_output_velocity")
        self.entry_output_vorticity_name.insert(0, "sci_output_vorticity")

    def widgets_init_similarity(self, frame):
        self.button_compute_all =               ttk.Button(frame, text="Compute All", command=self.compute_all_similarity_matrix)
        self.label_compute_status =             ttk.Label(frame, text="Compute Status:", width=lable_width, anchor='w')
        self.label_measurement_method =         ttk.Label(frame, text="Measurement Method:", anchor='w', width=lable_width)
        self.measurement_method_tag =           ttk.StringVar(frame)
        self.measurement_method_tag.set("None") 
        self.measurement_dropdown =             ttk.OptionMenu(frame, self.measurement_method_tag, *self.method_menu, command=self.draw_similarity_matrix)
        
        self.button_compute_all.grid(           row=0, column=0, sticky="w")
        self.label_compute_status.grid(         row=0, column=1, sticky="w")
        self.label_measurement_method.grid(     row=1, column=0, sticky="w")
        self.measurement_dropdown.grid(         row=1, column=1, sticky="we")

    def widgets_init_find_n_closest(self, frame):
        self.label_frame_number =               ttk.Label(frame, text="Selected Frame:",            anchor='w', width=lable_width)
        self.label_n_closest =                  ttk.Label(frame, text="Number of closest frames:",  anchor='w', width=lable_width)
        self.label_exclude_local =              ttk.Label(frame, text="Exclude local frames range:",anchor='w', width=lable_width)
        self.entry_frame_number =               ttk.Entry(frame)
        self.entry_n_closest =                  ttk.Entry(frame)
        self.entry_exclude_local =              ttk.Entry(frame)
        self.subsubframe_n_closest_button =     ttk.Frame(frame)
        self.button_find_n_closest =            ttk.Button(self.subsubframe_n_closest_button, text="Find", command=self.find)
        self.button_find_n_closes_next =        ttk.Button(self.subsubframe_n_closest_button, text="Find Next", command=self.find_next)
        self.button_find_n_closes_prev =        ttk.Button(self.subsubframe_n_closest_button, text="Find Previous", command=self.find_prev)
        self.display_tag =                      ttk.StringVar(self.subsubframe_n_closest_button)
        self.display_dropdown =                 ttk.OptionMenu(self.subsubframe_n_closest_button, self.display_tag, *self.attrType_menu)
        self.display_tag.set(self.attrType_menu[0])
        
        self.label_frame_number.grid(           row=2, column=0, sticky="w")
        self.label_n_closest.grid(              row=3, column=0, sticky="w")
        self.label_exclude_local.grid(          row=4, column=0, sticky="w")
        self.entry_frame_number.grid(           row=2, column=1, sticky="w")
        self.entry_n_closest.grid(              row=3, column=1, sticky="w")
        self.entry_exclude_local.grid(          row=4, column=1, sticky="w")
        self.subsubframe_n_closest_button.grid( row=5, column=0, columnspan=2, sticky="w")
        self.button_find_n_closes_prev.grid(    row=0, column=0, sticky="w")
        self.button_find_n_closest.grid(        row=0, column=1, sticky="w")
        self.button_find_n_closes_next.grid(    row=0, column=2, sticky="w")
        self.display_dropdown.grid(             row=0, column=3, sticky="we")

        self.entry_frame_number.insert(0, "50")
        self.entry_n_closest.insert(0, "30")
        self.entry_exclude_local.insert(0, "5")

    def widgets_init_customize_bottleneck(self, frame):
        self.subsubframe_bottleneck_button =    ttk.Frame(frame)
        self.subsubframe_bottleneck_checkbox =  ttk.Frame(frame)
        self.canvas_bottleneck_checkbox  =      ttk.Canvas(self.subsubframe_bottleneck_checkbox, height=400)
        self.framecanvas_bottleneck_checkbox =  ttk.Frame(self.canvas_bottleneck_checkbox)
        self.scroll_bottleneck_checkbox =       ttk.Scrollbar(self.subsubframe_bottleneck_checkbox, orient="vertical", command=self.canvas_bottleneck_checkbox.yview)
        self.button_refresh_bottleneck =        ttk.Button(self.subsubframe_bottleneck_button, text="Refresh", command=self.refresh_bottleneck)
        self.button_compute_bottleneck =        ttk.Button(self.subsubframe_bottleneck_button, text="update", command=self.compute_and_draw)

        self.canvas_bottleneck_checkbox.configure(yscrollcommand=self.scroll_bottleneck_checkbox.set)

        self.button_refresh_bottleneck.grid(      row=0, column=0, sticky="w")
        self.button_compute_bottleneck.grid(      row=0, column=1, sticky="w")
        self.subsubframe_bottleneck_button.grid(  row=0, column=0, columnspan=2, sticky="w")
        self.subsubframe_bottleneck_checkbox.grid(row=1, column=0, columnspan=2, sticky="w")
        self.scroll_bottleneck_checkbox.pack(side="right", fill="y")
        self.canvas_bottleneck_checkbox.pack(side="left", fill="both", expand=True)
        self.canvas_bottleneck_checkbox.create_window((0,0), window=self.framecanvas_bottleneck_checkbox, anchor="nw")
        self.framecanvas_bottleneck_checkbox.bind("<Configure>", 
                                                  lambda e: self.canvas_bottleneck_checkbox.configure(scrollregion=self.canvas_bottleneck_checkbox.bbox("all")))
        self.canvas_bottleneck_checkbox.bind_all("<MouseWheel>", self.on_mousewheel)  # For Windows
        self.canvas_bottleneck_checkbox.bind_all("<Button-4>", self.on_mousewheel)  # For Linux
        self.canvas_bottleneck_checkbox.bind_all("<Button-5>", self.on_mousewheel)  # For Linux
    
    def widgets_init_similarity_matrix(self, frame):
        self.fig_similarity_matrix = Figure(figsize=(5, 5), dpi=100)
        self.ax_similarity_matrix = self.fig_similarity_matrix.add_subplot(111)
        self.canvas_similarity_matrix = FigureCanvasTkAgg(self.fig_similarity_matrix, master=frame)
        self.canvas_similarity_matrix.get_tk_widget().pack(side=ttk.TOP, fill=ttk.BOTH, expand=1)
    
    def widgets_init_distance_plot(self, frame):
        self.fig_distance_plot = Figure(figsize=(5, 2), dpi=100)
        self.ax_distance_plot = self.fig_distance_plot.add_subplot(111)
        self.canvas_distance_plot = FigureCanvasTkAgg(self.fig_distance_plot, master=frame)
        self.canvas_distance_plot.get_tk_widget().pack(side=ttk.TOP, fill=ttk.BOTH, expand=1)

    def on_mousewheel(self, event):
        """Handle mouse wheel scroll for different platforms."""
        if event.num == 4 or event.delta > 0:  # Scroll up
            self.canvas_bottleneck_checkbox.yview_scroll(-1, "units")
        elif event.num == 5 or event.delta < 0:  # Scroll down
            self.canvas_bottleneck_checkbox.yview_scroll(1, "units")

    def refresh_bottleneck(self, _=None):
        self.row_frame_list = []
        for widget in self.framecanvas_bottleneck_checkbox.winfo_children():
            widget.destroy()
        self.feature_vector_size = self.arr_bottleneck[0].shape[1]
        self.bottleneck_checkbox_vars = [ttk.BooleanVar(value=False) for _ in range(self.feature_vector_size)]
        self.activate_mask = torch.zeros(self.feature_vector_size, dtype=bool, device=device)

        for i in range(self.feature_vector_size):
            col_index = i % self.feature_vectors_per_row
            if col_index == 0:
                row_frame = ttk.Frame(self.framecanvas_bottleneck_checkbox)
                row_frame.grid(row=i // self.feature_vectors_per_row, column=0, sticky="w")
                self.row_frame_list.append(row_frame)
            check_button = ttk.Checkbutton(row_frame, text=f"Feature {i}", variable=self.bottleneck_checkbox_vars[i],width=10, command=self.ticked(i))
            check_button.grid(row=0, column=col_index, sticky="w")

    def ticked(self, i, _=None):
        def handler():
            self.activate_mask[i] = not self.activate_mask[i]
            print(f"Feature {i} is {'activated' if self.activate_mask[i] else 'deactivated'}")
        return handler

    def compute_all_similarity_matrix(self, _=None):
        self.similarity_matrix_list = []
        for method in self.method_menu:
            self.similarity_matrix_list.append(self.compute_similarity_matrix(method))
        self.label_compute_status.config(text="Compute Status: Successfull")

    def compute_similarity_matrix(self, method):
        n_frames = self.num_files
        with torch.no_grad():
            similarity_matrix = torch.zeros((n_frames, n_frames), device=device)
            frames = self.arr_bottleneck
            if method == self.method_menu[0]:
                diffs = frames - frames.transpose(0, 1)
                similarity_matrix = torch.mean(diffs ** 2, dim=2)
            elif method == self.method_menu[1]:
                frames_squeezed = frames.squeeze(1)
                frames_norm = torch.nn.functional.normalize(frames_squeezed, p=2, dim=1)
                similarity_matrix = 1 - torch.mm(frames_norm, frames_norm.t())
        return similarity_matrix

    def load(self, _=None):
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
            self.arr_bottleneck = torch.tensor(np.array(self.load_arrays(self.main_path, self.attr_bottleneck, self.num_files)), device=device)
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
            filename = os.path.join(path, f'{attr_name}_{i}.npy')
            frame = np.load(filename)
            frames.append(frame)
        return frames

    def draw(self, _=None):
        self.draw_similarity_matrix()
        self.draw_distance_plot()

    def draw_similarity_matrix(self, selected_frame=None, closest_frames=None, _=None):
        self.ax_similarity_matrix.clear()
        if self.similarity_matrix_list:
            self.similarity_matrix = self.similarity_matrix_list[self.method_menu.index(self.measurement_method_tag.get())]
            similarity_matrix =      self.similarity_matrix_list[self.method_menu.index(self.measurement_method_tag.get())]
            self.ax_similarity_matrix.imshow(similarity_matrix.cpu().numpy(), cmap='viridis', interpolation='nearest')

            if selected_frame is not None:
                self.ax_similarity_matrix.scatter(selected_frame, selected_frame, s=10, color='red', edgecolor='black', zorder=4)
                if closest_frames is not None:
                    for closest_frame in closest_frames:
                        self.ax_similarity_matrix.scatter(selected_frame, closest_frame, s=10, color='white', edgecolor='black', zorder=3)
                        self.ax_similarity_matrix.scatter(closest_frame, selected_frame, s=10, color='white', edgecolor='black', zorder=3)

            self.canvas_similarity_matrix.draw()

    def draw_distance_plot(self, selected_frame=None, closest_frames=None, _=None):
        self.ax_distance_plot.clear()
        if self.similarity_matrix_list:
            self.similarity_matrix = self.similarity_matrix_list[self.method_menu.index(self.measurement_method_tag.get())]
            similarity_matrix =      self.similarity_matrix_list[self.method_menu.index(self.measurement_method_tag.get())]
            distances = similarity_matrix[selected_frame, :].cpu()
            self.ax_distance_plot.plot(distances, '-o', markersize=1, zorder=1)
            if closest_frames is not None:
                self.ax_distance_plot.scatter(selected_frame, distances[selected_frame], s=10, color='red', edgecolor='black', zorder=4)
                self.ax_distance_plot.scatter(closest_frames, distances[closest_frames], s=10, color='white', edgecolor='black', zorder=3)
            self.canvas_distance_plot.draw()

    def find(self, _=None):
        closest_frames, distances = self.find_n_closest_frames()
        self.draw_similarity_matrix(int(self.entry_frame_number.get()), closest_frames)
        self.draw_distance_plot(int(self.entry_frame_number.get()), closest_frames)

    def find_next(self, _=None):
        next_number = int(self.entry_frame_number.get()) + 1
        self.entry_frame_number.delete(0, ttk.END)
        self.entry_frame_number.insert(0, str(next_number))
        self.find()

    def find_prev(self, _=None):
        prev_number = int(self.entry_frame_number.get()) - 1
        self.entry_frame_number.delete(0, ttk.END)
        self.entry_frame_number.insert(0, str(prev_number))
        self.find()

    def find_n_closest_frames(self):
        frame_number = int(self.entry_frame_number.get())
        n = int(self.entry_n_closest.get())
        exclude_local_frames = int(self.entry_exclude_local.get())

        with torch.no_grad():
            similarity_matrix = self.similarity_matrix
            distance_values = similarity_matrix[frame_number].clone().detach()
            start = max(0, frame_number - exclude_local_frames)
            end = min(len(distance_values), frame_number + exclude_local_frames + 1)
            distance_values[start:end] = float('inf')
            closest_distances, closest_frames = torch.topk(distance_values, k=n, largest=False)
            closest_distances = -closest_distances
            
        return closest_frames.cpu(), closest_distances.cpu()

    def compute_and_draw(self, _=None):
        pass