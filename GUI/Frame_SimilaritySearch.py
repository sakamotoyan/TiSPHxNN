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

        self.root = root
        bold_font = font.Font(family="Helvetica", weight="bold")
        self.frame = ttk.LabelFrame(self.root, text="Step 1: Data Loading", font=bold_font)

        self.label_mainDir =                ttk.Label(self.frame, text="Main Directory:",        anchor='w', width=lable_width)
        self.label_bottleneck_name =        ttk.Label(self.frame, text="Bottleneck Name:",       anchor='w', width=lable_width)
        self.label_input_velocity_name =    ttk.Label(self.frame, text="Input Velocity Name:",   anchor='w', width=lable_width)
        self.label_input_vorticity_name =   ttk.Label(self.frame, text="Input Vorticity Name:",  anchor='w', width=lable_width)
        self.label_output_velocity_name =   ttk.Label(self.frame, text="Output Velocity Name:",  anchor='w', width=lable_width)
        self.label_output_vorticity_name =  ttk.Label(self.frame, text="Output Vorticity Name:", anchor='w', width=lable_width)
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
        self.label_input_velocity_name.grid     (row=3, column=0, sticky="w")
        self.label_input_vorticity_name.grid    (row=4, column=0, sticky="w")
        self.label_output_velocity_name.grid    (row=5, column=0, sticky="w")
        self.label_output_vorticity_name.grid   (row=6, column=0, sticky="w")
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
    def __init__(self, root, Frame_SimilaritySearchLoad:Frame_SimilaritySearchLoad):
        self.root = root
        self.Frame_SimilaritySearchLoad = Frame_SimilaritySearchLoad
        self.similarity_matrix = None

        bold_font =                     font.Font(family="Helvetica", weight="bold")
        self.frame =                    ttk.LabelFrame(self.root, text="Step 2: Compute Similarity Matrix", font=bold_font)

        self.label_measurement_method = ttk.Label(self.frame, text="Measurement Method:", anchor='w', width=lable_width)
        self.measurement_method_tag =   ttk.StringVar(self.frame)
        self.measurement_method_tag.set("None") 
        self.measurement_dropdown =     ttk.OptionMenu(self.frame, self.measurement_method_tag, "MSE", "Cosine_distance", command=self.compute_similarity_matrix_and_draw)
        
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.plot = self.fig.add_subplot(1, 1, 1)
        self.canvas = FigureCanvasTkAgg(self.fig, self.frame)

        self.label_measurement_method.grid(row=1, column=0, sticky="w")
        self.measurement_dropdown.grid(    row=1, column=1, sticky="we")
        self.canvas.get_tk_widget().grid(       row=2, column=0, columnspan=2, sticky="nsew")
        self.canvas.get_tk_widget().grid(       row=2, column=0, columnspan=2, sticky="nsew")
        self.canvas.get_tk_widget().grid_forget()
        
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
    
    def compute_similarity_matrix_and_draw(self, _=None):
        self.similarity_matrix = self.compute_similarity_matrix()
        self.draw_similarity_matrix()
    
    def compute_similarity_matrix(self):
        method = self.measurement_method_tag.get()
        if method == "None" or self.Frame_SimilaritySearchLoad.arr_bottleneck is None:
            return
        n_frames = self.Frame_SimilaritySearchLoad.num_files
        similarity_matrix = np.zeros((n_frames, n_frames))
        for i in range(n_frames):
            for j in range(n_frames):
                frame1 = self.Frame_SimilaritySearchLoad.arr_bottleneck[i]
                frame2 = self.Frame_SimilaritySearchLoad.arr_bottleneck[j]
                if method == "MSE":
                    similarity_matrix[i, j] = self.mse(frame1, frame2)
                elif method == "Cosine_distance":
                    similarity_matrix[i, j] = self.cosine_distance(frame1.flatten(), frame2.flatten())
        self.similarity_matrix = similarity_matrix
        return similarity_matrix
    
    def draw_similarity_matrix(self):
        if self.similarity_matrix is None or self.Frame_SimilaritySearchLoad.arr_bottleneck is None:
            return
        self.plot.clear()
        self.plot.imshow(self.similarity_matrix, cmap='viridis')
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=2, column=0, columnspan=2, sticky="nsew")


class Frame_SimilaritySearchFind:
    def __init__(self, root):
        self.root = root
        bold_font = font.Font(family="Helvetica", weight="bold")
        self.frame = ttk.LabelFrame(self.root, text="Step 3: Find Similar Frames", font=bold_font)

        self.label_frame_number =       ttk.Label(self.frame, text="Selected Frame:",            anchor='w', width=lable_width)
        self.label_n_closest =          ttk.Label(self.frame, text="Number of closest frames:",  anchor='w', width=lable_width)
        self.label_exclude_local =      ttk.Label(self.frame, text="Exclude local frames range:",anchor='w', width=lable_width)
        self.entry_frame_number =       ttk.Entry(self.frame)
        self.entry_n_closest =          ttk.Entry(self.frame)
        self.entry_exclude_local =      ttk.Entry(self.frame)

        self.label_frame_number.grid(      row=2, column=0, sticky="w")
        self.label_n_closest.grid(         row=3, column=0, sticky="w")
        self.label_exclude_local.grid(     row=4, column=0, sticky="w")
        self.entry_frame_number.grid(      row=2, column=1, sticky="w")
        self.entry_n_closest.grid(         row=3, column=1, sticky="w")
        self.entry_exclude_local.grid(     row=4, column=1, sticky="w")

        self.entry_frame_number.insert(0, "50")
        self.entry_n_closest.insert(0, "30")
        self.entry_exclude_local.insert(0, "5")