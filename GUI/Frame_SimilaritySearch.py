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

lable_width = 30

class Frame_SimilaritySearchLoad:
    def __init__(self, root):
        self.flag_load = False
        self.flag_success = True

        self.root = root
        self.frame = ttk.Frame(self.root)
        
        bold_font = font.Font(family="Helvetica", size=12, weight="bold")
        self.title = ttk.Label(self.frame, text="Step 1: Data Loading",   font=bold_font)
        self.title.grid(row=0, column=0, sticky="w")

        self.label_mainDir =                ttk.Label(self.frame, text="Main Directory:",        anchor='w', width=lable_width)
        self.label_bottleneck_name =        ttk.Label(self.frame, text="Bottleneck Name:",       anchor='w', width=lable_width)
        self.label_input_velocity_name =    ttk.Label(self.frame, text="Input Velocity Name:",   anchor='w', width=lable_width)
        self.label_input_vorticity_name =   ttk.Label(self.frame, text="Input Vorticity Name:",  anchor='w', width=lable_width)
        self.label_output_velocity_name =   ttk.Label(self.frame, text="Output Velocity Name:",  anchor='w', width=lable_width)
        self.label_output_vorticity_name =  ttk.Label(self.frame, text="Output Vorticity Name:", anchor='w', width=lable_width)
        self.load_status =                  ttk.Label(self.frame, text="Load Status")
        self.lable_status_bottleneck =      ttk.Label(self.frame, text=" ", width=10, anchor='w', wraplength=60)
        self.lable_status_input_velocity =  ttk.Label(self.frame, text=" ", width=10, anchor='w', wraplength=60)
        self.lable_status_input_vorticity = ttk.Label(self.frame, text=" ", width=10, anchor='w', wraplength=60)
        self.lable_status_output_velocity = ttk.Label(self.frame, text=" ", width=10, anchor='w', wraplength=60)
        self.lable_status_output_vorticity =ttk.Label(self.frame, text=" ", width=10, anchor='w', wraplength=60)
        self.lable_load =                   ttk.Label(self.frame, text="Load Status: ", width=15, anchor='w',)
        self.entry_mainDir =                ttk.Entry(self.frame)
        self.entry_bottleneck_name =        ttk.Entry(self.frame)
        self.entry_input_velocity_name =    ttk.Entry(self.frame)
        self.entry_input_vorticity_name =   ttk.Entry(self.frame)
        self.entry_output_velocity_name =   ttk.Entry(self.frame)
        self.entry_output_vorticity_name =  ttk.Entry(self.frame)
        self.botton_load =                  ttk.Button(self.frame, text="Load", command=self.load)

        self.entry_mainDir.insert(0, "../output/")
        self.entry_bottleneck_name.insert(0, "bottleneck")
        self.entry_input_velocity_name.insert(0, "sci_input_velocity")
        self.entry_input_vorticity_name.insert(0, "sci_input_vorticity")
        self.entry_output_velocity_name.insert(0, "sci_output_velocity")
        self.entry_output_vorticity_name.insert(0, "sci_output_vorticity")
        
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
        self.lable_load.grid                    (row=7, column=1, sticky="w")

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
            self.lable_load.config(text="Load Status: Successfull")
            self.num_files = self.inspect_num(self.main_path, self.attr_bottleneck)
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

class Frame_SimilaritySearchCompute:
    def __init__(self, root):
        self.root = root
        self.frame = ttk.Frame(self.root)

        bold_font = font.Font(family="Helvetica", size=12, weight="bold")
        self.label_title = ttk.Label(self.frame, text="Step 2: Compute Similarity Matrix", font=bold_font)
        self.label_title.grid(row=0, column=0, sticky="w", columnspan=2)

        self.label_frame_number =       ttk.Label(self.frame, text="Selected Frame:",            anchor='w', width=lable_width)
        self.label_n_closest =          ttk.Label(self.frame, text="Number of closest frames:",  anchor='w', width=lable_width)
        self.label_exclude_local =      ttk.Label(self.frame, text="Exclude local frames range:",anchor='w', width=lable_width)
        self.label_measurement_method = ttk.Label(self.frame, text="Measurement Method:",        anchor='w', width=lable_width)
        self.entry_frame_number =       ttk.Entry(self.frame)
        self.entry_n_closest =          ttk.Entry(self.frame)
        self.entry_exclude_local =      ttk.Entry(self.frame)
        self.measurement_method_tag =   ttk.StringVar(self.frame)
        self.measurement_dropdown =     ttk.OptionMenu(self.frame, self.measurement_method_tag, "MSE", "Cosine_distance")
        
        fig = Figure(figsize=(5, 4), dpi=100)
        plot = fig.add_subplot(1, 1, 1)
        canvas = FigureCanvasTkAgg(fig, self.frame)

        self.entry_frame_number.insert(0, "50")
        self.entry_n_closest.insert(0, "30")
        self.entry_exclude_local.insert(0, "5")
        self.measurement_method_tag.set("MSE") 

        self.label_measurement_method.grid(row=1, column=0, sticky="w")
        self.measurement_dropdown.grid(    row=1, column=1, sticky="we")
        canvas.get_tk_widget().grid(       row=2, column=0, columnspan=2, sticky="nsew")
        canvas.get_tk_widget().grid_forget()
        # remove canvas from frame
        


        # self.label_frame_number.grid(      row=2, column=0, sticky="w")
        # self.label_n_closest.grid(         row=3, column=0, sticky="w")
        # self.label_exclude_local.grid(     row=4, column=0, sticky="w")
        # self.entry_frame_number.grid(      row=2, column=1, sticky="w")
        # self.entry_n_closest.grid(         row=3, column=1, sticky="w")
        # self.entry_exclude_local.grid(     row=4, column=1, sticky="w")


