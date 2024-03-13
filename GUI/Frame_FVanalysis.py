import tkinter as Tk
from tkinter import ttk

import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
from tkinter import font
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.gridspec import GridSpec
import math
import torch

from Frame_ControlPanel import Frame_ControlPanel, Scrollable_frame

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lable_width = 20
status_width = 8
warp_length = status_width*10

class Frame_FVanalysis:
    
    def __init__(self, root, frame_control_panel:Frame_ControlPanel) -> None:
        ''' Input Parameters'''
        self.root = root
        self.frame_control_panel = frame_control_panel

        ''' Frame 1 '''
        self.frame_control = ttk.LabelFrame(self.root, text='FV Analysis')                          # Frame 1: Control Panel
        self.subframe_checkFeature = ttk.LabelFrame(self.frame_control, text='Entry Selection')     # Frame 1 -> Subframe 1: Check Feature
        self.widgets_init_checkFeature(self.subframe_checkFeature)                                  # Frame 1 -> Subframe 1 PACK
        self.subframe_selectFrame = ttk.LabelFrame(self.frame_control, text='Frame Selection')      # Frame 1 -> Subframe 2: Select Frame
        self.widgets_init_selectFrame(self.subframe_selectFrame)                                    # Frame 1 -> Subframe 2 PACK
        self.frame_control.pack(side='left', fill='both', expand=True)                              # Frame 1 PACK
        ''' Attributes of Frame 1 -> Subframe 1'''
        self.feature_vectors_per_row = 5                                                            # [Attr] Frame 1 -> Subframe 1: Assign number of feature vectors displayed per row
        self.row_frame_list = []                                                                    # [Attr] Frame 1 -> Subframe 1: Store checkbox row frame                                       
        self.feature_vector_size = None                                                             # [Attr] Frame 1 -> Subframe 1: Store feature vector size
        self.bottleneck_checkbox_vars = []                                                          # [Attr] Frame 1 -> Subframe 1: Store bottleneck checkbox variables 
        self.activate_mask = None                                                                   # [Attr] Frame 1 -> Subframe 1: Store activate mask for bottleneck checkbox variables

        ''' Frame 2 '''
        self.frame_vis = ttk.LabelFrame(self.root, text='Visualization')                            # Frame 2: Visualization
        self.subframe_FVplot = ttk.LabelFrame(self.frame_vis, text='Feature Vector Plot')           # Frame 2 -> Subframe 1: Feature Vector Plot
        self.widgets_init_FVplot(self.subframe_FVplot)                                              # Frame 2 -> Subframe 1 PACK
        self.frame_vis.pack(side='left', fill='both', expand=True)                                  # Frame 2 PACK
        ''' Attributes of Frame 2 -> Subframe 1'''

    def widgets_init_checkFeature(self, frame:ttk.Frame):
        self.subsubframe_checkFeature_buttons =         ttk.Frame(frame)
        self.subsubframe_checkFeature_checkboxes =      ttk.Frame(frame)
        self.scrollable_frame_checkFeature_checkboxes = Scrollable_frame(self.subsubframe_checkFeature_checkboxes)
        self.framecanvas_bottleneck_checkboxes =      self.scrollable_frame_checkFeature_checkboxes.scrollframe
        self.button_refresh_bottleneck =                ttk.Button(self.subsubframe_checkFeature_buttons, text='Refresh', command=self.refresh_bottleneck)
        self.button_select_all_bottleneck =             ttk.Button(self.subsubframe_checkFeature_buttons, text='Select All', command=self.select_all_bottleneck)
        self.button_deselect_all_bottleneck =           ttk.Button(self.subsubframe_checkFeature_buttons, text='Deselect All', command=self.deselect_all_bottleneck)
        self.button_refresh_bottleneck          .grid(row=0, column=0, sticky='nsew')
        self.button_select_all_bottleneck       .grid(row=0, column=1, sticky='nsew')
        self.button_deselect_all_bottleneck     .grid(row=0, column=2, sticky='nsew')
        self.subsubframe_checkFeature_buttons   .pack(side='top', fill='x', expand=True)
        self.subsubframe_checkFeature_checkboxes.pack(side='top', fill='both', expand=True)
        frame                                   .pack(side='left', fill='both', expand=True)

    def widgets_init_selectFrame(self, frame:ttk.Frame):
        pass
        
    def widgets_init_FVplot(self, frame:ttk.Frame):
        self.fig_FVplot = Figure(figsize=(5, 3), dpi=100)
        self.ax_FVplot = self.fig_FVplot.add_subplot(111)
        self.canvas_FVplot = FigureCanvasTkAgg(self.fig_FVplot, frame)
        self.canvas_FVplot.get_tk_widget().pack(side='top', fill='both', expand=True)
        frame.pack(side='top', fill='x', expand=True)
    
    def post_refresh_vis(func):
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)  # Execute the original function
            self.refresh_vis()  # Call self.refresh_vis() after the original function
            return result
        return wrapper
    
    def refresh_vis(self, _=None):
        self.draw_FVplot()

    @post_refresh_vis
    def refresh_bottleneck(self, _=None):
        self.row_frame_list = []
        for widget in self.framecanvas_bottleneck_checkboxes.winfo_children():
            widget.destroy()
        self.feature_vector_size = self.frame_control_panel.arr_bottleneck[0].shape[1]
        self.bottleneck_checkbox_vars = [Tk.BooleanVar(value=True) for _ in range(self.feature_vector_size)]
        self.activate_mask = torch.zeros(self.feature_vector_size, dtype=bool, device=device)
        self.activate_mask[...]=True

        for i in range(self.feature_vector_size):
            col_index = i % self.feature_vectors_per_row
            if col_index == 0:
                row_frame = ttk.Frame(self.framecanvas_bottleneck_checkboxes)
                row_frame.grid(row=i // self.feature_vectors_per_row, column=0, sticky="w")
                self.row_frame_list.append(row_frame)
            check_button = ttk.Checkbutton(row_frame, text=f"Feature {i}", variable=self.bottleneck_checkbox_vars[i], width=10, command=self.onChange_featureVector_selection(i))
            check_button.grid(row=0, column=col_index, sticky="w")

    @post_refresh_vis
    def select_all_bottleneck(self, _=None):
        for var in self.bottleneck_checkbox_vars:
            var.set(True)
        self.activate_mask[...] = True

    @post_refresh_vis
    def deselect_all_bottleneck(self, _=None):
        for var in self.bottleneck_checkbox_vars:
            var.set(False)
        self.activate_mask[...] = False

    
    def onChange_featureVector_selection(self, i, _=None):
        def handler():
            self.activate_mask[i] = not self.activate_mask[i]
            print(f"Feature {i} is {'activated' if self.activate_mask[i] else 'deactivated'}")
            self.refresh_vis()
        return handler
    
    def draw_FVplot(self):
        self.ax_FVplot.clear()
        print(self.frame_control_panel.arr_bottleneck.shape)
        FVcpu = self.frame_control_panel.arr_bottleneck.cpu()
        for i in range(self.feature_vector_size):
            if self.activate_mask[i]:
                self.ax_FVplot.plot(FVcpu[:, 0, i], label=f"Feature {i}")
        self.canvas_FVplot.draw()