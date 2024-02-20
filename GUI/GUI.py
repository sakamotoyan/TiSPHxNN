import numpy as np
import os
import matplotlib.pyplot as plt
import tkinter as ttk
from tkinter import scrolledtext
from tkinter import Toplevel, Label
from PIL import Image, ImageTk

from Frame_SimilaritySearch import Frame_SimilaritySearchLoad, \
    Frame_SimilaritySearchCompute, Frame_SimilaritySearchNClosest, Frame_SimilaritySearchBottleneckCustomize

class GUI:
    def __init__(self):
        self.root = ttk.Tk()
        self.root.title("GUI")

        self.frame_load = Frame_SimilaritySearchLoad(self.root)
        self.frame_load.frame.grid(row=0, column=0, sticky="nsew")

        self.frame_compute = Frame_SimilaritySearchCompute(self.root, self.frame_load)
        self.frame_compute.frame.grid(row=1, column=0, sticky="nsew")

        self.frame_n_closest = Frame_SimilaritySearchNClosest(self.root, self.frame_load, self.frame_compute)
        self.frame_n_closest.frame.grid(row=0, column=1, sticky="nsew", rowspan=2)

        self.frame_bottleneck_customize = Frame_SimilaritySearchBottleneckCustomize(self.root, self.frame_load, self.frame_compute, self.frame_n_closest)
        self.frame_bottleneck_customize.frame.grid(row=0, column=2, sticky="nsew")

        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
        # self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        self.root.mainloop()

gui = GUI()