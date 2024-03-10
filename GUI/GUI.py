import numpy as np
import os
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import Tk
from tkinter import ttk
import tkinter as tk
from PIL import Image, ImageTk

from Frame_ControlPanel import Frame_ControlPanel

from ctypes import windll
windll.shcore.SetProcessDpiAwareness(1)

class GUI:
    def __init__(self):
        self.root = Tk()
        self.root.title("GUI")

        self.notebook = ttk.Notebook(self.root)
        self.tagframe1 = ttk.Frame(self.notebook)
        self.tagframe2 = ttk.Frame(self.notebook)

        self.frame_control_panel = Frame_ControlPanel(self.tagframe1)
        self.frame_control_panel.frame_control.pack(side="left", fill="both", expand=True)
        self.frame_control_panel.frame_datavis.pack(side="left", fill="both", expand=True)
        self.frame_control_panel.frame_scivis.pack(side="left", fill="both", expand=True)


        self.notebook.add(self.tagframe1, text="Control Panel")
        self.notebook.add(self.tagframe2, text="Feature Vector Analysis")
        self.notebook.pack(fill="both", expand=True)

        self.root.mainloop()

gui = GUI()