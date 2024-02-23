import numpy as np
import os

class SingleFrameProcessor:
    def __init__(self, frame_number, frame_attrs, frame_path):
        self.frame_number = frame_number
        self.frame_attrs = frame_attrs
        self.frame_path = frame_path
        self.attr_paths = {}

        for attr in frame_attrs:
            self.attr_paths[attr] = os.path.join(frame_path, f"{attr}_{frame_number}.npy")
        