import numpy as np
import os
from scipy.ndimage import rotate

class SingleFrameProcessor:
    def __init__(self, frame_number, frame_attrs, frame_path, dim=2):
        if dim != 2: raise ValueError("Only 2D frames are supported at the moment")
        
        self.frame_number = frame_number
        self.frame_attrs = frame_attrs
        self.frame_path = frame_path
        self.attr_paths = {}

        for attr in frame_attrs:
            self.attr_paths[attr] = os.path.join(frame_path, f"{attr}_{frame_number}.npy")
        
        self.resolution = (np.load(self.attr_paths[frame_attrs[0]]).shape[0], np.load(self.attr_paths[frame_attrs[0]]).shape[1])
        
    def getCroppedData2D(self, x, y, w, h, attr):
        data = np.load(self.attr_paths[attr])
        return data[y:y+h, x:x+w]
    
    def getMask2D(self, x, y, w, h, attr, masked_area='outside'):
        data = np.load(self.attr_paths[attr])
        mask = np.zeros(data.shape, dtype=bool)
        mask[y:y+h, x:x+w] = True
        if masked_area == 'inside':
            mask = np.logical_not(mask)
        return mask
    
    def generateMaskedSerialFrames2D(self, x, y, w, h, output_path, motion='translation', stride=1, degree=5):
        for attr in self.frame_attrs:
            data = np.load(self.attr_paths[attr])
            masked_data = np.zeros_like(data)
            cropped_data = self.getCroppedData2D(x, y, w, h, attr)
            counter = 0
            if motion == 'translation':
                for x_move in range(self.resolution[0]-w+1 // stride):
                    for y_move in range(self.resolution[1]-h+1 // stride):
                        if x_move*stride+w > self.resolution[0] or y_move*stride+h > self.resolution[1]: break
                        masked_data[y_move*stride:y_move*stride+h, x_move*stride:x_move*stride+w] = cropped_data
                        np.save(os.path.join(output_path, f"{attr}_{counter}.npy"), masked_data)
                        counter += 1
                        masked_data.fill(.0)
            elif motion == 'rotation':
                    for angle in range(0, 360, degree):
                        # Initialize a larger array to hold the cropped_data centered within
                        larger_array = np.zeros((self.resolution[0], self.resolution[1]) + cropped_data.shape[2:], dtype=cropped_data.dtype)
                        x_offset = (self.resolution[0] - cropped_data.shape[0]) // 2
                        y_offset = (self.resolution[1] - cropped_data.shape[1]) // 2
                        # Place cropped_data in the center of larger_array
                        larger_array[x_offset:x_offset+cropped_data.shape[0], y_offset:y_offset+cropped_data.shape[1], ...] = cropped_data

                        # Rotate the larger array
                        rotated_array = np.zeros_like(larger_array)
                        for index in np.ndindex(rotated_array.shape[2:]):  # Iterate over additional dimensions
                            slice_index = (slice(None), slice(None)) + index
                            rotated_array[slice_index] = rotate(larger_array[slice_index], angle, reshape=False, mode='constant')

                        # Calculate the slicing indices to extract the centered region of rotated_array
                        x_start = max((rotated_array.shape[0] - self.resolution[0]) // 2, 0)
                        y_start = max((rotated_array.shape[1] - self.resolution[1]) // 2, 0)
                        x_end = x_start + self.resolution[0]
                        y_end = y_start + self.resolution[1]

                        # Ensure the slicing does not exceed the dimensions of masked_data
                        x_slice_len = min(masked_data.shape[0], x_end - x_start)
                        y_slice_len = min(masked_data.shape[1], y_end - y_start)

                        # Update masked_data with the centered slice from rotated_array, ensuring the shapes align
                        masked_data[:x_slice_len, :y_slice_len, ...] = rotated_array[x_start:x_start + x_slice_len, y_start:y_start + y_slice_len, ...]

                        np.save(os.path.join(output_path, f"{attr}_{counter}.npy"), masked_data)
                        counter += 1
                        masked_data.fill(0)
            else:
                raise ValueError("Only translation is supported at the moment")


main_path = os.path.join('../dataset_test2','dataset_old')
output_path = os.path.join('../dataset_test2','dataset')
if not os.path.exists(output_path): os.makedirs(output_path)
for file in os.listdir(output_path): print(f"Removing {file}"); os.remove(os.path.join(output_path, file))

attrs=['density', 'strainRate', 'strainRate2vorticity', 'vel2StrainRate', 'vel2vorticity', 'velocity']
frame_number = 400

singleFrameProcessor = SingleFrameProcessor(frame_number, attrs, main_path)
singleFrameProcessor.generateMaskedSerialFrames2D(100, 100, 128, 128, output_path, motion='rotation', degree=5)


        