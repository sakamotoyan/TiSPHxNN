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
        self.attr_shapes = {}

        for attr in frame_attrs:
            self.attr_paths[attr] = os.path.join(frame_path, f"{attr}_{frame_number}.npy")
        
        self.resolution = (np.load(self.attr_paths[frame_attrs[0]]).shape[0], np.load(self.attr_paths[frame_attrs[0]]).shape[1])
        for attr in frame_attrs:
            shape = np.load(self.attr_paths[attr]).shape[2:]
            if len(shape) == 0:
                shape = (1,)
            self.attr_shapes[attr] = shape
        
        # print(f"Resolution: {self.resolution}")
        # print(f"Attribute shapes: {self.attr_shapes}")
        # print(f"Attribute dimensions: {self.attr_dims}")

        
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
    
    def rotate_2Dvectors(self, x, y, angle):
        radians = np.radians(angle)
        cos_angle = np.cos(radians)
        sin_angle = np.sin(radians)
        x_rotated = x * cos_angle - y * sin_angle
        y_rotated = x * sin_angle + y * cos_angle
        return np.stack((x_rotated, y_rotated), axis=-1)
    
    def rotate_2x2_matrices(self, matrices, angle):
        # Assuming matrices have shape (..., N, N) where N is the size of the matrix (e.g., 3 for a 3x3 matrix)
        N = matrices.shape[-1]
        rotated_matrices = np.zeros_like(matrices)
        
        for i in range(N):  # Loop over rows
            row_vectors = matrices[..., i, :]  # Extract row vectors
            rotated_row_vectors = self.rotate_2Dvectors(row_vectors[..., 0], row_vectors[..., 1], angle)  # Rotate row vectors
            rotated_matrices[..., i, 0] = rotated_row_vectors[..., 0]
            rotated_matrices[..., i, 1] = rotated_row_vectors[..., 1]
            if N > 2:  # If the matrix is larger than 2x2, handle additional dimensions
                raise NotImplementedError("Matrices larger than 2x2 are not supported at the moment")
                
        
        return rotated_matrices

    def generateMaskedSerialFrames2D(self, x, y, w, h, output_path, motion='translation', stride=1, degree=5):
        for attr in self.frame_attrs:
            data = np.load(self.attr_paths[attr])
            cropped_data = self.getCroppedData2D(x, y, w, h, attr)
            counter = 0
            if motion == 'translation':
                for x_move in range(self.resolution[0]-w+1 // stride):
                    for y_move in range(self.resolution[1]-h+1 // stride):
                        masked_data = np.zeros_like(data)
                        if x_move*stride+w > self.resolution[0] or y_move*stride+h > self.resolution[1]: break
                        masked_data[y_move*stride:y_move*stride+h, x_move*stride:x_move*stride+w] = cropped_data
                        np.save(os.path.join(output_path, f"{attr}_{counter}.npy"), masked_data)
                        counter += 1
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

                        # Rotate vectors within the array if the last dimension size is 2 (indicating 2D vectors)
                        if cropped_data.shape[-1] == 2:
                            rotated_vectors = self.rotate_2Dvectors(rotated_array[..., 0], rotated_array[..., 1], angle)
                            rotated_array[..., 0] = rotated_vectors[..., 0]
                            rotated_array[..., 1] = rotated_vectors[..., 1]
                        
                        # Rotate matrices within the array if the last two dimensions are square (indicating 2D matrices)
                        if len(cropped_data.shape) > 3 and cropped_data.shape[-2] == cropped_data.shape[-1]:
                            rotated_matrices = self.rotate_2x2_matrices(rotated_array, angle)
                            rotated_array = rotated_matrices

                        np.save(os.path.join(output_path, f"{attr}_{counter}.npy"), rotated_array)
                        counter += 1
            else:
                raise ValueError("Only translation is supported at the moment")


main_path = os.path.join('../dataset_test2','dataset_old')
output_path = os.path.join('../dataset_test2','dataset')
if not os.path.exists(output_path): os.makedirs(output_path)
for file in os.listdir(output_path): print(f"Removing {file}"); os.remove(os.path.join(output_path, file))

attrs=['density', 'strainRate', 'strainRate2vorticity', 'vel2StrainRate', 'vel2vorticity', 'velocity']
attr_shapes = [(1), (2,2), (1), (2,2), (1), (2)]
frame_number = 400

singleFrameProcessor = SingleFrameProcessor(frame_number, attrs, main_path)
singleFrameProcessor.generateMaskedSerialFrames2D(100, 100, 128, 128, output_path, motion='rotation', degree=5)


        