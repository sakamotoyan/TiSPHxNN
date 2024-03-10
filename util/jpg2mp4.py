import cv2
import os

# Output MP4 file path
output_path = '../output008'
input_path = output_path

# Frame rate (frames per second) of the output video
start_index_list = [0,   4855,6515,7755,9925, 10755,11785,12815]
end_index_list =   [4850,6510,7750,9920,10750,11780,12810,13845]
start_index_list = [0,  206,477]
end_index_list =   [205,476,602]
frame_rate = 24
stride = 1


def process(file_name, seq):
    # Get the list of image files sorted numerically
    image_files = []
    for i in range(start_index,end_index,stride):
        image_files.append(f'./{input_path}/{file_name}_{i}.png')

    # Get the first image to determine the frame size
    first_image = cv2.imread(image_files[0])
    height, width, layers = first_image.shape

    print(height, width, layers)

    # Create a video writer object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use appropriate codec
    out = cv2.VideoWriter(f'{output_path}/{file_name}_{seq}.mp4', fourcc, frame_rate, (width, height))

    # Iterate through the image files and write frames to the video
    for image_file in image_files:
        image = cv2.imread(image_file)
        out.write(image)

    cv2.destroyAllWindows()
    # Release the video writer
    out.release()

# for start_index, end_index, i in zip(start_index_list, end_index_list, range(len(start_index_list))):
#     start_index = start_index
#     end_index = end_index
#     i = i
#     process('sci_density', i)
#     process('sci_velocity', i)
#     process('sci_strainRate2vorticity', i)
#     process('sci_vel2vorticity', i)


for start_index, end_index, i in zip(start_index_list, end_index_list, range(len(start_index_list))):
    start_index = start_index
    end_index = end_index
    i = i
    process('sci_input_velocity',  i)
    process('sci_output_velocity', i)
    process('sci_input_vorticity', i)
    process('sci_output_vorticity',i)

# process('frame_bottleneck')

# process('input_vorticity_hist')
# process('output_vorticity_hist')
# process('strainRate_compression')
# process('strainRate_rotation')
# process('strainRate_shear')
# process('vel_hsv')

