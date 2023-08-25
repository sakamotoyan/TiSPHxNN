import cv2
import os

# Output MP4 file path
output_path = './output_video'

# Frame rate (frames per second) of the output video
frame_rate = 60
stride = 1

# Function to sort files numerically
def numerical_sort(value):
    base = os.path.basename(value)
    return int(os.path.splitext(base)[0])

# Get the list of image files sorted numerically
image_files = []
for i in range(1,900,stride):
    image_files.append(f'./output/{i}.png')

# image_files = sorted(
#     [os.path.join(image_dir, file) for file in os.listdir(image_dir)],
#     key=numerical_sort
# )

# Get the first image to determine the frame size
first_image = cv2.imread(image_files[0])
height, width, layers = first_image.shape

print(height, width, layers)

# Create a video writer object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use appropriate codec
out = cv2.VideoWriter('multiphase.mp4', fourcc, frame_rate, (width, height))

# Iterate through the image files and write frames to the video
for image_file in image_files:
    image = cv2.imread(image_file)
    out.write(image)

cv2.destroyAllWindows()
# Release the video writer
out.release()