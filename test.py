import numpy as np
import cv2


dims = (64,48) # dimensions of images to train/test with

img = cv2.imread(r"/home/jakub/Desktop/SDU/masters/sem_1/embedded/MiniProject/test_image.jpg",0) # read img as grayscale
img = cv2.resize(img, dims, interpolation = cv2.INTER_AREA)	# resize img to fit dims
if img is not None:
    img = img / 255 # normalize pixel vals to be between 0 - 1

img_np = np.array(img) # Convert to numpy array
img_flat = img_np.flatten() # Flatten the image

with open("image.txt", "w")as f:
    f.write("{")
    
    for i in img_flat:
        f.write(str(i))
        f.write(", ")

    f.write("}")