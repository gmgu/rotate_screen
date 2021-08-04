import cv2
import os
import numpy as np
from tqdm import tqdm

#find minimum height and width of the images
PATH = "C:\\Users\\rgogo\\Documents\\GitHub\\rotate_screen\\data\\0"
files = [f for f in os.listdir(PATH) if os.path.isfile(os.path.join(PATH, f))]

#find min-max width-height
min_height = np.Inf
min_width = np.Inf
max_height = 0
max_width = 0
for f in tqdm(files):
    img = cv2.imread(PATH + "/" + f)
    height, width = img.shape[:2]

    if height < min_height:
        min_height = height
    if width < min_width:
        min_width = width
    if height > max_height:
        max_height = height
    if width > max_width:
        max_width = width

print("Min size: {} {}".format(min_width, min_height))
print("Max size: {} {}".format(max_width, max_height))