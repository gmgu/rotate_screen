#resize images and store the resized images

import numpy as np
import os
import cv2
import csv
from tqdm import tqdm

def resize_image(image, width, height):
    result = cv2.resize(image, dsize=(width, height), interpolation=cv2.INTER_AREA)
    return result

#old script
def process_data1():
    for i in range(0, 360, 90):
        path = "C:\\Users\\rgogo\\Documents\\GitHub\\rotate_screen\\data_all\\" + str(i)
        path_write = "C:\\Users\\rgogo\\Documents\\GitHub\\rotate_screen\\data\\" + str(i)
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        for f in tqdm(files, desc=("file "+str(i))):
            img = cv2.imread(path + "/" + f)
            img2 = resize_image(img, 64, 64)
            cv2.imwrite(path_write + "/" + f, img2)

with open('annotation.csv', 'w', newline='') as csvfile:
    for i in [0, 90, 270]:
        path = "C:\\Users\\rgogo\\Documents\\GitHub\\rotate_screen\\data_all\\" + str(i)
        path_write = "C:\\Users\\rgogo\\Documents\\GitHub\\rotate_screen\\data"
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

        for f in tqdm(files, desc=("file " + str(i))):
            if ".jpg" not in f:
                continue

            #img = cv2.imread(path + "/" + f)
            #img2 = resize_image(img, 300, 300)
            #cv2.imwrite(path_write + "/a" + str(i) + "-" + f, img2)

            csvout = csv.writer(csvfile, delimiter=',', quotechar='"',
                                quoting=csv.QUOTE_MINIMAL)
            csvout.writerow(["a" + str(i) + "-" +f, i])