#resize images and store the resized images

import torch
import numpy as np
import os
import cv2
from tqdm import tqdm
import shutil #move
import csv
import pandas as pd
import torchvision
from torchvision.io import read_image
from torchvision.transforms import ToTensor, Lambda
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

def rename_files():
    for i in range(0, 360, 90):
        path_read = "C:\\Users\\rgogo\\Documents\\GitHub\\rotate_screen\\data\\" + str(i)
        path_write = "C:\\Users\\rgogo\\Documents\\GitHub\\rotate_screen\\data"
        files = [f for f in os.listdir(path_read) if os.path.isfile(os.path.join(path_read, f))]
        for f in tqdm(files, desc=("file " + str(i))):
            src = path_read + "/" + f
            dst = path_write + "/" + "a" + str(i) + "-" + f
            shutil.move(src, dst)
def make_csv():
    path_read = "C:\\Users\\rgogo\\Documents\\GitHub\\rotate_screen\\data"
    files = [f for f in os.listdir(path_read) if os.path.isfile(os.path.join(path_read, f))]
    with open('annotation.csv', 'w', newline = '') as csvfile:
        for f in tqdm(files):
            #file = path_read + "/" + f
            if ".jpg" not in f:
                continue

            label = -1
            if "a0" in f:
                label = 0
            elif "a90" in f:
                label = 90
            elif "a180" in f:
                label = 180
            elif "a270" in f:
                label = 270
            csvout = csv.writer(csvfile, delimiter=',', quotechar='"',
                                quoting=csv.QUOTE_MINIMAL)
            csvout.writerow([f, label])

class ImageDataset(Dataset):
    def __init__(self, annotation_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotation_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

path_read = "C:\\Users\\rgogo\\Documents\\GitHub\\rotate_screen\\data"
data = ImageDataset(path_read + "/annotation.csv", path_read)#, None,
                    #Lambda(lambda y: torch.zeros(4, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)))

dataloader = DataLoader(data, batch_size=4, shuffle=True)
# Display image and label.
images, labels = next(iter(dataloader))
print(f"Feature batch shape: {images.size()}")
print(f"Labels batch shape: {labels.size()}")
img = images[0].squeeze()
label = labels[0]

images = [img.swapaxes(0,1).swapaxes(1, 2) for img in images]
#img = img.swapaxes(0,1).swapaxes(1,2)
print(f"Label: {label}")
plt.imshow(images[0])
plt.show()



