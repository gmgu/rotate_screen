import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import pyautogui as pg
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import pandas as pd
# from IPython.display import clear_output #for colab

classes = (0, 90, 270)


##################################################################################
# Convolutional Neural Network
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.conv4 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 14 * 14, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2, 2)  # 254, 254 -> 127, 127
        x = F.max_pool2d(F.relu(self.conv2(x)), 2, 2)  # 125, 125 -> 62, 62
        x = F.max_pool2d(F.relu(self.conv3(x)), 2, 2)  # 60, 60 -> 30, 30
        x = F.max_pool2d(F.relu(self.conv4(x)), 2, 2)  # 28, 28 -> 14, 14
        # print(x.size())
        x = x.view(-1, 64 * 14 * 14)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    height, width = image.shape[:2]  # image shape has 3 dimensions
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)  # get rotation matrix

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rot_mat[0, 0])
    abs_sin = abs(rot_mat[0, 1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rot_mat[0, 2] += bound_w / 2 - image_center[0]
    rot_mat[1, 2] += bound_h / 2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    result = cv2.warpAffine(image, rot_mat, (bound_w, bound_h))
    return result

    #result = cv2.warpAffine(image, rot_mat, (0, 0))  # rotate image
    #return result


def resize_image(image, width, height):
    result = cv2.resize(image, dsize=(width, height), interpolation=cv2.INTER_AREA)
    return result


###################################
model = CNN()
PATH = 'trained_model/256_cnn.pth'
model.load_state_dict(torch.load(PATH))
###################################

tf = transforms.Compose([
    transforms.CenterCrop(714),
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
w_size = pg.size()  # w_size[0]: width, w_size[1]: height

queue = []# QUEUE_SIZE frame results
QUEUE_SIZE = 100

mywin = "mywindow"
cv2.namedWindow(mywin)   # create a named window
cv2.moveWindow(mywin, 1400, 0)   # Move it to (40, 30)
i = 0
while True:
    with torch.no_grad():
        i += 1
        img = PIL.ImageGrab.grab((0, 150, w_size[0], w_size[1] - 150))  # x_left, y_left, x_right, y_right
        img_tensor = tf(img)
        img_tensor = torch.unsqueeze(img_tensor, 0)

        output = model(img_tensor)
        _, predicted = torch.max(output, 1) #returns value, index
        angle = int(classes[predicted])
        print(output, angle)

        queue.append(angle)
        if len(queue) > QUEUE_SIZE:
            queue.pop(0)

        img = PIL.ImageGrab.grab((0, 0, w_size[0], w_size[1]))  # full screen capture
        SCALE = 0.8
        img_scale = img.resize(
            (int(w_size[0] * SCALE), int((w_size[1]) * SCALE))
             )
        img_frame = np.array(img_scale)
        img_frame = cv2.cvtColor(img_frame, cv2.COLOR_RGB2BGR)

        max_angle = max(set(queue), key=queue.count)
        if max_angle == 90:
            img_frame = rotate_image(img_frame, 270)
        elif max_angle == 270:
            img_frame = rotate_image(img_frame, 90)

        cv2.imshow(mywin, img_frame)
        if cv2.waitKey(1) == ord('q'):
            break
cv2.destroyAllWindows()
