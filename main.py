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
#from IPython.display import clear_output #for colab


classes = (0, 90, 270)


##################################################################################
# Convolutional Neural Network
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3) #298, 298
        self.conv2 = nn.Conv2d(6, 16, 3) #147, 147
        self.conv3 = nn.Conv2d(16, 32, 3) #71, 71
        self.conv4 = nn.Conv2d(32, 64, 3) #33, 33
        self.fc1 = nn.Linear(64 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2, 2) # 62, 62 -> 31, 31        #298, 298 -> 149, 149
        x = F.max_pool2d(F.relu(self.conv2(x)), 2, 2) # 28, 28 -> 14, 14        #147, 147 -> 73, 73
        x = F.max_pool2d(F.relu(self.conv3(x)), 2, 2) # 12, 12 -> 6, 6          #71, 71 -> 35, 35
        x = F.max_pool2d(F.relu(self.conv4(x)), 2, 2)  # 12, 12 -> 6, 6          #33, 33 ->16, 16
        # print(x.size())
        x = x.view(-1, 64 * 16 * 16)
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
PATH = './trained_model/cnn.pth'
model.load_state_dict(torch.load(PATH))
###################################

tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
w_size = pg.size() # w_size[0]: width, w_size[1]: height

while True:
    img = PIL.ImageGrab.grab((30, 150, 800, 550)) # x_left, y_left, x_right, y_right
    img_small = img.resize((300, 300))
    img_tensor = tf(img_small)
    img_tensor = torch.unsqueeze(img_tensor, 0)

    output = model(img_tensor)
    _, predicted = torch.max(output, 1) #returns value, index
    angle = int(classes[predicted])
    print(output, angle)

    img_frame = np.array(img)
    img_frame = cv2.cvtColor(img_frame, cv2.COLOR_RGB2BGR)

    if angle == 90:
        img_frame = rotate_image(img_frame, 270)
    elif angle == 270:
        img_frame = rotate_image(img_frame, 90)

    mywin = "mywindow"
    cv2.namedWindow(mywin)   # create a named window
    cv2.moveWindow(mywin, 1540, 100)   # Move it to (40, 30)
    cv2.imshow(mywin, img_frame)
    if cv2.waitKey(100) == ord('q'):
        #cv2.destroyAllWindows()
        break
cv2.destroyAllWindows()



#img = cv2.imread("trash/image.jpg")
# img90 = rotate_image(img, 90)
# cv2.imwrite("image90.jpg", img90)


#for i in range(0, 360, 30):
#    img2 = rotate_image(img, i)
#    img3 = resize_image(img2, 200, 200)
#    cv2.imshow(mywin, img3)
#    cv2.waitKey(1000)
#    cv2.destroyAllWindows()