# This script produces screen captured images
# input: image directory, [angle], [delay]
# Output: images that named in the form "a[angle]-[id].jpg" to the image directory
#           for every [delay] ms
import pyautogui as pg
import numpy as np
import cv2
import PIL
import os
import csv

DIR = "../train_data"
ANGLE = 90
DELAY = 100


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


num_image = len(os.listdir(DIR))

csvfile = open(DIR + "/annotation.csv", 'a', newline='')
csvout = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

mywin = "mywindow"
cv2.namedWindow(mywin)   # create a named window
cv2.moveWindow(mywin, 1400, 0)   # Move it to (40, 30)

w_size = pg.size()  # w_size[0]: width, w_size[1]: height
while True:
    img = PIL.ImageGrab.grab((0, 155, w_size[0], w_size[1] - 155))  # x_left, y_left, x_right, y_right
    img_frame = np.array(img)
    img_frame = cv2.cvtColor(img_frame, cv2.COLOR_RGB2BGR)

    # make img_frame straight
    if ANGLE == 0:
        pass
    elif ANGLE == 90:
        img_frame = rotate_image(img_frame, -90)
    elif ANGLE == 270:
        img_frame = rotate_image(img_frame, 90)

    img90 = rotate_image(img_frame, 90)
    img270 = rotate_image(img_frame, 270)

    # store three images
    images = [img_frame, img90, img270]
    angles = [0, 90, 270]
    for i in range(3):
        image_file = "a" + str(angles[i]) + "-" + str(num_image) + ".jpg"
        cv2.imwrite(DIR + "/" + image_file, images[i])
        csvout.writerow([image_file, angles[i]])
        num_image += 1

    cv2.imshow(mywin, img_frame)
    if cv2.waitKey(DELAY) == ord('q'):
        break
cv2.destroyAllWindows()
csvfile.close()