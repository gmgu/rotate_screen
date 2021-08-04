# For each video file
#   Capture screen for each 0.1 sec
# For each captured screen
#   generate 36 rotated screen with label (rotation)
# Save generated labeled-screen
#   directory (label) - files (images)

import numpy as np
import os
import cv2

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


filenames0 = ["0-1.mp4", "0-2.mp4", "0-3.mp4", "0-4.mp4"]
filenames90 = ["90-1.mp4", "90-2.mp4", "90-3.mp4", "90-4.mp4"]

#0 rotated videos
for file in filenames0:
    video = cv2.VideoCapture("rawdata/" + file)
    fps = video.get(cv2.CAP_PROP_FPS) #frame per second
    fpm = round(fps * 0.1) #frame per millisecond

    num_frame = 0
    while True:
        ret, frame = video.read()
        num_frame += 1
        if ret:
            if num_frame % fpm == 0: #Capture screen for each 0.1 sec
                # print("save screen at", num_frame)
                for angle in range(0, 360, 10):
                    rframe = rotate_image(frame, angle) #generate 36 rotated screen
                    if not os.path.exists("data/" + str(angle)):
                        os.makedirs("data/" + str(angle))
                    cv2.imwrite("data/" + str(angle) + "/" + file.replace(".mp4", "-") + str(num_frame) + ".jpg", rframe)
        else:
            break

    video.release()

#90 rotated videos
for file in filenames90:
    video = cv2.VideoCapture("rawdata/" + file)
    fps = video.get(cv2.CAP_PROP_FPS) #frame per second
    fpm = round(fps * 0.1) #frame per millisecond

    num_frame = 0
    while True:
        ret, frame = video.read()
        num_frame += 1
        if ret:
            if num_frame % fpm == 0: #Capture screen for each 0.1 sec
                # print("save screen at", num_frame)
                for angle in range(0, 360, 10):
                    rframe = rotate_image(frame, angle) #generate 36 rotated screen
                    angle = (angle + 90) % 360
                    if not os.path.exists("data/" + str(angle)):
                        os.makedirs("data/" + str(angle))
                    cv2.imwrite("data/" + str(angle) + "/" + file.replace(".mp4", "-") + str(num_frame) + ".jpg", rframe)
        else:
            break

    video.release()