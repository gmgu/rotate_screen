import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
#from IPython.display import clear_output #for colab


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

img = cv2.imread("trash/image.jpg")
# img90 = rotate_image(img, 90)
# cv2.imwrite("image90.jpg", img90)

mywin = "mywindow"
cv2.namedWindow(mywin)   # create a named window
cv2.moveWindow(mywin, 1540, 100)   # Move it to (40, 30)

for i in range(0, 360, 30):
    img2 = rotate_image(img, i)
    img3 = resize_image(img2, 200, 200)
    cv2.imshow(mywin, img3)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()

    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)  # cv2 uses BGR, pyplot uses RGB
    #plt.imshow(img2)
    #plt.show()
    #time.sleep(2)
    #plt.clf()
    #clear_output() #for colab
