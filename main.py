import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
#from IPython.display import clear_output #for colab


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)  # get rotation matrix
    result = cv2.warpAffine(image, rot_mat, (0, 0))  # rotate image
    return result


img = cv2.imread("image.jpg")
# img90 = rotate_image(img, 90)
# cv2.imwrite("image90.jpg", img90)

mywin = "mywindow"
cv2.namedWindow(mywin)   # create a named window
cv2.moveWindow(mywin, 1540, 100)   # Move it to (40, 30)

for i in range(0, 360, 30):
    img2 = rotate_image(img, i)
    cv2.imshow(mywin, img2)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()

    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)  # cv2 uses BGR, pyplot uses RGB
    #plt.imshow(img2)
    #plt.show()
    #time.sleep(2)
    #plt.clf()
    #clear_output() #for colab
