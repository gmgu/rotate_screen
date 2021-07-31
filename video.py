import numpy as np
import cv2
import time
from ffpyplayer.player import MediaPlayer
file_name = ""
video_path = "D:\\Users\\gmgu\\Downloads\\" + file_name
cap = cv2.VideoCapture(video_path)
#sound = MediaPlayer(video_path)

fps = cap.get(cv2.CAP_PROP_FPS) #frame per second
delay = int(1000 / fps)

while True:
    ret, frame = cap.read()
    cv2.imshow('frame', frame)

    if cv2.waitKey(delay) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
