import cv2
import numpy as np

while True:
    cap = cv2.VideoCapture("./face_expressions/sad_reverse.avi")
    if (cap.isOpened()== False):
        print("Error opening video file")

    while(cap.isOpened()):

        ret, frame = cap.read()
        if ret == True:

            cv2.imshow('Frame', frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        else:
            break
    cap.release()
