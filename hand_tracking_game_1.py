#!/usr/bin/python3

import cv2
import mediapipe as mp
import time
import hand_tracking_module as htm

cam = cv2.VideoCapture(0)

prev_time = 0
current_time = 0
detector = htm.HandDetector()

while True:
    ret, img = cam.read()
    img = cv2.flip(img, 1)
    img = detector.find_hands(img)
    lm_list = detector.find_position(img)
    if len(lm_list) != 0:
        print(lm_list[4])

    current_time = time.time()
    fps = 1/(current_time-prev_time)
    prev_time = current_time

    cv2.putText(img, str(int(fps)), (0,30), cv2.FONT_HERSHEY_SIMPLEX, 1,
            (0,0,255), 2)
    cv2.imshow('cam', img)
    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break
cam.release()
cv2.destroyAllWindows()
