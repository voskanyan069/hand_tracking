#!/usr/bin/python3

import cv2
import numpy as np
import autopy
import time
import sys

sys.path.append('../../modules')

import hand_tracking_module as htm

cam_w = 640
cam_h = 480
frame_r = 100
smoothening = 7
prev_loc_x, prev_loc_y = 0, 0
current_loc_x, current_loc_y = 0, 0

cam = cv2.VideoCapture(0)
cam.set(3, cam_w)
cam.set(4, cam_h)

detector = htm.HandDetector(detection_con=0.7, max_hands=1)
screen_w, screen_h = autopy.screen.size()
prev_time = 0

while True:
    ret, img = cam.read()
    img = cv2.flip(img, 1)
    img = detector.find_hands(img)
    lm_list, mouse_box = detector.find_position(img, draw=False, draw_box=True)
    if len(lm_list) != 0:
        x1, y1 = lm_list[8][1:]
        x2, y2 = lm_list[12][1:]
        fingers = detector.fingers_up()
        cv2.rectangle(img, (frame_r,frame_r), (cam_w-frame_r,cam_h-frame_r),
                (255,0,255), 2)
        
        if fingers[1] == 1 and fingers[2] == 0:
            x3 = np.interp(x1, (frame_r,cam_w-frame_r), (0, screen_w))
            y3 = np.interp(y1, (frame_r,cam_h-frame_r), (0, screen_h))
            cv2.circle(img, (x1,y1), 15, (0,0,255), cv2.FILLED)

            current_loc_x = prev_loc_x + (x3-prev_loc_x) / smoothening
            current_loc_y = prev_loc_y + (y3-prev_loc_y) / smoothening

            if x3 > 0 and x3 < screen_w and y3 > 0 and y3 < screen_h:
                autopy.mouse.move(current_loc_x, current_loc_y)
            prev_loc_x, prev_loc_y = current_loc_x, current_loc_y
        if fingers[1] == 1 and fingers[2] == 1:
            length, img, line_info = detector.find_distance(8, 12, img)
            if length < 40:
               cv2.circle(img, (line_info[4], line_info[5]), 10,
                       (0,255,0), cv2.FILLED)
               autopy.mouse.click()

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
