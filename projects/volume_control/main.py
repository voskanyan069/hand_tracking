#!/usr/bin/python3

import cv2
import numpy as np
import alsaaudio
import math
import sys

sys.path.append('../../modules')

import hand_tracking_module as htm

cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)

detector = htm.HandDetector(detection_con=0.7)

m = alsaaudio.Mixer()
m.setmute(0)
vol = m.getvolume()
vol_bar = 400

while True:
    ret, img = cam.read()
    img = cv2.flip(img, 1)
    img = detector.find_hands(img)
    lm_list = detector.find_position(img, draw=False)
    if len(lm_list) != 0:
        x1, y1 = lm_list[4][1], lm_list[4][2]
        x2, y2 = lm_list[8][1], lm_list[8][2]
        cx, cy = (x1+x2)//2, (y1+y2)//2

        cv2.circle(img, (x1,y1), 15, (255,0,0), cv2.FILLED)
        cv2.circle(img, (x2,y2), 15, (255,0,0), cv2.FILLED)
        cv2.line(img, (x1,y1), (x2,y2), (255,0,0), 3)
        cv2.circle(img, (cx,cy), 15, (255,0,0), cv2.FILLED)
        length = math.hypot(x2-x1, y2-y1)

        vol = np.interp(length, [50,300], [0,100])
        vol_bar = np.interp(length, [50,300], [400,150])
        m.setvolume(int(vol))
        cv2.putText(img, f'{int(vol)}%', (40,450), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255,0,0), 2)

        if length < 50:
            cv2.circle(img, (cx,cy), 15, (0,0,255), cv2.FILLED)

    cv2.rectangle(img, (50, 150), (85,400), (255,0,0), 3)
    if not isinstance(vol_bar, list):
        cv2.rectangle(img, (50, int(vol_bar)), (85,400), (255,0,0),
                cv2.FILLED)

    cv2.imshow('cam', img)
    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break
cam.release()
cv2.destroyAllWindows()
