# -*- coding: utf-8 -*-
"""
Created on Thu April 01 10:40:14 2019

@author: sokil
"""

import cv2
import numpy as np

cap=cv2.VideoCapture('2019_03_28_11_17_25_745.mp4')
cv2.namedWindow('live')

if cap.isOpened():
    ret, frame=cap.read()
else:
    ret=False
    
while ret:
    ret, frame=cap.read()
    output=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    low=(120,120,120)
    up=(255,255,255)
    
    check=cv2.inRange(output,low,up)
    
    
    
    cv2.imshow("HSV",output)
    cv2.imshow("live", frame)
    cv2.imshow("check", check)
    if cv2.waitKey(1)==27:
        break
cv2.destroyAllWindows()
cap.release()