# -*- coding: utf-8 -*-
"""
Created on Thu April 08 10:42:17 2019

@author: sokil
"""

import numpy as np
import cv2

cap=cv2.VideoCapture('2019_03_28_11_17_25_745.mp4')
cv2.namedWindow('live')

upperBody_cascade = cv2.CascadeClassifier('haarcascade_upperbody.xml')    

if cap.isOpened():
    ret, frame=cap.read()
else:
    ret=False
    
while ret:
    ret, frame=cap.read()
    arrUpperBody = upperBody_cascade.detectMultiScale(frame,scaleFactor = 1.1,minNeighbors = 1,minSize = (5,5),flags = cv2.CASCADE_SCALE_IMAGE)
    if arrUpperBody != ():
        for (x,y,w,h) in arrUpperBody:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        print ('body found')
        
    cv2.imshow("live", frame)
    if cv2.waitKey(1)==27:
        break
cv2.destroyAllWindows()
cap.release()