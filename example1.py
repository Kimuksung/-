# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 15:21:17 2019

@author: kimuk
"""

import cv2
import numpy as np

cap=cv2.VideoCapture('offside.mp4')
human_cascade = cv2.CascadeClassifier('haarcascade_frontalcatface.xml')

while True:
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    human=human_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=2,minSize=(25,25))
    for(x,y,w,h) in human:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,220),3)
    cv2.imshow('video',frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break;
'''
while True:
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    
    rows = blur.shape[0]
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, rows / 8,param1=100, param2=30,minRadius=1, maxRadius=30)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv2.circle(frame, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv2.circle(frame, center, radius, (255, 0, 255), 3)
            cv2.imshow('video',frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break;
'''
cap.release()
cv2.destroyAllWindows()