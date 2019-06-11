# -*- coding: utf-8 -*-
"""
Created on Thu April 16 10:52:15 2019

@author: sokil
"""

import cv2
import numpy as np
cascade_file="haarcascade_frontalface_alt.xml "
cascade=cv2.CascadeClassifier(cascade_file)

cap=cv2.VideoCapture('2019_03_28_11_17_25_745.mp4')
windowName="live"
cv2.namedWindow(windowName)
if cap.isOpened():
    ret, frame=cap.read()
else:
    ret=False
    
while ret:
    ret, frame=cap.read()
    output=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    face_list=cascade.detectMultiScale(output,scaleFactor=1.1,minNeighbors=1,minSize=(150,150))
    if len(face_list)>0:
        print(face_list)
        color=(255,0,0)
        for face in face_list:
            x,y,w,h=face
            cv2.rectangle(frame,(x,y),(x+w,y+h),color,thickness=2)
    else:
        print("no face")
        
    
    cv2.imshow(windowName, frame)
    if cv2.waitKey(1)==27:
        break
cv2.destroyAllWindows()
cap.release()