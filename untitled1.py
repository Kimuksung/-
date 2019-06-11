# -*- coding: utf-8 -*-
"""
Created on Thu April 01 10:35:58 2019

@author: sokil
"""

import cv2
import numpy as np

cap=cv2.VideoCapture('2019_03_28_11_17_25_745.mp4')
cv2.namedWindow('live')
total=0

if cap.isOpened():
    ret, frame=cap.read()
else:
    ret=False
    
while ret:
    ret, frame=cap.read()
    output=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    edges=cv2.Canny(output,400,200)
    lines=cv2.HoughLinesP(edges, rho=1,theta=np.pi/180.0,threshold=100)
    print('lines.shape=',lines.shape)
    
    for line in lines:
        x1,y1,x2,y2=line[0]
        cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),3)
        
        inclination=(y2-y1)/(x2-x1)
        
        if inclination>=1 or inclination<=-1:
            total+=inclination
            
    final=total/line
    print(final)
    
    
    
    cv2.imshow("canny",edges)
    cv2.imshow("live", frame)
    if cv2.waitKey(1)==27:
        break
cv2.destroyAllWindows()
cap.release()