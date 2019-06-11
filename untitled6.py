# -*- coding: utf-8 -*-
"""
Created on Thu April 30 10:53:39 2019

@author: sokil
"""

import numpy as np
import cv2

def nothing(x):
    pass

cv2.namedWindow('check')

imgPath = '20190413.png';
src = cv2.imread(imgPath);
img=cv2.resize(src, dsize=(520, 640), interpolation=cv2.INTER_AREA)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV);

cv2.imshow("Original",img)
cv2.imshow("HSV",hsv)

cv2.createTrackbar('LOW_H','check',0,180,nothing)
cv2.createTrackbar('LOW_S','check',0,255,nothing)
cv2.createTrackbar('LOW_V','check',0,255,nothing)
cv2.createTrackbar('UP_H','check',0,180,nothing)
cv2.createTrackbar('UP_S','check',0,255,nothing)
cv2.createTrackbar('UP_V','check',0,255,nothing)


while(1):
    if cv2.waitKey(1) & 0xFF == 27:
        break
        
    LOW_H=cv2.getTrackbarPos('LOW_H','check')
    LOW_S=cv2.getTrackbarPos('LOW_S','check')
    LOW_V=cv2.getTrackbarPos('LOW_V','check')
    UP_H=cv2.getTrackbarPos('UP_H','check')
    UP_S=cv2.getTrackbarPos('UP_S','check')
    UP_V=cv2.getTrackbarPos('UP_V','check')


    low=(LOW_H,LOW_S,LOW_V)
    up=(UP_H,UP_S,UP_V)

    ch=cv2.inRange(hsv,low,up)
    
    
    cv2.imshow("check", ch)


cv2.destroyAllWindows()