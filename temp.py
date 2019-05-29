# -*- coding: utf-8 -*-
"""
Created on Tue May 28 23:35:38 2019

@author: kimmi
"""

import cv2
from darkflow.net.build import TFNet
import numpy as np

options = {
    'model': 'cfg/yolo.cfg',
    'load': 'bin/yolo.weights',
    'threshold': 0.3,
    'gpu': 1.0
}

tfnet = TFNet(options)

font = cv2.FONT_HERSHEY_SIMPLEX
lower_white = np.array([110,0,200])
upper_white = np.array([150,25,255])
lower_red = np.array([151,35,150])
upper_red = np.array([170,177,255])

lower_yello = np.array([18,70,240])
upper_yello = np.array([32,255,255])
lower_argen = np.array([96,15,139])
upper_argen = np.array([115,85,255])

lower_brazil = np.array([25,160,111])
upper_brazil = np.array([35,255,226])
lower_german = np.array([0,50,38])
upper_german = np.array([10,151,148])

img=cv2.imread('GB.png')
result = tfnet.return_predict(img)

for c in result:
    x=c['topleft']['x']
    y=c['topleft']['y']
    w=c['bottomright']['x']-x
    h=c['bottomright']['y']-y
    
    player_img = img[y:y+h,x:x+w]
    player_hsv = cv2.cvtColor(player_img,cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(player_hsv, lower_brazil, upper_brazil)
    res1 = cv2.bitwise_and(player_img, player_img, mask=mask1)
    res1 = cv2.cvtColor(res1,cv2.COLOR_HSV2BGR)
    res1 = cv2.cvtColor(res1,cv2.COLOR_BGR2GRAY)
    ACount = cv2.countNonZero(res1)
    if(ACount >= 100):
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)
        cv2.putText(img, 'A', (x-2, y-2), font, 0.8, (255,0,0), 2, cv2.LINE_AA)

    mask2 = cv2.inRange(player_hsv, lower_german, upper_german)
    res2 = cv2.bitwise_and(player_img, player_img, mask=mask2)
    res2 = cv2.cvtColor(res2,cv2.COLOR_HSV2BGR)
    res2 = cv2.cvtColor(res2,cv2.COLOR_BGR2GRAY)
    BCount = cv2.countNonZero(res2)
    if(BCount >= 100):
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),3)
        cv2.putText(img, 'B', (x-2, y-2), font, 0.8, (255,255,0), 2, cv2.LINE_AA)
    print(BCount)
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllwindows()

