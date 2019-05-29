# -*- coding: utf-8 -*-
"""
Created on Tue May 28 20:31:43 2019

@author: kimmi
"""

import cv2
#from darkflow.net.build import TFNet
import matplotlib.pyplot as plt
import numpy as np

# define the model options and run
'''
options = {
    'model': 'cfg/yolo.cfg',
    'load': 'bin/yolo.weights',
    'threshold': 0.3,
    'gpu': 1.0
}

tfnet = TFNet(options)

img = cv2.imread('offsideimg.png', cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
result = tfnet.return_predict(img)
'''
img = cv2.imread('offsideimg.png', cv2.IMREAD_COLOR)
hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

lower_green = np.array([55,0, 100])
upper_green = np.array([109, 150, 255])

mask = cv2.inRange(hsv, lower_green, upper_green)
res = cv2.bitwise_and(img, img, mask=mask)
res_bgr = cv2.cvtColor(res,cv2.COLOR_HSV2BGR)
res_gray = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)

kernel = np.ones((13,13),np.uint8)
thresh = cv2.threshold(res_gray,127,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
im2,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

for c in contours:
    x,y,w,h=cv2.boundingRect(c)
    print(str(x)+" "+ str(y)+" "+str(w)+" "+str(h))


cv2.imshow('orginal',img)
cv2.imshow('thresh',thresh)
cv2.imshow('contours',im2)
cv2.waitKey(0)==27
cv2.destroyAllwindows()
