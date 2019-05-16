# -*- coding: utf-8 -*-
"""
Created on Wed May 15 10:43:04 2019

@author: kimuk

지정된 hsv값에 대하여 이진
"""

import numpy as np
import cv2

img = cv2.imread('img.png')
hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

#green range
lower_green = np.array([55,0, 100])
upper_green = np.array([109, 150, 150])
	#blue range
lower_blue = np.array([36,56,144])
upper_blue = np.array([45,64,151])

	#Red range
lower_red = np.array([151,35,150])
upper_red = np.array([170,177,255])

	#white range
lower_white = np.array([110,0,200])
upper_white = np.array([150,25,255])

mask = cv2.inRange(hsv, lower_white, upper_white)
res = cv2.bitwise_and(img, img, mask=mask)
res_bgr = cv2.cvtColor(res,cv2.COLOR_HSV2BGR)
res_gray = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
kernel = np.ones((13,13),np.uint8)
thresh1 = cv2.threshold(res_gray,127,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
thresh2 = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)

im2,contours,hierarchy = cv2.findContours(thresh2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
for c in contours:
    x,y,w,h = cv2.boundingRect(c)	
    player_img = img[y:y+h,x:x+w]
    player_hsv = cv2.cvtColor(player_img,cv2.COLOR_BGR2HSV)
    cv2.imshow('player_hsv',player_hsv)
    
cv2.imshow('original', img)
cv2.imshow('res', res)
#cv2.imshow('thresh1', thresh1)
#cv2.imshow('thresh2', thresh2)
cv2.imshow('contour', im2)
cv2.waitKey(0)
cv2.destroyAllWindows()