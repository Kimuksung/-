# -*- coding: utf-8 -*-

import cv2
import numpy as np

img = cv2.imread('img.png')
img_original = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#blur = cv2.medianBlur(gray, 5)
#edges = cv2.Canny(gray,120,160,apertureSize=3)
#공 인식
'''
rows = blur.shape[0]
circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, rows / 8,param1=100, param2=30,minRadius=1, maxRadius=30)
if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv2.circle(img, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv2.circle(img, center, radius, (255, 0, 255), 3)
   ''' 

'''
#사람 인식
human_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
human=human_cascade.detectMultiScale(gray,1.1,4)

for(x,y,w,h) in human:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,220),3)

'''

#경기장 인식
edges = cv2.Canny(gray,400,450,apertureSize=3)
lines = cv2.HoughLines(edges,1,np.pi/180,100)

for i in range(len(lines)):
    for rho, theta in lines[i]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0+1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 -1000*(a))

        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
cv2.imshow('canny',edges)  
cv2.imshow('result',img)
cv2.waitKey(0)
cv2.destroyAllWindows()