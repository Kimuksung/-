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
    'threshold': 0.2,
    'gpu': 1.0
}

average_inclination=0.0
team_a_x=[(9999,9999,9999)]
team_b_x=[(9999,9999,9999)]


tfnet = TFNet(options)

font = cv2.FONT_HERSHEY_SIMPLEX
lower_white = np.array([30,40,110])
upper_white = np.array([50,100,180])

lower_a = np.array([7,175,120])
upper_a = np.array([23,255,205])
lower_b = np.array([106,30,30])
upper_b = np.array([118,175,180])

#오프사이드
img=cv2.imread('soccer1.png')
max_x,max_y,t = img.shape
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
hist2 = cv2.calcHist([hsv], [1, 2], None, [256, 256], [0, 256, 0, 256])
mask_white=cv2.inRange(hsv,lower_white,upper_white)
result2 = cv2.bitwise_and(img,img,mask=mask_white)
gray = cv2.cvtColor(result2,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150)

lines = cv2.HoughLines(edges,1,np.pi/180,200)
i=1
inclination_arr=[]
for line in lines:
    rho,theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    inclination = (x2-x1)/(y2-y1)
        
    if(-5<inclination<5):
        inclination_arr.append(inclination)

temp=0.0
for a in inclination_arr:
    temp=temp+a
if(len(inclination_arr)!=0):
    average_inclination = temp/len(inclination_arr)
else:
    inclination_arr=0

#yolo detect
result = tfnet.return_predict(img)

for c in result:
    x=c['topleft']['x']
    y=c['topleft']['y']
    w=c['bottomright']['x']-x
    h=c['bottomright']['y']-y

    if(w<100):
        player_img = img[y:y+h,x:x+w]
        player_hsv = cv2.cvtColor(player_img,cv2.COLOR_BGR2HSV)
    
        mask1 = cv2.inRange(player_hsv, lower_a, upper_a)
        res1 = cv2.bitwise_and(player_img, player_img, mask=mask1)
        res1 = cv2.cvtColor(res1,cv2.COLOR_HSV2BGR)
        res1 = cv2.cvtColor(res1,cv2.COLOR_BGR2GRAY)
        ACount = cv2.countNonZero(res1)
        
        if(ACount >= 30):
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)
            cv2.putText(img, 'A', (x-2, y-2), font, 0.8, (255,0,0), 2, cv2.LINE_AA)
            if(average_inclination<0):
                temp = int(y-x/average_inclination)
                team_a_x.append((x,y,temp))
            else:
                temp = int(y-x/average_inclination)
                team_a_x.append((x+w,y,temp))

        mask2 = cv2.inRange(player_hsv, lower_b, upper_b)
        res2 = cv2.bitwise_and(player_img, player_img, mask=mask2)
        res2 = cv2.cvtColor(res2,cv2.COLOR_HSV2BGR)
        res2 = cv2.cvtColor(res2,cv2.COLOR_BGR2GRAY)
        BCount = cv2.countNonZero(res2)
        
        if(BCount >=30):
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),3)
            cv2.putText(img, 'B', (x-2, y-2), font, 0.8, (255,255,0), 2, cv2.LINE_AA)
            if(average_inclination<0):
                temp = int(y-x/average_inclination)
                team_b_x.append((x,y,temp))
            else:
                temp = int(y-x/average_inclination)
                team_b_x.append((x+w,y,temp))
                
if(average_inclination<0):                
    team_a_x.sort(key=lambda tup: tup[2])
    team_b_x.sort(key=lambda tup: tup[2])

    cv2.line(img,(int(-team_a_x[0][2]*average_inclination),0),(0,int(team_a_x[0][2])),(0,0,255),2)
    cv2.line(img,(int(-team_b_x[0][2]*average_inclination),0),(0,int(team_b_x[0][2])),(30,255,255),2)

else:
    team_a_x.sort(key=lambda tup: tup[2])
    team_b_x.sort(key=lambda tup: tup[2])
    
    cv2.line(img,(int((max_y-int(team_a_x[0][2]))*average_inclination),max_y),(0,int(team_a_x[0][2])),(0,0,255),2)
    cv2.line(img,(int((max_y-int(team_b_x[0][2]))*average_inclination),max_y),(0,int(team_b_x[0][2])),(30,255,255),2)
cv2.imshow('img',img)
cv2.waitKey(0)


