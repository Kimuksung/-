# -*- coding: utf-8 -*-
"""
Created on Thu May 16 12:10:56 2019

@author: sokil
"""

import cv2
import numpy as np

cap=cv2.VideoCapture('2019_03_28_11_17_25_745.mp4')
if(not cap.isOpened()):
    print('Error opening video')
    
height, width=(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
               int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))

bgMog1=cv2.createBackgroundSubtractorMOG2()
bgMog2=cv2.createBackgroundSubtractorMOG2(varThreshold=25,detectShadows=False)

bgKnn1=cv2.createBackgroundSubtractorKNN()
bgKnn2=cv2.createBackgroundSubtractorKNN(dist2Threshold=1000,detectShadows=False)

AREA_TH=80
def findObjectAndDraw(blmage, src):
    res=src.copy()
    blmage=cv2.erode(blmage,None,5)
    blmage=cv2.dilate(blmage,None,5)
    blmage=cv2.erode(blmage,None,7)
    contours, _=cv2.findContours(blmage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(src, contours, -1,(255,0,0), 1)
    for i, cnt in enumerate(contours):
        area=cv2.contourArea(cnt)
        if area>AREA_TH:
            x,y,width,height=cv2.boundingRect(cnt)
            cv2.rectangle(res,(x,y),(x+width,y+height),(0,0,255),2)
    return res

t=0
while True:
    ret, frame=cap.read()
    if not ret:
        break
    t+=1
    print('t=',t)
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur=cv2.GaussianBlur(frame,(5,5),0.0)
    
    blmage1=bgMog1.apply(blur)
    blmage2=bgMog2.apply(blur)
    blmage3=bgKnn1.apply(blur)
    blmage4=bgKnn2.apply(blur)
    dst1=findObjectAndDraw(blmage1,frame)
    dst2=findObjectAndDraw(blmage2,frame)
    dst3=findObjectAndDraw(blmage3,frame)
    dst4=findObjectAndDraw(blmage4,frame)
    
    cv2.imshow('blmage1',blmage1)
    cv2.imshow('bgMog1',dst1)
    cv2.imshow('blmage2',blmage2)
    cv2.imshow('bgMog2',dst2)
    cv2.imshow('blmage3',blmage3)
    cv2.imshow('bgKnn1',dst3)
    cv2.imshow('blmage4',blmage4)
    cv2.imshow('bgKnn2',dst4)
        
    key=cv2.waitKey(27)
    if key==27:
        break
            
if cap.isOpened():
    cap.release()
cv2.destroyAllWindows()